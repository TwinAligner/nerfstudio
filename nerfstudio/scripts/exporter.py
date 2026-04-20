# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script for exporting NeRF into other formats.
"""


from __future__ import annotations

from nerfstudio.utils import env
env.set_env_variables()

import json
import os
import sys
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import open3d as o3d
import torch
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.datamanagers.random_cameras_datamanager import RandomCamerasDataManager
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import collect_camera_poses, generate_point_cloud, get_mesh_from_filename
from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
from nerfstudio.fields.sdf_field import SDFField  # noqa
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.models.splatfacto_art import SplatfactoArtModel
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path = None
    """Path to the output directory."""
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = self.load_config.parent


def validate_pipeline(normal_method: str, normal_output_name: str, pipeline: Pipeline) -> None:
    """Check that the pipeline is valid for this exporter.

    Args:
        normal_method: Method to estimate normals with. Either "open3d" or "model_output".
        normal_output_name: Name of the normal output.
        pipeline: Pipeline to evaluate with.
    """
    if normal_method == "model_output":
        CONSOLE.print("Checking that the pipeline has a normal output.")
        origins = torch.zeros((1, 3), device=pipeline.device)
        directions = torch.ones_like(origins)
        pixel_area = torch.ones_like(origins[..., :1])
        camera_indices = torch.zeros_like(origins[..., :1])
        ray_bundle = RayBundle(
            origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
        )
        outputs = pipeline.model(ray_bundle)
        if normal_output_name not in outputs:
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{normal_output_name}' not found in pipeline outputs.")
            CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
            CONSOLE.print(
                "[bold yellow]Warning: Please train a model with normals "
                "(e.g., nerfacto with predicted normals turned on)."
            )
            CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
            CONSOLE.print("[bold yellow]Exiting early.")
            sys.exit(1)


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = False
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(
            pipeline.datamanager,
            (VanillaDataManager, ParallelDataManager, FullImageDatamanager, RandomCamerasDataManager),
        )
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        crop_obb = None
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        if self.save_world_frame:
            # apply the inverse dataparser transform to the point cloud
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
            poses[:, :3, 3] = points
            poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            points = poses[:, :3, 3].numpy()
            pcd.points = o3d.utility.Vector3dVector(points)

        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


@dataclass
class ExportTSDFMesh(Exporter):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 2
    """Downscale the images starting from the resolution used for training."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [128, 128, 128])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""
    use_bounding_box: bool = True
    """Whether to use a bounding box for the TSDF volume."""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    texture_method: Literal["tsdf", "nerf"] = "tsdf"
    """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        tsdf_utils.export_tsdf_mesh(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            self.depth_output_name,
            self.rgb_output_name,
            self.resolution,
            self.batch_size,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
        )

        # possibly
        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the tsdf export
            mesh = get_mesh_from_filename(
                str(self.output_dir / "tsdf_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportPoissonMesh(Exporter):
    """
    Export a mesh using poisson surface reconstruction.
    """

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    save_point_cloud: bool = False
    """Whether to save the point cloud."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    texture_method: Literal["point_cloud", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'point_cloud' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(
            pipeline.datamanager,
            (VanillaDataManager, ParallelDataManager, FullImageDatamanager, RandomCamerasDataManager),
        )
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        else:
            crop_obb = None

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()
        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")

        if self.save_point_cloud:
            CONSOLE.print("Saving Point Cloud...")
            o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")

        CONSOLE.print("Computing Mesh... this may take a while.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

        CONSOLE.print("Saving Mesh...")
        o3d.io.write_triangle_mesh(str(self.output_dir / "poisson_mesh.ply"), mesh)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")

        # This will texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the poisson reconstruction
            mesh = get_mesh_from_filename(
                str(self.output_dir / "poisson_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportMarchingCubesMesh(Exporter):
    """Export a mesh using marching cubes."""

    isosurface_threshold: float = 0.0
    """The isosurface threshold for extraction. For SDF based methods the surface is the zero level set."""
    resolution: int = 1024
    """Marching cube resolution."""
    simplify_mesh: bool = False
    """Whether to simplify the mesh."""
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Minimum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Maximum of the bounding box."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Main function."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        # TODO: Make this work with Density Field
        assert hasattr(pipeline.model.config, "sdf_field"), "Model must have an SDF field."

        CONSOLE.print("Extracting mesh with marching cubes... which may take a while")

        assert self.resolution % 512 == 0, f"""resolution must be divisible by 512, got {self.resolution}.
        This is important because the algorithm uses a multi-resolution approach
        to evaluate the SDF where the minimum resolution is 512."""

        # Extract mesh using marching cubes for sdf at a multi-scale resolution.
        multi_res_mesh = generate_mesh_with_multires_marching_cubes(
            geometry_callable_field=lambda x: cast(SDFField, pipeline.model.field)
            .forward_geonetwork(x)[:, 0]
            .contiguous(),
            resolution=self.resolution,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            isosurface_threshold=self.isosurface_threshold,
            coarse_mask=None,
        )
        filename = self.output_dir / "sdf_marching_cubes_mesh.ply"
        multi_res_mesh.export(filename)

        # load the mesh from the marching cubes export
        mesh = get_mesh_from_filename(str(filename), target_num_faces=self.target_num_faces)
        CONSOLE.print("Texturing mesh with NeRF...")
        texture_utils.export_textured_mesh(
            mesh,
            pipeline,
            self.output_dir,
            px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
            unwrap_method=self.unwrap_method,
            num_pixels_per_side=self.num_pixels_per_side,
        )


@dataclass
class ExportCameraPoses(Exporter):
    """
    Export camera poses to a .json file.
    """

    optimized: bool = False
    """Apply camera optimizer deltas before exporting poses."""
    world_frame: bool = False
    """Export poses in original world coordinates (inverse of dataparser transform)."""

    def main(self) -> None:
        """Export camera poses"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)
        assert isinstance(pipeline, VanillaPipeline)
        # Check if any frames were filtered during training
        if hasattr(pipeline.datamanager, 'filtered_train_indices') and pipeline.datamanager.filtered_train_indices is not None:
            num_filtered = len(pipeline.datamanager.filtered_train_indices)
            CONSOLE.print(f"[bold yellow]Note: {num_filtered} training frames were filtered during training and will be excluded from export.")
        
        train_frames, eval_frames = collect_camera_poses(pipeline, optimized=self.optimized, world_frame=self.world_frame)

        for file_name, frames in [("transforms_train.json", train_frames), ("transforms_eval.json", eval_frames)]:
            if len(frames) == 0:
                CONSOLE.print(f"[bold yellow]No frames found for {file_name}. Skipping.")
                continue

            output_file_path = os.path.join(self.output_dir, file_name)

            with open(output_file_path, "w", encoding="UTF-8") as f:
                json.dump(frames, f, indent=4)

            CONSOLE.print(f"[bold green]:white_check_mark: Saved poses to {output_file_path}")
        
        # Export object poses for TwoPartBlendModel
        self._export_object_poses(pipeline)
    
    def _export_object_poses(self, pipeline) -> None:
        """Export optimized object poses for TwoPartBlendModel"""
        from nerfstudio.models.twopart_blend import TwoPartBlendModel
        from nerfstudio.cameras.lie_groups import exp_map_SO3xR3
        
        model = pipeline.model
        if not isinstance(model, TwoPartBlendModel):
            return
        
        if not hasattr(model, '_T_obj_base') or not hasattr(model, 'obj_pose_adjustment'):
            CONSOLE.print("[bold yellow]Model does not have object pose attributes. Skipping object pose export.")
            return
        
        with torch.no_grad():
            # Get base object poses and adjustments
            T_obj_base = model._T_obj_base  # [N, 4, 4] in normalized (dataparser) space
            obj_adj = model.obj_pose_adjustment  # [N, 6]
            
            n_frames = T_obj_base.shape[0]
            
            # Pre-fetch dataparser transform for mapping to original world space
            dp_outputs = getattr(pipeline.datamanager, 'train_dataparser_outputs', None)
            R_dp = None
            t_dp = None
            s_dp = None
            if self.world_frame and dp_outputs is not None and hasattr(dp_outputs, 'dataparser_transform'):
                dp_tf = dp_outputs.dataparser_transform.detach().cpu().numpy()  # [3,4]
                R_dp = dp_tf[:, :3].astype(np.float32)
                t_dp = dp_tf[:, 3].astype(np.float32)
                s_dp = float(dp_outputs.dataparser_scale)
                R_dp_inv = np.linalg.inv(R_dp)
            
            # Compute optimized object poses per frame
            T_obj_list = []  # list of 3x4 in desired output frame
            for i in range(n_frames):
                # Convert se(3) 6D vector to SE(3) transformation [3, 4]
                adj_3x4 = exp_map_SO3xR3(obj_adj[i:i+1]).squeeze(0)  # [3, 4]
                
                # Convert to [4, 4]
                Adj = torch.eye(4, device=T_obj_base.device, dtype=T_obj_base.dtype)
                Adj[:3, :4] = adj_3x4
                
                # Apply adjustment: T_ns = T_obj_base @ Adj  (normalized coords)
                T_ns = (T_obj_base[i] @ Adj).detach().cpu().numpy().astype(np.float32)  # [4,4]
                R_ns = T_ns[:3, :3]
                t_ns = T_ns[:3, 3]
                
                if self.world_frame and R_dp is not None:
                    # Map normalized transform to original world coordinates
                    # x_ns = s * (R_dp x_w + t_dp)
                    # x_ns' = R_ns x_ns + t_ns
                    # => x_w' = R_dp^{-1} R_ns R_dp x_w + R_dp^{-1}(R_ns t_dp + t_ns / s - t_dp)
                    R_w = R_dp_inv @ (R_ns @ R_dp)
                    t_w = R_dp_inv @ (R_ns @ t_dp + (t_ns / s_dp) - t_dp)
                    T_out = np.zeros((3, 4), dtype=np.float32)
                    T_out[:, :3] = R_w
                    T_out[:, 3] = t_w
                else:
                    # Keep in normalized space
                    T_out = T_ns[:3, :]
                
                T_obj_list.append(T_out.tolist())
            
            # Save to JSON files
            train_dataset = pipeline.datamanager.train_dataset
            eval_dataset = pipeline.datamanager.eval_dataset
            
            # Determine which frames are train vs eval
            n_train = len(train_dataset.image_filenames) if train_dataset else 0
            n_eval = len(eval_dataset.image_filenames) if eval_dataset else 0
            
            if n_train > 0:
                # Exclude filtered training frames if available
                filtered = getattr(pipeline.datamanager, 'filtered_train_indices', None)
                filtered_set = set(filtered.tolist()) if (filtered is not None) else set()
                # Ensure frame 0 is never excluded during export
                filtered_set.discard(0)
                include_indices = [i for i in range(min(n_train, len(T_obj_list))) if i not in filtered_set]
                train_obj_poses = [{"frame_index": i, "T_obj": T_obj_list[i]} for i in include_indices]
                train_output_path = os.path.join(self.output_dir, "object_poses_train.json")
                with open(train_output_path, "w", encoding="UTF-8") as f:
                    json.dump(train_obj_poses, f, indent=4)
                CONSOLE.print(f"[bold green]:white_check_mark: Saved {len(train_obj_poses)} object poses to {train_output_path}")
            
            if n_eval > 0 and n_train + n_eval <= len(T_obj_list):
                eval_obj_poses = [{"frame_index": i, "T_obj": T_obj_list[n_train + i]} for i in range(n_eval)]
                eval_output_path = os.path.join(self.output_dir, "object_poses_eval.json")
                with open(eval_output_path, "w", encoding="UTF-8") as f:
                    json.dump(eval_obj_poses, f, indent=4)
                CONSOLE.print(f"[bold green]:white_check_mark: Saved {len(eval_obj_poses)} object poses to {eval_output_path}")


@dataclass
class ExportGaussianSplat(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    # New option: export splats in original world coordinates (inverse of dataparser transform)
    world_frame: bool = False
    """Export splats in original world coordinates (inverse of dataparser transform)."""

    @staticmethod
    def write_ply(
        filename: str,
        count: int,
        map_to_tensors: typing.OrderedDict[str, np.ndarray],
    ):
        """
        Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
        Note: All float values will be converted to float32 for writing.

        Parameters:
        filename (str): The name of the file to write.
        count (int): The number of vertices to write.
        map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
            Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
        """

        # Ensure count matches the length of all tensors
        if not all(len(tensor) == count for tensor in map_to_tensors.values()):
            raise ValueError("Count does not match the length of all tensors")

        # Type check for numpy arrays of type float or uint8 and non-empty
        if not all(
            isinstance(tensor, np.ndarray)
            and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
            and tensor.size > 0
            for tensor in map_to_tensors.values()
        ):
            raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

        with open(filename, "wb") as ply_file:
            # Write PLY header
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")

            ply_file.write(f"element vertex {count}\n".encode())

            # Write properties, in order due to OrderedDict
            for key, tensor in map_to_tensors.items():
                data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())

            ply_file.write(b"end_header\n")

            # Write binary data
            # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
            for i in range(count):
                for tensor in map_to_tensors.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    elif tensor.dtype == np.uint8:
                        ply_file.write(value.tobytes())
        CONSOLE.print(f"[bold green]:white_check_mark: Saved {count} gaussians to {filename}")

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model
        is_splatfacto_art = isinstance(model, SplatfactoArtModel)
        filename = self.output_dir / "object_3dgs.ply"
        # source_urdf_filename = pipeline.datamanager.dataparser.config.resume_sdfstudio_dir / "object.urdf"
        # source_part_filepath = pipeline.datamanager.dataparser.config.resume_sdfstudio_dir / "parts"
        # if source_urdf_filename.exists() and source_part_filepath.exists():
        #     parts_dirname = pipeline.datamanager.dataparser.config.resume_sdfstudio_dir / "parts"
        #     part_idx = pipeline.datamanager.dataparser.config.reconstruct_part_id
        #     filename = parts_dirname / f"part_{part_idx}_3dgs.ply"
        count = 0
        map_to_tensors = OrderedDict()

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            # Also fetch scales and quats early so they can be transformed consistently if needed
            scales = model.scales.data.cpu().numpy()
            quats = model.quats.data.cpu().numpy()  # XYZW in model
            # Convert XYZW (model) -> WXYZ (PLY convention)
            quats_wxyz = np.stack([quats[:, 3], quats[:, 0], quats[:, 1], quats[:, 2]], axis=1)
            
            # For SplatfactoArtModel: Apply rotation_offset and first-frame joint transform to motion part
            if is_splatfacto_art:
                # Use full SE(3) offset (rotation + translation)
                T_offset = model.rotation_offset_se3.cpu().numpy()  # [4, 4]
                R_offset = T_offset[:3, :3]
                t_offset = T_offset[:3, 3]
                
                # Check if rotation offset is non-trivial (not identity)
                rotation_offset_angle = np.arccos(np.clip((np.trace(R_offset) - 1.0) / 2.0, -1.0, 1.0))
                
                part_weights = model.part_weights.squeeze(-1).cpu().numpy()  # [N]
                motion_part = int(model.motion_part)
                
                if motion_part == 1:
                    motion_mask = part_weights > 0.5
                else:
                    motion_mask = part_weights <= 0.5
                
                CONSOLE.print(f"[cyan]Applying rotation offset ({np.degrees(rotation_offset_angle):.2f}°) to {motion_mask.sum()}/{len(motion_mask)} motion part gaussians")
                

                # Apply rotation offset SE3: p' = R p + t
                pos_subset = positions[motion_mask]
                positions[motion_mask] = (R_offset @ pos_subset.T).T + t_offset[None, :]
                
                def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
                    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
                    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
                    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
                    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
                    trace = m00 + m11 + m22
                    if trace > 0.0:
                        s = 0.5 / np.sqrt(trace + 1.0)
                        w = 0.25 / s
                        x = (m21 - m12) * s
                        y = (m02 - m20) * s
                        z = (m10 - m01) * s
                    else:
                        if m00 > m11 and m00 > m22:
                            s = 2.0 * np.sqrt(max(1.0 + m00 - m11 - m22, 0.0))
                            w = (m21 - m12) / s
                            x = 0.25 * s
                            y = (m01 + m10) / s
                            z = (m02 + m20) / s
                        elif m11 > m22:
                            s = 2.0 * np.sqrt(max(1.0 + m11 - m00 - m22, 0.0))
                            w = (m02 - m20) / s
                            x = (m01 + m10) / s
                            y = 0.25 * s
                            z = (m12 + m21) / s
                        else:
                            s = 2.0 * np.sqrt(max(1.0 + m22 - m00 - m11, 0.0))
                            w = (m10 - m01) / s
                            x = (m02 + m20) / s
                            y = (m12 + m21) / s
                            z = 0.25 * s
                    q = np.array([w, x, y, z], dtype=np.float32)
                    q /= np.linalg.norm(q) + 1e-8
                    return q
                
                def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
                    """Hamilton product for quaternions in [w, x, y, z] order."""
                    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
                    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
                    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                    out = np.stack([w, x, y, z], axis=-1)
                    out /= np.linalg.norm(out, axis=-1, keepdims=True) + 1e-8
                    return out
                
                q_offset = rotmat_to_quat_wxyz(R_offset)  # [4] wxyz
                quats_motion = quats_wxyz[motion_mask]  # [M, 4] wxyz
                quats_wxyz[motion_mask] = quat_mul_wxyz(np.broadcast_to(q_offset, quats_motion.shape), quats_motion)

                # Apply first-frame joint transform
                joint_type = model.joint_type
                axis = model.joint_axis.detach().cpu().numpy().astype(np.float32)  # normalized
                origin = model.joint_origin.detach().cpu().numpy().astype(np.float32)
                param0 = float(model.joint_params[0].detach().cpu().numpy())

                def axis_angle_to_rotmat(a: np.ndarray, theta: float) -> np.ndarray:
                    a = a / (np.linalg.norm(a) + 1e-8)
                    x, y, z = a[0], a[1], a[2]
                    c = np.cos(theta)
                    s = np.sin(theta)
                    C = 1.0 - c
                    return np.array([
                        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
                    ], dtype=np.float32)

                if joint_type == "revolute":
                    # Positions: p'' = R_joint (p' - o) + o
                    R_joint = axis_angle_to_rotmat(axis, param0)
                    pos_subset = positions[motion_mask]
                    positions[motion_mask] = (R_joint @ (pos_subset - origin[None, :]).T).T + origin[None, :]

                    # Orientations: q'' = q_R * q'
                    q_R = rotmat_to_quat_wxyz(R_joint)
                    quats_motion = quats_wxyz[motion_mask]
                    quats_wxyz[motion_mask] = quat_mul_wxyz(np.broadcast_to(q_R, quats_motion.shape), quats_motion)
                elif joint_type == "prismatic":
                    # Positions: p'' = p' + axis * distance
                    pos_subset = positions[motion_mask]
                    positions[motion_mask] = pos_subset + axis[None, :] * param0
                    # Orientations unchanged beyond q_offset

            # If requested, convert from normalized (dataparser) space back to original world frame
            if self.world_frame:
                dp_outputs = pipeline.datamanager.train_dataparser_outputs
                # dataparser_transform is a 3x4 matrix [R|t] applied in preprocessing; scale is a scalar s
                dp_tf = dp_outputs.dataparser_transform.cpu().numpy()
                R_dp = dp_tf[:, :3]
                t_dp = dp_tf[:, 3]
                s = float(dp_outputs.dataparser_scale)
                # Invert the similarity transform for positions: x_world = R^-1 * (x_norm / s - t)
                R_inv = np.linalg.inv(R_dp)
                positions = (R_inv @ ((positions / s).T - t_dp[:, None])).T
                # Scales are lengths -> divide by s
                scales = scales - np.log(s)

                # Rotate local gaussian orientations into world axes: Q_world = R_inv @ Q_norm
                # Implement via quaternion left-multiply by q(R_inv) in wxyz convention
                def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
                    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
                    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
                    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
                    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
                    trace = m00 + m11 + m22
                    if trace > 0.0:
                        s = 0.5 / np.sqrt(trace + 1.0)
                        w = 0.25 / s
                        x = (m21 - m12) * s
                        y = (m02 - m20) * s
                        z = (m10 - m01) * s
                    else:
                        if m00 > m11 and m00 > m22:
                            s = 2.0 * np.sqrt(max(1.0 + m00 - m11 - m22, 0.0))
                            w = (m21 - m12) / s
                            x = 0.25 * s
                            y = (m01 + m10) / s
                            z = (m02 + m20) / s
                        elif m11 > m22:
                            s = 2.0 * np.sqrt(max(1.0 + m11 - m00 - m22, 0.0))
                            w = (m02 - m20) / s
                            x = (m01 + m10) / s
                            y = 0.25 * s
                            z = (m12 + m21) / s
                        else:
                            s = 2.0 * np.sqrt(max(1.0 + m22 - m00 - m11, 0.0))
                            w = (m10 - m01) / s
                            x = (m02 + m20) / s
                            y = (m12 + m21) / s
                            z = 0.25 * s
                    q = np.array([w, x, y, z], dtype=np.float32)
                    # Normalize to be safe
                    q /= np.linalg.norm(q) + 1e-8
                    return q

                def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
                    """Hamilton product for quaternions in [w, x, y, z] order. Supports broadcasting over first dim."""
                    # q1, q2 shapes: (..., 4)
                    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
                    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
                    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                    out = np.stack([w, x, y, z], axis=-1)
                    # Normalize
                    out /= np.linalg.norm(out, axis=-1, keepdims=True) + 1e-8
                    return out

                q_Rinv = rotmat_to_quat_wxyz(R_inv)  # WXYZ
                quats_wxyz = quat_mul_wxyz(np.broadcast_to(q_Rinv, quats_wxyz.shape), quats_wxyz)

            count = positions.shape[0]
            n = count
            map_to_tensors["x"] = positions[:, 0]
            map_to_tensors["y"] = positions[:, 1]
            map_to_tensors["z"] = positions[:, 2]
            map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

            if model.config.sh_degree > 0:
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
            else:
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]
                map_to_tensors["red"] = (colors[:, 0] * 255).astype(np.uint8)
                map_to_tensors["green"] = (colors[:, 1] * 255).astype(np.uint8)
                map_to_tensors["blue"] = (colors[:, 2] * 255).astype(np.uint8)
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            # Use (possibly transformed) scales and quats
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            # Normalize quaternions for numerical safety
            quats_wxyz = quats_wxyz / (np.linalg.norm(quats_wxyz, axis=1, keepdims=True) + 1e-8)
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats_wxyz[:, i, None]
            
            # Get part_weights if available (for TwoPartBlendModel)
            part_weights_np = None
            if hasattr(model, 'part_weights'):
                part_weights_np = model.part_weights.squeeze(-1).cpu().numpy().astype(np.float32)

            if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
                crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
                assert crop_obb is not None
                mask = crop_obb.within(torch.from_numpy(positions)).numpy()
                for k, t in map_to_tensors.items():
                    map_to_tensors[k] = map_to_tensors[k][mask]
                
                # Apply mask to part_weights if available
                if part_weights_np is not None:
                    part_weights_np = part_weights_np[mask]

                n = map_to_tensors["x"].shape[0]
                count = n

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=-1))
            n_after = np.sum(select)
            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")

        if np.sum(select) < n:
            CONSOLE.print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select]
            count = np.sum(select)

        ExportGaussianSplat.write_ply(str(filename), count, map_to_tensors)
        
        # Export part_weights for TwoPartBlendModel
        if part_weights_np is not None:
            try:
                weight_filename = self.output_dir / "3dgs_part_weight.txt"
                # Apply the same finite value filter
                part_weights_filtered = part_weights_np[select]
                
                np.savetxt(
                    str(weight_filename),
                    part_weights_filtered,
                    fmt='%.6f',
                    header=f'Part weights for {len(part_weights_filtered)} gaussians (sigmoid output, [0,1] range)\n'
                           f'Values closer to 0 belong more to part0, closer to 1 belong more to part1',
                    comments='# '
                )
                CONSOLE.print(f"[bold green]:white_check_mark: Saved {len(part_weights_filtered)} part weights to {weight_filename}")
            except Exception as e:
                CONSOLE.print(f"[bold yellow]Warning: Failed to export part weights: {e}")


@dataclass
class ExportArticulation(Exporter):
    """
    Export articulation parameters for SplatfactoArt model:
    - joint_type, joint_axis, joint_origin
    - per-frame rotation angle (rad/deg) and rotation matrix (3x3)
    If world_frame is True, values are mapped back to original world coordinates.
    """

    world_frame: bool = True

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        model = pipeline.model
        if not isinstance(model, SplatfactoArtModel):
            CONSOLE.print("[bold yellow]Model is not SplatfactoArtModel. Skipping articulation export.")
            return

        # Fetch parameters from model (normalized coordinates)
        with torch.no_grad():
            joint_type = model.joint_type
            # Get optimized axis (via property, applies rotation adjustment)
            axis = model.joint_axis.detach().cpu().numpy().astype(np.float32)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            
            # Get base axis and adjustment for debugging/analysis
            axis_base = model.joint_axis_base.detach().cpu().numpy().astype(np.float32)
            axis_adjustment = model.joint_axis_adjustment.detach().cpu().numpy().astype(np.float32)
            axis_adjustment_angle = float(np.linalg.norm(axis_adjustment))
            
            # Get origin (optimized and initial)
            origin = model.joint_origin.detach().cpu().numpy().astype(np.float32)
            origin_init = model.joint_origin_init.detach().cpu().numpy().astype(np.float32)
            origin_delta = origin - origin_init
            
            # Get joint parameters (optimized and initial)
            params = model.joint_params.detach().cpu().numpy().astype(np.float32).tolist()
            params_init = model.joint_params_init.detach().cpu().numpy().astype(np.float32).tolist()

        # Optionally transform axis/origin to world coordinates
        axis_out = axis.copy()
        axis_base_out = axis_base.copy()
        origin_out = origin.copy()
        origin_init_out = origin_init.copy()
        
        if self.world_frame:
            dp_outputs = pipeline.datamanager.train_dataparser_outputs
            dp_tf = dp_outputs.dataparser_transform.cpu().numpy()  # [3,4]
            R_dp = dp_tf[:, :3]
            t_dp = dp_tf[:, 3]
            s = float(dp_outputs.dataparser_scale)
            R_inv = np.linalg.inv(R_dp)
            if joint_type == "prismatic":
                params = [param / s for param in params if param is not None]
            # Transform axes (directions): inverse rotation only
            axis_out = (R_inv @ axis_out)
            axis_out = axis_out / (np.linalg.norm(axis_out) + 1e-8)
            axis_base_out = (R_inv @ axis_base_out)
            axis_base_out = axis_base_out / (np.linalg.norm(axis_base_out) + 1e-8)
            
            # Transform origins (points): inverse similarity
            origin_out = (R_inv @ (origin_out / s - t_dp))
            origin_init_out = (R_inv @ (origin_init_out / s - t_dp))

        def axis_angle_to_rotmat(a: np.ndarray, theta: float) -> np.ndarray:
            a = a / (np.linalg.norm(a) + 1e-8)
            x, y, z = a[0], a[1], a[2]
            c = np.cos(theta)
            s = np.sin(theta)
            C = 1.0 - c
            return np.array([
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ], dtype=np.float32)

        # Determine train/eval split and filtered frames
        train_dataset = pipeline.datamanager.train_dataset if hasattr(pipeline, 'datamanager') else None
        eval_dataset = pipeline.datamanager.eval_dataset if hasattr(pipeline, 'datamanager') else None
        n_train = len(train_dataset) if train_dataset is not None else 0
        n_eval = len(eval_dataset) if eval_dataset is not None else 0
        total = len(params)
        filtered = getattr(pipeline.datamanager, 'filtered_train_indices', None) if hasattr(pipeline, 'datamanager') else None
        filtered_set = set(filtered.tolist()) if (filtered is not None) else set()
        # Ensure frame 0 is never excluded during export
        filtered_set.discard(0)

        frames_all = []
        frames_train = []
        frames_eval = []
        if joint_type == "revolute":
            # Offset so that frame 0 is zero angle
            angle0 = float(params[0]) if len(params) > 0 else 0.0
            for i, angle in enumerate(params):
                angle_rel = float(angle) - angle0
                R = axis_angle_to_rotmat(axis_out, angle_rel)
                rec = {
                    "frame_index": int(i),
                    "angle_rad": float(angle_rel),
                    "angle_deg": float(np.degrees(angle_rel)),
                    "rotation_matrix": R.tolist(),
                }
                # Append to all
                frames_all.append(rec)
                # Append to split lists honoring filtered train frames
                if i < n_train:
                    if i not in filtered_set:
                        frames_train.append(rec)
                elif i < n_train + n_eval:
                    frames_eval.append(rec)
        else:
            # prismatic: export distance along axis; rotation is identity
            I = np.eye(3, dtype=np.float32)
            # Offset so that frame 0 is zero distance
            dist0 = float(params[0]) if len(params) > 0 else 0.0
            for i, dist in enumerate(params):
                dist_rel = float(dist) - dist0
                rec = {
                    "frame_index": int(i),
                    "distance": float(dist_rel),
                    "rotation_matrix": I.tolist(),
                }
                frames_all.append(rec)
                if i < n_train:
                    if i not in filtered_set:
                        frames_train.append(rec)
                elif i < n_train + n_eval:
                    frames_eval.append(rec)

        # Compute angular difference between optimized and initial axis
        cos_angle = float(np.dot(axis_out, axis_base_out))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        axis_change_deg = float(np.degrees(np.arccos(cos_angle)))
        
        out = {
            "joint_type": joint_type,
            "joint_axis_world": axis_out.tolist(),
            "joint_origin_world": origin_out.tolist(),
            # keep backward-compatible 'frames' containing all frames (train+eval, with train-filter applied)
            "frames": [rec for rec in frames_all if not (rec["frame_index"] < n_train and rec["frame_index"] in filtered_set)],
            # explicit splits
            "frames_train": frames_train,
            "frames_eval": frames_eval,
            # Add optimization details
            "optimization_details": {
                "joint_axis_base": axis_base_out.tolist(),
                "joint_axis_optimized": axis_out.tolist(),
                "axis_adjustment_angle_rad": float(axis_adjustment_angle),
                "axis_adjustment_angle_deg": float(np.degrees(axis_adjustment_angle)),
                "axis_change_deg": axis_change_deg,
                "joint_origin_init": origin_init_out.tolist(),
                "joint_origin_optimized": origin_out.tolist(),
                "origin_delta_norm": float(np.linalg.norm(origin_delta)),
                "params_init": params_init,
                "params_optimized": params,
            },
        }

        out_path = self.output_dir / "articulation.json"
        with open(out_path, "w", encoding="UTF-8") as f:
            json.dump(out, f, indent=2)
        CONSOLE.print(f"[bold green]:white_check_mark: Saved articulation to {out_path}")
        
        # Print optimization summary
        CONSOLE.print("\n[bold cyan]Optimization Summary:")
        CONSOLE.print(f"  Axis adjustment: {axis_change_deg:.4f}° (rotation angle: {float(np.degrees(axis_adjustment_angle)):.4f}°)")
        CONSOLE.print(f"  Origin shift: {float(np.linalg.norm(origin_delta)):.6f} units")
        
        if joint_type == "revolute":
            params_np = np.array(params)
            params_init_np = np.array(params_init)
            param_delta = params_np - params_init_np
            CONSOLE.print(f"  Joint angle changes: RMS={np.degrees(np.sqrt((param_delta**2).mean())):.4f}°, "
                         f"Max={np.degrees(np.abs(param_delta).max()):.4f}°")

Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
        Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
        Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
        Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="cameras")],
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="gaussian-splat")],
        Annotated[ExportArticulation, tyro.conf.subcommand(name="articulation")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
