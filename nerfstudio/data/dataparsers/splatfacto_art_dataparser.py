"""DataParser for Articulated Gaussian Splatting (SplatfactoArt)"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type
import json
import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
    CV_TO_GL,
    to4x4,
    read_camK,
    load_3dgs_ply,
    apply_left_se3_to_gaussians,
)
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class SplatfactoArtDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: SplatfactoArtDataParser)
    data_dir: Path = Path()
    """Directory containing the data"""
    twopart_blend_dir: Path = Path()
    """Directory containing twopart_blend outputs (3dgs.ply, part_weight.txt, etc.)"""
    motion_part: Literal[0, 1] = 1
    """Which part is considered the moving part (1 by default)."""
    output_dir: Path = Path()
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    train_split_fraction: float = 0.9
    eval_interval: int = 8
    
    scene_scale: float = 1.0
    """Scene scale for bounding box"""
    
    part_threshold: float = 0.5
    """Threshold for part assignment (weights > threshold belong to part1)"""
    
    depth_unit_scale_factor: float = 1.0
    """Scaling factor to apply to depth values"""
    
    orientation_method: Literal["pca", "up", "none"] = "up"
    """The method to use for orientation."""
    
    center_poses: bool = True
    """Whether to center the poses."""
    
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""


@dataclass
class SplatfactoArtDataParser(DataParser):
    config: SplatfactoArtDataParserConfig
    
    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        data_dir = self.config.data_dir
        assert data_dir.exists(), f"Data directory {data_dir} does not exist"
        
        blend_dir = self.config.twopart_blend_dir
        assert blend_dir.exists(), f"TwoPartBlend directory {blend_dir} does not exist"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load 3D Gaussians
        ply_path = blend_dir / "object_3dgs.ply"
        assert ply_path.exists(), f"3DGS file not found: {ply_path}"
        gaussians = load_3dgs_ply(ply_path, device)
        
        # Load part weights
        weight_path = blend_dir / "3dgs_part_weight.txt"
        assert weight_path.exists(), f"Part weight file not found: {weight_path}"
        part_weights = torch.tensor(np.loadtxt(str(weight_path)), dtype=torch.float32, device=device)
        
        # Load camera poses (only train)
        cam_train_path = blend_dir / "transforms_train.json"
        with open(cam_train_path) as f:
            cam_train_data = json.load(f)
        cam_all = list(cam_train_data)
        
        # Load pre-estimated articulation from preprocessing/estimate_joint.py
        est_path = blend_dir / "estimated_joint.json"
        assert est_path.exists(), f"Estimated joint file not found: {est_path}"
        with est_path.open("r", encoding="utf-8") as f:
            est = json.load(f)
        joint_type = str(est["joint_type"])  # "revolute" or "prismatic"
        joint_axis = np.asarray(est["axis"], dtype=np.float32)
        joint_origin = np.asarray(est["origin"], dtype=np.float32)
        rotation_offset = np.asarray(est["global_transform"], dtype=np.float32)
        joint_params = [float(x) for x in est.get("joint_params", [])]
        # Verify camera and joint params counts match (if provided)
        if len(joint_params) > 0:
            assert len(cam_all) == len(joint_params), \
                f"Camera frames ({len(cam_all)}) != joint_params ({len(joint_params)})"
        CONSOLE.log(f"[green]Loaded joint: type={joint_type}, axis={joint_axis}, origin={joint_origin}")
        if len(joint_params) > 0:
            if joint_type == "revolute":
                CONSOLE.log(f"[green]Angle range: [{min(joint_params):.3f}, {max(joint_params):.3f}] rad "
                           f"({np.degrees(min(joint_params)):.1f}° to {np.degrees(max(joint_params)):.1f}°)")
            else:
                CONSOLE.log(f"[green]Distance range: [{min(joint_params):.3f}, {max(joint_params):.3f}]")
        
        # Prepare camera data
        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []
        
        for frame in cam_all:
            # Image path
            img_path = Path(frame["file_path"])
            image_filenames.append(img_path)
            
            # Mask path (replace images with masks)
            mask_filepath = Path(str(frame["file_path"]).replace("images", "masks"))
            if mask_filepath.exists():
                mask_filenames.append(mask_filepath)
            
            # Depth path (replace images with depth and .png with .npy or .npz)
            depth_filepath = Path(str(frame["file_path"]).replace("images", "depth").replace(".png", ".npz"))
            if depth_filepath.exists():
                depth_filenames.append(depth_filepath)
            
            # Camera pose (OpenCV -> OpenGL conversion)
            T = to4x4(frame["transform"])
            T = T @ CV_TO_GL
            poses.append(T)
        
        poses = torch.tensor(np.array(poses), dtype=torch.float32)
        
        # Normalize camera poses (auto orient and center)
        poses_normalized, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_poses=self.config.center_poses,
        )
        
        # Scale poses
        dataparser_scale = 1.0
        if self.config.auto_scale_poses:
            dataparser_scale /= float(torch.max(torch.abs(poses_normalized[:, :3, 3])))
        dataparser_scale *= self.config.scale_factor
        
        poses_normalized[:, :3, 3] *= dataparser_scale
        
        # Apply the same transform and scale to gaussians
        # transform_matrix is [3,4], convert to [4,4]
        T_norm = torch.eye(4, dtype=torch.float32)
        T_norm[:3, :] = transform_matrix
        
        # Apply normalization transform to gaussians (rigid + orientation) using shared helper
        gaussians = apply_left_se3_to_gaussians(gaussians, T_norm.to(device), device)
        gaussians["means"] = gaussians["means"] * dataparser_scale
        scale_log = torch.log(torch.tensor(float(dataparser_scale), dtype=gaussians["scales"].dtype, device=device))
        gaussians["scales"] = gaussians["scales"] + scale_log

        
        # Map joint axis/origin/params to the same normalized space as gaussians & cameras
        Rn_np = transform_matrix[:3, :3].detach().cpu().numpy()
        tn_np = transform_matrix[:3, 3].detach().cpu().numpy()
        axis_norm = (Rn_np @ np.asarray(joint_axis, dtype=np.float32))
        axis_norm = axis_norm / (np.linalg.norm(axis_norm) + 1e-8)
        origin_norm = (Rn_np @ np.asarray(joint_origin, dtype=np.float32) + tn_np)
        origin_norm = origin_norm * float(dataparser_scale)
        if joint_type == "prismatic":
            joint_params_mapped = [float(p) * float(dataparser_scale) for p in joint_params]
        else:
            joint_params_mapped = [float(p) for p in joint_params]
        # rotation_offset is 4x4 (T_off in world). Map to normalized space under similarity:
        # x_ns = s * (R_n x_w + t_n), so
        # R_off_ns = R_n R_off R_n^T
        # t_off_ns = - s R_n R_off R_n^T t_n + s R_n t_off + s t_n
        Toff = np.asarray(rotation_offset, dtype=np.float32)
        R_off = Toff[:3, :3].astype(np.float32)
        t_off = Toff[:3, 3].astype(np.float32)
        Rn_T = Rn_np.T
        R_off_ns = (Rn_np @ R_off) @ Rn_T
        t_off_ns = (-float(dataparser_scale)) * (R_off_ns @ tn_np) + (float(dataparser_scale)) * (Rn_np @ t_off) + (float(dataparser_scale)) * tn_np
        Toff_norm = np.eye(4, dtype=np.float32)
        Toff_norm[:3, :3] = R_off_ns.astype(np.float32)
        Toff_norm[:3, 3] = t_off_ns.astype(np.float32)

        # Convert [N, 3, 4] poses back to [N, 4, 4] for model compatibility
        N_poses = poses_normalized.shape[0]
        poses_full = torch.zeros((N_poses, 4, 4), dtype=torch.float32)
        poses_full[:, :3, :] = poses_normalized
        poses_full[:, 3, 3] = 1.0
        poses = poses_full
        
        # Get image dimensions from the first image
        with Image.open(image_filenames[0]) as im:
            width, height = im.size

        
        # Create cameras (assumes pinhole intrinsics from cam_K.txt)
        # Try to find cam_K.txt in dataset directory
        camK_path = data_dir / "cam_K.txt"
        fx, fy, cx, cy = read_camK(camK_path)
        n_frames = len(image_filenames)
        
        # Split train/eval
        if self.config.eval_mode == "fraction":
            i_train, i_eval = get_train_eval_split_fraction(
                image_filenames, self.config.train_split_fraction, True
            )
        elif self.config.eval_mode == "interval":
            i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
        else:
            i_train, i_eval = get_train_eval_split_all(image_filenames)
        
        indices = i_train if split == "train" else i_eval
        
        # Slice data
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) == n_frames else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) == n_frames else depth_filenames
        poses = poses[indices]
        # Pass ALL joint params (not just this split) so model can use global frame indices
        joint_params_all = joint_params_mapped
        
        # Create cameras
        # Use GLOBAL indices (not local split indices) for frame_index
        # This ensures eval frames can correctly index into joint_params which is sized based on train split
        cameras = Cameras(
            fx=torch.full((len(indices),), fx, dtype=torch.float32),
            fy=torch.full((len(indices),), fy, dtype=torch.float32),
            cx=torch.full((len(indices),), cx, dtype=torch.float32),
            cy=torch.full((len(indices),), cy, dtype=torch.float32),
            height=torch.full((len(indices),), height, dtype=torch.int32),
            width=torch.full((len(indices),), width, dtype=torch.int32),
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
            metadata={"frame_index": torch.tensor(indices, dtype=torch.long).unsqueeze(-1)},
        )
        
        # Scene box
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], 
                 [aabb_scale, aabb_scale, aabb_scale]], 
                dtype=torch.float32
            )
        )
        
        # Metadata
        # Pass ALL joint params so model can use global frame indices during eval
        metadata = {
            "gaussians": gaussians,
            "part_weights": part_weights,
            "part_threshold": self.config.part_threshold,
            "joint_type": joint_type,
            "joint_axis_init": torch.tensor(axis_norm, dtype=torch.float32),
            "joint_origin_init": torch.tensor(origin_norm, dtype=torch.float32),
            "joint_params_init": torch.tensor(joint_params_all, dtype=torch.float32),
            "rotation_offset_init": torch.tensor(Toff_norm, dtype=torch.float32),  # [4, 4] SE3 matrix
            "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
            "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            "motion_part": int(self.config.motion_part),
            "output_dir": self.config.output_dir,
        }
        
        return DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=dataparser_scale,
            dataparser_transform=transform_matrix,
            metadata=metadata,
        )
