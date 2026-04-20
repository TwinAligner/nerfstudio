"""Articulated Gaussian Splatting Model (SplatfactoArt)"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch.nn import Parameter
import json

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.cameras.lie_groups import exp_map_SO3xR3

from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
    quaternion_raw_multiply,
)
from pytorch3d.loss import chamfer_distance

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics
from PIL import Image


@dataclass
class SplatfactoArtModelConfig(SplatfactoModelConfig):
    """Configuration for articulated Gaussian splatting model"""
    
    _target: Type = field(default_factory=lambda: SplatfactoArtModel)
    
    # Gaussian refinement
    enable_gs_refinement: bool = True
    """Enable Gaussian splitting and culling during training"""
    
    # Part weight optimization
    trainable_weights: bool = True
    """Optimize per-point part weights (sigmoid in [0,1])"""
    
    # Joint parameter optimization
    optimize_joint_axis: bool = True
    """Optimize joint axis direction"""
    
    optimize_joint_origin: bool = True
    """Optimize joint origin point (pivot/reference)"""
    
    optimize_joint_params: bool = True
    """Optimize per-frame joint parameters (angle/distance)"""
    
    optimize_rotation_offset: bool = True
    """Optimize common rotation offset (3-param axis-angle)"""
    
    # Regularization
    axis_adjustment_weight: float = 1e-3
    """L2 penalty on axis rotation adjustment to prefer small changes"""
    
    origin_adjustment_weight: float = 1e-4
    """L2 penalty on origin adjustment to prefer small changes"""
    
    rotation_offset_weight: float = 1e-3
    """L2 penalty on rotation offset to prefer small changes from initial estimate"""
    
    param_smoothness_weight: float = 0.0
    """Temporal smoothness penalty for joint parameters (reduced from 1e-2)"""
    
    use_second_order_smoothness: bool = True
    """Use second-order (acceleration) instead of first-order (velocity) smoothness"""

    # RGB visualization configuration (save each frame once at training start)
    visualize_rgb: bool = True
    """Save rendered RGBs during early training (first time each frame is seen)."""

    visualize_rgb_post_step: Optional[int] = 30000
    """If set, save each frame once more after training reaches this global step."""


class SplatfactoArtModel(SplatfactoModel):
    """Articulated Gaussian splatting model with joint transformations"""
    
    config: SplatfactoArtModelConfig
    
    def __init__(self, *args, metadata: Optional[Dict] = None, seed_points=None, **kwargs):
        self.metadata = metadata
        super().__init__(*args, seed_points=seed_points, **kwargs)
        # Track which frames have been visualized at least once
        self._visualized_frames: set[int] = set()
        self._post_visualized_frames: set[int] = set()
        self._evaluated_frames: set[int] = set()
    
    def populate_modules(self):
        """Initialize model parameters from metadata"""
        if self.metadata is None:
            raise ValueError("Metadata must be provided for SplatfactoArtModel")
        
        super().populate_modules()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Gaussians from metadata
        self._init_gaussians(device)
        
        # Load part weights
        self._init_part_weights(device)
        
        # Initialize joint parameters
        self._init_joint_parameters(device)
    
    # def _init_gaussians(self, device: str):
    #     """Initialize Gaussian parameters from metadata"""
    #     gaussians = self.metadata["gaussians"]
    #     self.gauss_params = torch.nn.ParameterDict({
    #         "means": Parameter(gaussians["means"].to(device), requires_grad=True),
    #         "scales": Parameter(gaussians["scales"].to(device), requires_grad=True),
    #         "quats": Parameter(gaussians["quats"].to(device), requires_grad=True),
    #         "opacities": Parameter(gaussians["opacities"].to(device), requires_grad=True),
    #         "features_dc": Parameter(gaussians["features_dc"].to(device), requires_grad=True),
    #         "features_rest": Parameter(gaussians["features_rest"].to(device), requires_grad=True),
    #     })
    import torch
    from torch.nn import Parameter
    
    

    def random_quat_tensor(N: int, device="cpu"):
        import math
        """生成随机单位四元数"""
        u = torch.rand(N, device=device)
        v = torch.rand(N, device=device)
        w = torch.rand(N, device=device)
        return torch.stack([
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ], dim=-1)


    def _init_gaussians(self, device: str):
        from sklearn.neighbors import NearestNeighbors
        """
        初始化 Gaussian 参数：
        - 从 seed_points（点云） 或 随机初始化
        - 自动计算 scale (基于最近邻距离)
        - 随机初始化四元数 (quaternion)
        - 初始化颜色 (features_dc/rest)
        - 初始化透明度 (opacities)
        """
        gaussians = self.metadata["gaussians"]
        cfg = self.config  # 假设你有 config 或对应参数
        seed_points = getattr(self, "seed_points", None)
        def random_quat_tensor(N: int, device="cpu"):
            import math
            """生成随机单位四元数"""
            u = torch.rand(N, device=device)
            v = torch.rand(N, device=device)
            w = torch.rand(N, device=device)
            return torch.stack([
                torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
                torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
                torch.sqrt(u) * torch.sin(2 * math.pi * w),
                torch.sqrt(u) * torch.cos(2 * math.pi * w),
            ], dim=-1)

        num_points = gaussians["means"].shape[0]

        # === 2️⃣ 计算 scale (基于邻域距离) ===
        x_np = gaussians["means"].detach().cpu().numpy()
        nn_model = NearestNeighbors(n_neighbors=4, algorithm="auto").fit(x_np)
        distances, _ = nn_model.kneighbors(x_np)
        # 去掉自身距离 (第一个是0)
        distances = torch.from_numpy(distances[:, 1:4]).to(device)
        # sklearn returns float64; clamp -> float32 keeps numerics stable for log
        avg_dist = distances.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        avg_dist = avg_dist.to(dtype=gaussians["means"].dtype)
        scales = torch.log(avg_dist.repeat(1, 3)).to(dtype=torch.float32)
        # === 3️⃣ 随机旋转 quaternion ===
        quats = random_quat_tensor(num_points, device=device).to(dtype=torch.float32)

        # === 5️⃣ 初始化透明度 ===
        # opacities = torch.logit(0.1 * torch.ones(num_points, 1, device=device)).to(dtype=torch.float32)

        # === 6️⃣ 打包为 ParameterDict ===
        self.gauss_params = torch.nn.ParameterDict({
            "means": Parameter(gaussians["means"].to(device), requires_grad=True),
            "scales": Parameter(scales, requires_grad=True),
            "quats": Parameter(quats, requires_grad=True),
            "opacities": Parameter(gaussians["opacities"].to(device), requires_grad=True),
            "features_dc": Parameter(gaussians["features_dc"].to(device), requires_grad=True),
            "features_rest": Parameter(gaussians["features_rest"].to(device), requires_grad=True),
        })


    def _init_part_weights(self, device: str):
        """Initialize per-point part weights from metadata"""
        pw_init = self.metadata["part_weights"].to(device)
        # Ensure shape [N]
        pw_init = pw_init.view(-1).clamp(1e-4, 1 - 1e-4)
        logits_init = torch.log(pw_init / (1.0 - pw_init)).view(-1, 1)
        self.gauss_params["part_weights"] = Parameter(
            logits_init, 
            requires_grad=self.config.trainable_weights
        )
        
        # self.part_threshold = float(self.metadata.get("part_threshold", 0.5))
        # n_part0 = (pw_init < self.part_threshold).sum().item()
        # n_part1 = (pw_init >= self.part_threshold).sum().item()
        # print(f"[SplatfactoArt] Part0: {n_part0}, Part1: {n_part1} gaussians")
    
    def _init_joint_parameters(self, device: str):
        """Initialize joint parameters from metadata"""
        self.joint_type = self.metadata["joint_type"]  # "revolute" or "prismatic"
        joint_axis_init = self.metadata["joint_axis_init"].to(device)
        joint_origin_init = self.metadata["joint_origin_init"].to(device)
        joint_params_init = self.metadata["joint_params_init"].to(device)
        rotation_offset_init = self.metadata["rotation_offset_init"].to(device)  # [4, 4] SE3 matrix
        
        # Store initial axis as buffer (for rotation adjustment approach)
        self.register_buffer("joint_axis_base", joint_axis_init.clone())
        
        # Optimize rotation adjustment (axis-angle in SO(3) tangent space)
        # This is more principled than directly optimizing the axis vector
        self.joint_axis_adjustment = Parameter(
            torch.zeros(3, device=device),
            requires_grad=self.config.optimize_joint_axis
        )
        
        self.joint_origin = Parameter(
            joint_origin_init.clone(),
            requires_grad=self.config.optimize_joint_origin
        )
        self.joint_params = Parameter(
            joint_params_init.clone(),
            requires_grad=self.config.optimize_joint_params
        )
        
        # Common rotation offset: base matrix + delta (axis-angle in SO(3) tangent space)
        # This corrects systematic rotation bias in initial point cloud alignment
        # Similar to camera pose optimization: optimize small perturbation in Lie algebra
        # Store full 4x4 base; optimize 6-DoF SE(3) delta (tx, ty, tz, rx, ry, rz)
        self.register_buffer("rotation_offset_base", rotation_offset_init.clone())
        self.rotation_offset_delta = Parameter(
            torch.zeros(6, device=device),
            requires_grad=self.config.optimize_rotation_offset
        )
        
        # Store initial values as buffers (no gradient) for computing deltas
        self.register_buffer("joint_origin_init", joint_origin_init.clone())
        self.register_buffer("joint_params_init", joint_params_init.clone())
        
        self.motion_part = int(self.metadata["motion_part"])
        
        # print(f"[SplatfactoArt] Joint type: {self.joint_type}")
        # print(f"[SplatfactoArt] Frames: {len(joint_params_init)}")
        # print(f"[SplatfactoArt] Axis: {joint_axis_init.cpu().numpy()}")
        # print(f"[SplatfactoArt] Origin: {joint_origin_init.cpu().numpy()}")
    
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get Gaussian parameter groups including part weights"""
        gps = super().get_gaussian_param_groups()
        if "part_weights" in self.gauss_params:
            gps["part_weights"] = [self.gauss_params["part_weights"]]
        return gps
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get parameter groups for optimization"""
        param_groups = super().get_param_groups()
        
        if self.config.optimize_joint_axis:
            param_groups["joint_axis_adjustment"] = [self.joint_axis_adjustment]
        if self.config.optimize_joint_origin:
            param_groups["joint_origin"] = [self.joint_origin]
        if self.config.optimize_joint_params:
            param_groups["joint_params"] = [self.joint_params]
        if self.config.optimize_rotation_offset:
            param_groups["rotation_offset"] = [self.rotation_offset_delta]
        
        return param_groups
    
    @property
    def joint_axis(self) -> torch.Tensor:
        """Compute current joint axis by applying rotation adjustment to base axis.
        
        Uses exponential map to convert axis-angle adjustment to rotation matrix,
        then applies to base axis. This ensures the axis remains unit-length without
        explicit normalization or regularization.
        
        Returns:
            Normalized joint axis [3]
        """
        if not self.config.optimize_joint_axis:
            # Optimization disabled, return base axis
            return self.joint_axis_base
        
        # Always compute rotation to maintain gradient flow
        # Convert axis-angle to rotation matrix using pytorch3d
        # joint_axis_adjustment is axis-angle representation [3]
        R_adjust = axis_angle_to_matrix(self.joint_axis_adjustment)  # [3, 3]
        
        # Apply rotation to base axis
        axis_rotated = R_adjust @ self.joint_axis_base  # [3]
        
        # Normalize to ensure unit length (should already be close due to rotation)
        return F.normalize(axis_rotated, dim=0)
    
    @property
    def rotation_offset_se3(self) -> torch.Tensor:
        """Current 4x4 SE3 rotation offset with optimized 6-DoF delta (like cam pose)."""
        if not self.config.optimize_rotation_offset:
            return self.rotation_offset_base
        # Build Adj from 6-DoF delta (match camera optimizer convention)
        adj_3x4 = exp_map_SO3xR3(self.rotation_offset_delta[None, :]).squeeze(0)  # [3,4]
        bottom_row = torch.tensor([[0., 0., 0., 1.]], device=adj_3x4.device, dtype=adj_3x4.dtype)
        Adj = torch.cat([adj_3x4, bottom_row], dim=0)  # [4,4]
        # Compose in the same order as cam pose (post-multiply base)
        T = self.rotation_offset_base @ Adj
        return T

    @property
    def rotation_offset(self) -> torch.Tensor:
        """Return only rotation part of the current 4x4 offset for convenience."""
        T = self.rotation_offset_se3
        return T[:3, :3]
    
    @property
    def part_weights(self) -> torch.Tensor:
        """Get part weights in [0,1] range, shape [N, 1]"""
        weights = torch.sigmoid(self.gauss_params["part_weights"])
        # Ensure shape is [N, 1] for proper broadcasting
        if weights.ndim == 1:
            weights = weights.unsqueeze(-1)
        return weights
    
    def apply_joint_transform(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        param: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply full rigid joint transformation to all points.
        
        This applies the complete transformation to all input gaussians.
        The transformation sequence is:
        1. Apply rotation_offset (common alignment correction)
        2. Apply joint transformation (revolute or prismatic)
        
        The caller is responsible for selecting which points to transform.
        
        Args:
            means: Gaussian positions [N,3]
            quats: Gaussian quaternions [N,4]
            param: Joint parameter (angle for revolute, distance for prismatic)
            
        Returns:
            Transformed means and quaternions
        """
        # Get current joint axis (already normalized via property)
        axis = self.joint_axis
        origin = self.joint_origin
        
        means_transformed = means
        quats_transformed = quats
        
        # Step 1: Apply rotation offset (common alignment correction)
        T_offset = self.rotation_offset_se3  # [4, 4]
        R_offset = T_offset[:3, :3]
        t_offset = T_offset[:3, 3]
        
        # Transform positions with full SE3: p' = R p + t
        means_transformed = (R_offset @ means.T).T + t_offset[None, :]
        
        # Transform orientations: q' = q_R * q
        q_offset_wxyz = matrix_to_quaternion(R_offset[None])[0]  # [4] in wxyz
        quats_wxyz = torch.stack(
            [self.quats[..., 3], self.quats[..., 0], self.quats[..., 1], self.quats[..., 2]],
            dim=-1,
        )
        quats_transformed_wxyz = quaternion_raw_multiply(q_offset_wxyz[None, :], quats_wxyz)
        quats_transformed = torch.stack(
            [
                quats_transformed_wxyz[..., 1],
                quats_transformed_wxyz[..., 2],
                quats_transformed_wxyz[..., 3],
                quats_transformed_wxyz[..., 0],
            ],
            dim=-1,
        )
        
        # Step 2: Apply joint transformation
        if self.joint_type == "revolute":
            angle = param
            # Full rigid rotation: R = exp(axis * angle)
            axis_angle = axis * angle  # [3]
            R_joint = axis_angle_to_matrix(axis_angle)  # [3, 3]
            
            # Transform positions: p' = R(p - o) + o
            means_centered = means_transformed - origin[None, :]
            means_transformed = (R_joint @ means_centered.T).T + origin[None, :]
            
            # Transform orientations: q' = q_R * q
            q_R_wxyz = matrix_to_quaternion(R_joint[None])[0]  # [4] in wxyz
            quats_wxyz = torch.stack(
                [self.quats[..., 3], self.quats[..., 0], self.quats[..., 1], self.quats[..., 2]],
                dim=-1,
            )
            quats_transformed_wxyz = quaternion_raw_multiply(q_R_wxyz[None, :], quats_wxyz)
            quats_transformed = torch.stack(
                [
                    quats_transformed_wxyz[..., 1],
                    quats_transformed_wxyz[..., 2],
                    quats_transformed_wxyz[..., 3],
                    quats_transformed_wxyz[..., 0],
                ],
                dim=-1,
            )
        
        elif self.joint_type == "prismatic":
            # Full rigid translation: p' = p + axis * distance
            distance = param
            translation = axis * distance
            means_transformed = means_transformed + translation[None, :]
        
        return means_transformed, quats_transformed
    
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Forward pass with concatenated rendering (SplArt-style)
        
        Concatenates static and mobile gaussians and renders once:
        - Static gaussians: original positions with opacity * (1-w)
        - Mobile gaussians: transformed positions with opacity * w
        - Render all together for proper depth sorting and alpha blending
        """
        assert isinstance(camera, Cameras)
        assert camera.shape[0] == 1, "Only one camera at a time"
        
        camera = camera.to(self.device)
        
        # Get background color
        if self.config.background_color == "random":
            background = torch.rand(3, device=self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            background = self.background_color.to(self.device)
        
        # Get frame index from camera metadata
        frame_idx = int(camera.metadata["frame_index"][0].item())
 
        # Get part weights [N, 1]
        w = self.part_weights  # [N, 1], values in [0, 1]
        if self.motion_part == 0:
            w = 1 - w
        
        # Handle crop box
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                H, W = int(camera.height.item()), int(camera.width.item())
                rgb = background[None, None, :].expand(H, W, 3)
                depth = torch.zeros(H, W, 1, device=self.device)
                alpha_out = torch.zeros(H, W, 1, device=self.device)
                return {"rgb": rgb, "depth": depth, "accumulation": alpha_out, "background": background}
        else:
            crop_ids = None
        
        # Transform mobile gaussians
        param = self.joint_params[frame_idx]
        means_transformed, quats_transformed = self.apply_joint_transform(
            self.means, self.quats, param
        )
        
        # Concatenate static and mobile gaussians
        # Keep gradients for transformed positions/orientations to optimize joint parameters
        means_concat = torch.cat([self.means, means_transformed], dim=0)
        quats_concat = torch.cat([self.quats, quats_transformed], dim=0)
        scales_concat = torch.cat([self.scales, self.scales], dim=0)
        features_dc_concat = torch.cat([self.features_dc, self.features_dc], dim=0)
        features_rest_concat = torch.cat([self.features_rest, self.features_rest], dim=0)
        
        # For opacities: mix in alpha space
        w_static = 1 - w
        w_mobile = w
        alpha = torch.sigmoid(self.opacities)  # [N,1]
        alpha_static = alpha * w_static
        alpha_mobile = alpha * w_mobile
        alpha_concat = torch.cat([alpha_static, alpha_mobile], dim=0)
        
        # === DIRECT RENDERING - NO PARAMETER REPLACEMENT ===
        # This ensures complete gradient flow from output to transformation parameters
        
        # Get optimized camera pose
        if self.training:
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)[0, ...]
        else:
            optimized_camera_to_world = camera.camera_to_worlds[0, ...]
        
        # Prepare camera matrices for gsplat
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        
        R = optimized_camera_to_world[:3, :3]  # 3 x 3
        T = optimized_camera_to_world[:3, 3:4]  # 3 x 1
        
        # Flip z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1., -1., -1.], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        
        # World2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.zeros(4, 4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        viewmat[3, 3] = 1.0
        
        # Camera parameters
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        
        # Concatenate colors
        colors_concat = torch.cat((features_dc_concat[:, None, :], features_rest_concat), dim=1)
        
        # Project gaussians - using concat tensors directly (preserves gradients!)
        BLOCK_WIDTH = 16
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(
            means_concat,  # Gradients flow through here!
            torch.exp(scales_concat),
            1,
            quats_concat / quats_concat.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            camera.fx.item(),
            camera.fy.item(),
            cx, cy, H, W,
            BLOCK_WIDTH,
        )
        
        # Store for Gaussian refinement (needed by parent class after_train)
        self.xys = xys
        self.radii = radii
        
        # Rescale camera back
        camera.rescale_output_resolution(camera_downscale)
        
        if (radii).sum() == 0:
            rgb = background[None, None, :].expand(H, W, 3)
            depth = torch.zeros(H, W, 1, device=self.device)
            alpha_out = torch.zeros(H, W, 1, device=self.device)
            return {"rgb": rgb, "depth": depth, "accumulation": alpha_out, "background": background}
        
        # Compute colors with spherical harmonics
        if self.config.sh_degree > 0:
            viewdirs = means_concat.detach() - optimized_camera_to_world.detach()[:3, 3]
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_concat)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)
        else:
            rgbs = torch.sigmoid(colors_concat[:, 0, :])
        
        # Apply opacity (with screen space antialiasing if enabled)
        if self.config.rasterize_mode == "antialiased":
            opacities_render = alpha_concat * comp[:, None]
        else:
            opacities_render = alpha_concat
        
        # Rasterize - gradients flow through rgbs and opacities_render!
        rgb, alpha_out = rasterize_gaussians(
            xys, depths, radii, conics, num_tiles_hit,
            rgbs,  # Gradients flow here
            opacities_render,  # And here (from alpha_concat -> w -> transformation)
            H, W, BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )
        alpha_out = alpha_out[..., None]
        rgb = torch.clamp(rgb, max=1.0)
        
        # Optional depth rendering
        depth_im = None
        current_step = getattr(self, "step", 0)
        is_eval_step = (self.config.visualize_rgb_post_step is not None and 
                        current_step >= self.config.visualize_rgb_post_step)
        
        if self.config.output_depth_during_training or not self.training or is_eval_step:
            depth_im = rasterize_gaussians(
                xys, depths, radii, conics, num_tiles_hit,
                depths[:, None].repeat(1, 3),
                opacities_render,
                H, W, BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]
            depth_im = torch.where(alpha_out > 0, depth_im / alpha_out, depth_im.detach().max())
        
        outputs = {
            "rgb": rgb, 
            "depth": depth_im, 
            "accumulation": alpha_out, 
            "background": background,
            "frame_idx": frame_idx,
            "camera": camera,
            "c2w": optimized_camera_to_world
        }

        # Save training RGB the first time we see each frame
        if self.training and self.config.visualize_rgb:
            try:
                self.visualize_training_rgb(outputs, camera)
            except Exception:
                pass

        return outputs

    @torch.no_grad()
    def visualize_training_rgb(self, outputs: Dict, camera: Cameras) -> None:
        """Save rendered RGB once per frame at training start.

        Args:
            outputs: dict containing 'rgb' tensor [H, W, 3] in [0,1]
            camera: Cameras used for rendering (expects metadata['frame_index'])
        """
        frame_idx = int(camera.metadata["frame_index"][0].item())

        targets: List[Tuple[Path, str]] = []

        if frame_idx not in self._visualized_frames:
            self._visualized_frames.add(frame_idx)
            vis_dir = Path(self.metadata["output_dir"]) / "rgb_vis_art"
            targets.append((vis_dir, f"train_init_frame{frame_idx:05d}.png"))

        if not targets:
            return

        rgb = outputs["rgb"].detach().cpu().numpy().clip(0, 1)
        rgb_u8 = (rgb * 255).astype("uint8")

        for directory, filename in targets:
            directory.mkdir(parents=True, exist_ok=True)
            img_path = directory / filename
            Image.fromarray(rgb_u8).save(img_path)
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Compute losses with regularization"""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        
        # Axis adjustment regularization: prefer small rotations from initial axis
        if self.config.optimize_joint_axis and self.config.axis_adjustment_weight > 0:
            # L2 on axis-angle representation (axis_adjustment is in tangent space)
            loss_dict["axis_adjustment_reg"] = \
                self.config.axis_adjustment_weight * (self.joint_axis_adjustment ** 2).sum()
        
        # Origin adjustment regularization: prefer small translations from initial origin
        if self.config.optimize_joint_origin and self.config.origin_adjustment_weight > 0:
            origin_delta = self.joint_origin - self.joint_origin_init
            loss_dict["origin_adjustment_reg"] = \
                self.config.origin_adjustment_weight * (origin_delta ** 2).sum()
        
        # Rotation offset regularization: prefer small delta (stay close to base rotation)
        if self.config.optimize_rotation_offset and self.config.rotation_offset_weight > 0:
            # Regularize delta to be small (L2 penalty on axis-angle magnitude)
            # This encourages staying close to the initial rotation estimate
            loss_dict["rotation_offset_delta_reg"] = \
                self.config.rotation_offset_weight * (self.rotation_offset_delta ** 2).sum()
        
        # Temporal smoothness for joint parameters
        if self.config.optimize_joint_params and self.config.param_smoothness_weight > 0:
            if self.config.use_second_order_smoothness and len(self.joint_params) > 2:
                # Second-order smoothness (acceleration): penalize changes in velocity
                # This allows constant-velocity motion (uniform rotation/translation)
                # params[2:] - 2*params[1:-1] + params[:-2] = acceleration
                param_accel = self.joint_params[2:] - 2 * self.joint_params[1:-1] + self.joint_params[:-2]
                loss_dict["param_smoothness"] = \
                    self.config.param_smoothness_weight * (param_accel ** 2).mean()
            elif len(self.joint_params) > 1:
                # First-order smoothness (velocity): penalize large frame-to-frame changes
                param_diff = self.joint_params[1:] - self.joint_params[:-1]
                loss_dict["param_smoothness"] = \
                    self.config.param_smoothness_weight * (param_diff ** 2).mean()
        
        return loss_dict
    
    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute metrics with joint statistics"""
        metrics_dict = super().get_metrics_dict(outputs, batch)
        
        with torch.no_grad():
            # Joint parameter statistics for monitoring optimization progress
            # metrics_dict["joint_param_min"] = self.joint_params.min()
            # metrics_dict["joint_param_max"] = self.joint_params.max()
            # metrics_dict["joint_param_range"] = self.joint_params.max() - self.joint_params.min()
            # metrics_dict["joint_param_mean"] = self.joint_params.abs().mean()
            
            # if self.joint_type == "revolute":
            #     metrics_dict["joint_angle_deg_min"] = torch.rad2deg(self.joint_params.min())
            #     metrics_dict["joint_angle_deg_max"] = torch.rad2deg(self.joint_params.max())
            
            # Compute changes relative to initialization
            # Joint parameters change (per-frame)
            param_delta = self.joint_params - self.joint_params_init
            # metrics_dict["joint_param_delta_rms"] = torch.sqrt((param_delta ** 2).mean())
            # metrics_dict["joint_param_delta_max"] = param_delta.abs().max()
            metrics_dict["joint_param_delta_mean"] = param_delta.abs().mean()
            
            # Joint axis change (if optimized)
            if self.config.optimize_joint_axis:
                # Magnitude of axis-angle adjustment
                adjustment_angle = self.joint_axis_adjustment.norm()
                # metrics_dict["joint_axis_adjustment_norm"] = adjustment_angle
                # metrics_dict["joint_axis_adjustment_deg"] = torch.rad2deg(adjustment_angle)
                
                # Angular difference between current and initial axis (computed via dot product)
                cos_angle = (self.joint_axis * self.joint_axis_base).sum()
                # Clamp to avoid numerical issues with acos
                cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                metrics_dict["joint_axis_angle_deg"] = torch.rad2deg(torch.acos(cos_angle))
            
            # Joint origin change (if optimized)
            if self.config.optimize_joint_origin:
                origin_delta = self.joint_origin - self.joint_origin_init
                metrics_dict["joint_origin_delta_norm"] = origin_delta.norm()
                # metrics_dict["joint_origin_delta_max"] = origin_delta.abs().max()
            
            # Rotation offset change (if optimized)
            if self.config.optimize_rotation_offset:
                # Delta magnitude (axis-angle norm)
                # Last 3 are rotation axis-angle
                delta_angle = self.rotation_offset_delta[3:].norm()
                metrics_dict["rotation_offset_delta_deg"] = torch.rad2deg(delta_angle)
                
                # Total rotation angle from base
                # T = self.rotation_offset_se3
                # R = T[:3, :3]
                # R_base = self.rotation_offset_base[:3, :3]
                # trace_val = torch.trace(R.T @ R_base)
                # cos_theta = (trace_val - 1.0) / 2.0
                # cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Numerical stability
                # total_angle = torch.acos(cos_theta)
                # metrics_dict["rotation_offset_total_deg"] = torch.rad2deg(total_angle)

            # End-of-training evaluation and saving
            current_step = getattr(self, "step", 0)
            post_step = self.config.visualize_rgb_post_step
            frame_idx = outputs.get("frame_idx")
            
            if (post_step is not None and current_step >= post_step and 
                frame_idx is not None and frame_idx not in self._evaluated_frames):
                
                self._evaluated_frames.add(frame_idx)
                vis_dir = Path(self.metadata["output_dir"]) / "eval_results"
                vis_dir.mkdir(parents=True, exist_ok=True)
                
                # 1. Save RGB
                rgb = outputs["rgb"].detach().cpu().numpy().clip(0, 1)
                rgb_u8 = (rgb * 255).astype("uint8")
                Image.fromarray(rgb_u8).save(vis_dir / f"rgb_{frame_idx:05d}.png")
                
                # # 2. Save Depth
                # if outputs["depth"] is not None:
                #     depth = outputs["depth"].detach().cpu().numpy()
                #     d_min, d_max = depth.min(), depth.max()
                #     depth_vis = (depth - d_min) / (d_max - d_min + 1e-8)
                #     depth_vis_u8 = (depth_vis.clip(0, 1) * 255).astype("uint8")[..., 0]
                #     Image.fromarray(depth_vis_u8).save(vis_dir / f"depth_{frame_idx:05d}.png")

                # # 3. Calculate metrics if GT exists
                # frame_metrics = {"psnr": float(metrics_dict.get("psnr", 0))}
                
                # # Check for depth in batch (DepthDataset uses "depth_image")
                # gt_depth_key = "depth_image" if "depth_image" in batch else ("depth" if "depth" in batch else None)
                
                # if gt_depth_key is not None and outputs["depth"] is not None:
                #     # Point Cloud Chamfer Distance
                #     camera = outputs["camera"]
                #     gt_depth = batch[gt_depth_key].to(self.device).squeeze(-1)
                #     pred_depth = outputs["depth"].squeeze(-1)
                #     acc = outputs["accumulation"].squeeze(-1)
                    
                #     # Mask for background
                #     mask_pred = acc > 0.5
                #     mask_gt = gt_depth > 0
                    
                #     # Only calculate CD for pixels within the mask
                #     if "mask" in batch:
                #         gt_mask = batch["mask"].to(self.device).squeeze(-1) > 0.5
                #         mask_gt = mask_gt & gt_mask
                #         mask_pred = mask_pred & gt_mask
                    
                #     if mask_pred.sum() > 0 and mask_gt.sum() > 0:
                #         # Use manual unprojection with intrinsics (Camera Space)
                #         def unproject_depth(depth_map, mask_m, cam):
                #             H_m, W_m = depth_map.shape
                #             # Get intrinsics for this resolution
                #             orig_H, orig_W = cam.height[0].item(), cam.width[0].item()
                #             scale_w = W_m / orig_W
                #             scale_h = H_m / orig_H
                            
                #             fx_m = cam.fx[0].item() * scale_w
                #             fy_m = cam.fy[0].item() * scale_h
                #             cx_m = cam.cx[0].item() * scale_w
                #             cy_m = cam.cy[0].item() * scale_h
                            
                #             y_m, x_m = torch.meshgrid(
                #                 torch.arange(H_m, device=self.device), 
                #                 torch.arange(W_m, device=self.device), 
                #                 indexing='ij'
                #             )
                #             u_m = x_m.float() + 0.5
                #             v_m = y_m.float() + 0.5
                            
                #             z = depth_map[mask_m]
                #             x = (u_m[mask_m] - cx_m) * z / fx_m
                #             y = (v_m[mask_m] - cy_m) * z / fy_m
                #             return torch.stack([x, y, z], dim=-1)

                #         p_pred = unproject_depth(pred_depth, mask_pred, camera)
                #         p_gt = unproject_depth(gt_depth, mask_gt, camera)

                #         # Filter p_pred based on p_gt bounding box
                #         if p_gt.shape[0] > 0 and p_pred.shape[0] > 0:
                #             bbox_min = p_gt.min(dim=0)[0]
                #             bbox_max = p_gt.max(dim=0)[0]
                #             # Keep points in pred that are within gt's bounding box
                #             in_bbox_mask = torch.all((p_pred >= bbox_min) & (p_pred <= bbox_max), dim=-1)
                #             p_pred = p_pred[in_bbox_mask]

                #         # Save combined point clouds as PLY with colors
                #         def save_combined_ply(path, p_pred, p_gt):
                #             p_pred = p_pred.detach().cpu().numpy()
                #             p_gt = p_gt.detach().cpu().numpy()
                #             n_pred = len(p_pred)
                #             n_gt = len(p_gt)
                            
                #             with open(path, 'w') as f:
                #                 header = (
                #                     "ply\n"
                #                     "format ascii 1.0\n"
                #                     f"element vertex {n_pred + n_gt}\n"
                #                     "property float x\n"
                #                     "property float y\n"
                #                     "property float z\n"
                #                     "property uchar red\n"
                #                     "property uchar green\n"
                #                     "property uchar blue\n"
                #                     "end_header\n"
                #                 )
                #                 f.write(header)
                #                 # Pred: Red [255, 0, 0]
                #                 for p in p_pred:
                #                     f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 255 0 0\n")
                #                 # GT: Green [0, 255, 0]
                #                 for p in p_gt:
                #                     f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 0 255 0\n")

                #         save_combined_ply(vis_dir / f"pc_combined_{frame_idx:05d}.ply", p_pred, p_gt)
                        
                #         # Downsample for metric efficiency
                #         if p_pred.shape[0] > 50000:
                #             p_pred_sub = p_pred[torch.randperm(p_pred.shape[0])[:50000]]
                #         else:
                #             p_pred_sub = p_pred
                            
                #         if p_gt.shape[0] > 50000:
                #             p_gt_sub = p_gt[torch.randperm(p_gt.shape[0])[:50000]]
                #         else:
                #             p_gt_sub = p_gt
                        
                #         # Calculate squared CD and take sqrt for Euclidean distance
                #         # Note: pytorch3d chamfer_distance returns squared distance
                #         cd_val, _ = chamfer_distance(p_pred_sub[None], p_gt_sub[None])
                #         cd_val = torch.sqrt(cd_val)
                        
                #         frame_metrics["chamfer_distance"] = float(cd_val)
                #         metrics_dict["chamfer_distance"] = cd_val
                
                # # 4. Update summary JSON
                # summary_path = vis_dir / "eval_summary.json"
                # summary = {}
                # if summary_path.exists():
                #     try:
                #         with open(summary_path, "r") as f:
                #             summary = json.load(f)
                #     except Exception:
                #         pass
                
                # summary[str(frame_idx)] = frame_metrics
                
                # # Compute averages across all evaluated frames so far
                # all_psnrs = [v["psnr"] for k, v in summary.items() if k != "average" and "psnr" in v]
                # all_cds = [v["chamfer_distance"] for k, v in summary.items() if k != "average" and "chamfer_distance" in v]
                
                # summary["average"] = {
                #     "psnr": sum(all_psnrs) / len(all_psnrs) if all_psnrs else 0,
                #     "chamfer_distance": sum(all_cds) / len(all_cds) if all_cds else 0
                # }
                
                # with open(summary_path, "w") as f:
                #     json.dump(summary, f, indent=4)
        
        return metrics_dict
    
    def after_train(self, step: int):
        """Aggregate refinement stats from 2N concat to N base gaussians.

        This model renders a concatenation of static and mobile gaussians (2N).
        For refinement, we aggregate screen-space gradients and sizes back to
        the original N gaussians so densification/culling behaves like the
        original Splatfacto implementation.
        """
        if not self.config.enable_gs_refinement:
            return
        assert step == self.step
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # Expect xys and radii computed on concatenated tensors in get_outputs
            visible_mask_2n = (self.radii > 0).flatten()
            assert self.xys.absgrad is not None
            grads_2n = self.xys.absgrad.detach().norm(dim=-1)

            n = self.means.shape[0]
            # Split static/mobile halves
            grads_static = grads_2n[:n]
            grads_mobile = grads_2n[n:]
            vis_static = visible_mask_2n[:n]
            vis_mobile = visible_mask_2n[n:]

            # Aggregate per-gaussian metrics
            vis_any = vis_static | vis_mobile
            grads_point = torch.maximum(grads_static, grads_mobile)

            # Track moving sums of grad norms and visibility counts (N)
            if getattr(self, "xys_grad_norm", None) is None:
                self.xys_grad_norm = torch.zeros_like(grads_point)
                self.vis_counts = torch.zeros_like(grads_point)
            assert self.vis_counts is not None
            self.vis_counts[vis_any] = self.vis_counts[vis_any] + 1
            self.xys_grad_norm[vis_any] = grads_point[vis_any] + self.xys_grad_norm[vis_any]

            # Update max screen size ratio per gaussian (N)
            if getattr(self, "max_2Dsize", None) is None:
                self.max_2Dsize = torch.zeros(n, device=grads_point.device, dtype=torch.float32)
            radii_static = self.radii.detach()[:n]
            radii_mobile = self.radii.detach()[n:]
            new_radii = torch.maximum(radii_static, radii_mobile)
            size_ratio = new_radii / float(max(self.last_size[0], self.last_size[1]))
            self.max_2Dsize[vis_any] = torch.maximum(self.max_2Dsize[vis_any], size_ratio[vis_any])
    
    def refinement_after(self, optimizers, step):
        """Enable Gaussian refinement using parent implementation"""
        if not self.config.enable_gs_refinement:
            return
        # Ensure screen-size stats exist even if aggregation was skipped
        if getattr(self, "max_2Dsize", None) is None:
            # match current number of points
            self.max_2Dsize = torch.zeros(self.num_points, device=self.device, dtype=torch.float32)
        super().refinement_after(optimizers, step)

