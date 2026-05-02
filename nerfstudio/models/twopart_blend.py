"""Two-Part Blending Gaussian Splatting Model"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from typing_extensions import Literal
from PIL import Image

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lie_groups import exp_map_SO3xR3
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig

from pytorch3d.transforms import (
    quaternion_raw_multiply,
    matrix_to_quaternion,
)

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics

@dataclass
class TwoPartBlendModelConfig(SplatfactoModelConfig):
    """Configuration for two-part blending model"""
    
    _target: Type = field(default_factory=lambda: TwoPartBlendModel)
    
    # Part motion configuration
    motion_part: Literal[0, 1] = 1
    """Which part represents object motion (0 or 1)"""
    
    # Part weight optimization
    trainable_weights: bool = True
    """Optimize per-point part weights (sigmoid in [0,1])"""
    
    # Rotation interpolation
    so3_interp: Literal["logexp", "none"] = "logexp"
    """SO(3) interpolation method"""
    
    # Object pose optimization
    obj_pose_opt_enabled: bool = True
    """Enable learnable object pose adjustments"""
    
    obj_trans_l2_penalty: float = 0.0
    """L2 penalty for translation adjustments"""
    
    obj_rot_l2_penalty: float = 0.0
    """L2 penalty for rotation adjustments"""

    export_ply_dir: Optional[str] = None
    """Directory for PLY exports"""
    
    # RGB visualization configuration
    visualize_rgb: bool = True
    """Visualize rendered RGB images during training"""
    
    visualize_rgb_every: int = 500
    """Visualization interval in training steps"""
    
    visualize_rgb_dir: Optional[str] = None
    """Directory for RGB visualizations (default: output_dir/rgb_vis)"""
    
    # Gaussian refinement
    enable_gs_refinement: bool = True
    """Enable Gaussian splitting and culling during training"""


class TwoPartBlendModel(SplatfactoModel):
    """Two-part blending model with per-frame pose interpolation"""
    
    config: TwoPartBlendModelConfig
    
    def __init__(self, *args, metadata: Optional[Dict] = None, **kwargs):
        self.metadata = metadata
        self._visualized_step_frame: set[Tuple[int, int]] = set()
        self.motion_part = metadata["motion_part"]
        super().__init__(*args, **kwargs)
        self.config.motion_part = metadata["motion_part"]
    
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get Gaussian parameter groups including part weights"""
        gps = super().get_gaussian_param_groups()
        gps["part_weights"] = [self.gauss_params["part_weights"]]
        return gps
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get parameter groups for optimization"""
        gps = super().get_param_groups()
        if hasattr(self, "obj_pose_adjustment") and self.config.obj_pose_opt_enabled:
            gps["obj_opt"] = [self.obj_pose_adjustment]
        return gps
    
    @property
    def part_weights(self) -> torch.Tensor:
        """Get part weights in [0,1] range, shape [N, 1]"""
        weights = torch.sigmoid(self.gauss_params["part_weights"])
        # Ensure shape is [N, 1] for proper broadcasting
        if weights.ndim == 1:
            weights = weights.unsqueeze(-1)
        return weights

    def populate_modules(self):
        """Initialize model parameters from metadata"""
        super().populate_modules()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load camera poses
        self._load_camera_poses(device)
        
        # Load fused Gaussians
        self._load_fused_gaussians(device)
        
        # Compute object poses and adjustments
        self._init_object_poses(device)
    
    def _load_camera_poses(self, device: str):
        """Load aligned camera poses from metadata"""
        md = self.metadata
        if md and "c2w_aligned_part0" in md and "c2w_aligned_part1" in md:
            self._c2w0 = torch.tensor(md["c2w_aligned_part0"], dtype=torch.float32, device=device)
            self._c2w1 = torch.tensor(md["c2w_aligned_part1"], dtype=torch.float32, device=device)
        else:
            raise ValueError("Aligned camera poses not found in metadata")
    
    def _load_fused_gaussians(self, device: str):
        """Load fused Gaussians from metadata"""
        md = self.metadata
        assert md is not None, "TwoPartBlend metadata is required"
        assert md["fused_gaussians"] is not None, "metadata['fused_gaussians'] is required"
        fused = md["fused_gaussians"]
        counts_info = md["fused_counts"]
        
        # Load Gaussian parameters
        self.gauss_params["means"].data = fused["means"].to(device)
        self.gauss_params["scales"].data = fused["scales"].to(device)
        self.gauss_params["quats"].data = fused["quats"].to(device)

        # Opacity: convert probability [0,1] to logit if necessary
        op = fused["opacities"].to(device)
        if op.min().item() >= 0.0 and op.max().item() <= 1.0:
            op = torch.logit(op.clamp(1e-6, 1 - 1e-6))
        self.gauss_params["opacities"].data = op

        # Color/SH features: adapt to model's sh_degree
        fdc = fused["features_dc"].to(device)
        fr = fused["features_rest"]
        fr = fr.to(device) if isinstance(fr, torch.Tensor) else fr
        if isinstance(fr, torch.Tensor) and fr.dim() == 2:
            n, k = fr.shape
            c = max(k // 3, 0)
            fr = fr.view(n, c, 3).contiguous()
        self.gauss_params["features_dc"].data = fdc
        self.gauss_params["features_rest"].data = fr
        
        # Initialize part weights
        n0 = counts_info["n0"]
        n1 = fused["means"].shape[0] - n0
        p0, p1 = 0.0001, 0.9999
        neg_logit = math.log(p0 / (1 - p0))
        pos_logit = math.log(p1 / (1 - p1))
        logits_init = torch.cat([
            torch.full((n0, 1), neg_logit, device=device),
            torch.full((n1, 1), pos_logit, device=device),
        ], dim=0)
        self.gauss_params["part_weights"] = Parameter(
            logits_init,
            requires_grad=self.config.trainable_weights
        )
    
    def _init_object_poses(self, device: str):
        """Initialize object poses and learnable adjustments"""
        # Compute base object poses: T_obj = C_ref @ inv(C_mot)
                    
        print(f"Using motion part {self.motion_part}")
        if self.motion_part == 1:
            C_ref, C_mot = self._c2w0, self._c2w1
        else:
            C_ref, C_mot = self._c2w1, self._c2w0
        
        R_mot = C_mot[:, :3, :3]
        t_mot = C_mot[:, :3, 3:4]
        R_mot_inv = R_mot.transpose(1, 2)
        t_mot_inv = -R_mot_inv @ t_mot
        T_mot_inv = torch.eye(4, device=device).repeat(len(C_mot), 1, 1)
        T_mot_inv[:, :3, :3] = R_mot_inv
        T_mot_inv[:, :3, 3:4] = t_mot_inv

        # Base object poses (map mot view to ref view): T_obj = C_ref @ inv(C_mot)
        T_obj_base = C_ref @ T_mot_inv

        self.register_buffer("_T_obj_base", T_obj_base)
        # First frame can now be optimized through obj_pose_adjustment
        
        # Learnable pose adjustments (se(3) -> SE(3))
        n_frames = self._T_obj_base.shape[0]
        self.obj_pose_adjustment = Parameter(
            torch.zeros((n_frames, 6), device=device),
            requires_grad=self.config.obj_pose_opt_enabled
        )
        
        # Register gradient hook for monitoring (optional)
        if self.config.obj_pose_opt_enabled:
            print(f"[TwoPartBlend] Object pose optimization enabled: {n_frames} frames") 

    def _select_frame_index(self, camera: Cameras) -> int:
        """Select frame index from camera metadata."""
        assert camera.metadata is not None, "camera.metadata['frame_index'] is required"
        frame_index = camera.metadata["frame_index"]
        if isinstance(frame_index, torch.Tensor):
            assert frame_index.numel() == 1, "Expected a single frame_index"
            return int(frame_index.item())
        return int(frame_index)
    
    def _get_object_transform(self, frame_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get full rigid object transformation for a frame
        
        Args:
            frame_idx: Current frame index
            
        Returns:
            R_obj: Rotation matrix [3,3]
            t_obj: Translation [3,1]
            q_obj: Quaternion [4] (xyzw)
        """
        # Get object pose with learnable adjustment
        T_base = self._T_obj_base[frame_idx]
        adj_3x4 = exp_map_SO3xR3(self.obj_pose_adjustment[frame_idx:frame_idx + 1]).squeeze(0)
        
        # Build Adj matrix in a gradient-friendly way
        # Don't use torch.eye with index assignment as it breaks gradients
        bottom_row = torch.tensor([[0., 0., 0., 1.]], device=adj_3x4.device, dtype=adj_3x4.dtype)
        Adj = torch.cat([adj_3x4, bottom_row], dim=0)  # [4, 4]
        
        T = T_base @ Adj
        R_obj = T[:3, :3]
        t_obj = T[:3, 3:4]
        
        # Convert rotation to quaternion
        q_obj = matrix_to_quaternion(R_obj[None])[0]  # [4] in wxyz
        # Convert to xyzw
        q_obj = torch.stack([q_obj[1], q_obj[2], q_obj[3], q_obj[0]])
        
        return R_obj, t_obj, q_obj
    
    @torch.no_grad()
    def visualize_training_rgb(self, outputs: Dict, camera: Cameras, step: int) -> None:
        """Visualize rendered RGB during training
        
        Args:
            outputs: Model outputs containing 'rgb' tensor
            camera: Camera used for rendering
            step: Current training step
        """
        # Get frame index
        frame_idx = self._select_frame_index(camera)
        
        # Check if already visualized this step-frame combination
        key = (step, frame_idx)
        if key in self._visualized_step_frame:
            return
        self._visualized_step_frame.add(key)
        
        # Get RGB output [H, W, 3] in range [0, 1]
        rgb = outputs["rgb"].detach()
        
        # Convert to uint8 image
        rgb_np = (rgb.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        
        # Create output directory
        if self.config.visualize_rgb_dir:
            vis_dir = Path(self.config.visualize_rgb_dir)
        else:
            vis_dir = Path(self.metadata["output_dir"]) / "rgb_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        img_fname = vis_dir / f"train_step{step:06d}_frame{frame_idx:05d}.png"
        Image.fromarray(rgb_np).save(img_fname)
        
        print(f"[TwoPartBlend] Saved training RGB to {img_fname.name}")
        

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Forward pass with concatenated rendering (SplArt-style)
        
        Concatenates static and mobile gaussians and renders once:
        - Static gaussians: original positions with opacity * (1-w)
        - Mobile gaussians: transformed positions with opacity * w
        - Render all together for proper depth sorting and alpha blending
        """
        assert camera.shape[0] == 1, "Only one camera at a time"
        
        camera = camera.to(self.device)
        
        # Get background color
        # Note: camera optimization is handled by the base class in super().get_outputs()
        if self.config.background_color == "random":
            background = torch.rand(3, device=self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            background = self.background_color.to(self.device)
        
        # Get frame index and part weights [N, 1]
        frame_idx = self._select_frame_index(camera)
        w = self.part_weights  # [N, 1]
        if self.metadata["motion_part"] == 0:
            w = 1 - w
        
        # Handle crop box
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), 
                    int(camera.height.item()), 
                    background
                )
        else:
            crop_ids = None
        
        # Get full rigid transformation
        R_obj, t_obj, q_obj = self._get_object_transform(frame_idx)
        
        # Transform all Gaussians
        means_transformed = (R_obj @ self.means.T).T + t_obj.T
        q_obj_wxyz = torch.stack([q_obj[3], q_obj[0], q_obj[1], q_obj[2]])
        quats_wxyz = torch.stack(
            [self.quats[..., 3], self.quats[..., 0], self.quats[..., 1], self.quats[..., 2]],
            dim=-1,
        )
        quats_transformed_wxyz = quaternion_raw_multiply(q_obj_wxyz[None, :], quats_wxyz)
        quats_transformed = torch.stack(
            [
                quats_transformed_wxyz[..., 1],
                quats_transformed_wxyz[..., 2],
                quats_transformed_wxyz[..., 3],
                quats_transformed_wxyz[..., 0],
            ],
            dim=-1,
        )
        
        # Concatenate static and mobile gaussians
        # Keep gradients for transformed positions/orientations to optimize obj_pose_adjustment
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
        if self.config.output_depth_during_training or not self.training:
            depth_im = rasterize_gaussians(
                xys, depths, radii, conics, num_tiles_hit,
                depths[:, None].repeat(1, 3),
                opacities_render,
                H, W, BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]
            depth_im = torch.where(alpha_out > 0, depth_im / alpha_out, depth_im.detach().max())
        
        outputs = {"rgb": rgb, "depth": depth_im, "accumulation": alpha_out, "background": background}
        
        # Visualize training RGB at specified intervals
        # Use self.step from parent class which is properly managed by the training loop
        if self.training and self.config.visualize_rgb:
            if self.step % self.config.visualize_rgb_every == 0:
                self.visualize_training_rgb(outputs, camera, self.step)
        
        return outputs
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Compute losses with object pose regularization"""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        
        # Add L2 regularization for object pose adjustments
        if self.training and self.config.obj_pose_opt_enabled and hasattr(self, "obj_pose_adjustment"):
            t_norm = self.obj_pose_adjustment[:, :3].norm(dim=-1).mean()
            r_norm = self.obj_pose_adjustment[:, 3:].norm(dim=-1).mean()
            loss_dict["obj_opt_regularizer"] = (
                t_norm * self.config.obj_trans_l2_penalty + 
                r_norm * self.config.obj_rot_l2_penalty
            )
        
        return loss_dict
    
    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute metrics including gradient monitoring"""
        metrics_dict = super().get_metrics_dict(outputs, batch)
        
        # Add object pose adjustment monitoring metrics
        if self.training and self.config.obj_pose_opt_enabled and hasattr(self, "obj_pose_adjustment"):
            with torch.no_grad():
                # Parameter statistics for monitoring optimization progress
                # metrics_dict["obj_adj_max"] = self.obj_pose_adjustment.abs().max()
                metrics_dict["obj_adj_mean"] = self.obj_pose_adjustment.abs().mean()
                # metrics_dict["obj_adj_trans_norm"] = self.obj_pose_adjustment[:, :3].norm(dim=-1).mean()
                # metrics_dict["obj_adj_rot_norm"] = self.obj_pose_adjustment[:, 3:].norm(dim=-1).mean()
        
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
        """Skip Gaussian refinement if disabled"""
        if not self.config.enable_gs_refinement:
            return
        super().refinement_after(optimizers, step)
    
    @torch.no_grad()
    def export_part_weights(self, output_dir: Optional[Path] = None) -> Path:
        """Export part weights to text file
        
        Args:
            output_dir: Output directory (default: config.export_ply_dir)
            
        Returns:
            Path to exported file
        """
        if output_dir is None:
            if self.config.export_ply_dir:
                output_dir = Path(self.config.export_ply_dir)
            else:
                output_dir = Path("outputs") / "rotated_point_clouds"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get part weights
        w_all = self.part_weights.squeeze(-1)
        w_all_np = w_all.detach().cpu().numpy().astype(np.float32)
        
        # Save to text file
        weight_fname = output_dir / "3dgs_part_weight.txt"
        np.savetxt(
            str(weight_fname),
            w_all_np,
            fmt='%.6f',
            header=f'Part weights for {len(w_all_np)} gaussians (sigmoid output, [0,1] range)\n'
                   f'Values closer to 0 belong more to part0, closer to 1 belong more to part1',
            comments='# '
        )
        
        print(f"[TwoPartBlend] Exported part weights to {weight_fname}")
        return weight_fname
