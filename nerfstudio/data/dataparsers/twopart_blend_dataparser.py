from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Type

import json
import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
    CV_TO_GL,
    to4x4,
    read_camK,
    load_3dgs_ply,
    apply_left_se3_to_gaussians,
    save_gaussians_to_ply,
)
from nerfstudio.utils.rich_utils import CONSOLE


def _load_transforms_file(trans_path: Path) -> List[dict]:
    with trans_path.open() as f:
        meta = json.load(f)
    frames = meta if isinstance(meta, list) else meta.get("frames", [])
    # tolerate both keys: "transform" or "transform_matrix"
    out = []
    for fr in frames:
        T = fr.get("transform", fr.get("transform_matrix"))
        if T is None:
            raise RuntimeError(f"Frame missing transform in {trans_path}")
        out.append({"file_path": fr["file_path"], "T": to4x4(T)})
    return out


def _load_all_frames_for_part(part_dir: Path) -> List[dict]:
    """Load all frames for a part. Prefer merged transforms.json; else concat train+eval."""
    train_json = part_dir / "transforms_train.json"
    # eval_json = part_dir / "transforms_eval.json"
    merged_json = part_dir / "transforms.json"
    frames: List[dict] = []
    if merged_json.exists():
        frames = _load_transforms_file(merged_json)
    else:
        if train_json.exists():
            frames.extend(_load_transforms_file(train_json))
        # if eval_json.exists():
        #     frames.extend(_load_transforms_file(eval_json))
    if not frames:
        raise FileNotFoundError(f"No transforms file found in {part_dir}")
    return frames


def _align_two_pose_lists_by_first_frame(frames0: List[dict], frames1: List[dict]) -> Tuple[np.ndarray, np.ndarray, List[dict], np.ndarray, np.ndarray]:
    """Align two pose sequences so their first frames are Identity.

    - Filter to common frames by basename (keep part0 order)
    - Compute N0 = inv(C0[0]), N1 = inv(C1[0])
    - Return aligned sequences and original common sequences
    """
    # build name -> T maps
    name_to_T0 = {Path(fr["file_path"]).name: fr["T"] for fr in frames0}
    name_to_T1 = {Path(fr["file_path"]).name: fr["T"] for fr in frames1}

    # intersection, keep order of part0
    common_frames: List[dict] = []
    c0_list, c1_list = [], []
    dropped0 = 0
    for fr in frames0:
        n = Path(fr["file_path"]).name
        if n in name_to_T1:
            common_frames.append(fr)
            c0_list.append(name_to_T0[n])
            c1_list.append(name_to_T1[n])
        else:
            dropped0 += 1
    dropped1 = len(name_to_T1) - len(common_frames)

    if len(common_frames) == 0:
        raise RuntimeError("No common frames between part0 and part1; cannot align.")
    if dropped0 or dropped1:
        CONSOLE.log(f"[yellow] Filtered to common frames: kept {len(common_frames)}, dropped part0-only={dropped0}, part1-only={dropped1}.")

    C0 = np.stack(c0_list, 0).astype(np.float32)
    C1 = np.stack(c1_list, 0).astype(np.float32)

    # normalize both sequences so first frame is Identity
    N0 = np.linalg.inv(C0[0])
    N1 = np.linalg.inv(C1[0])
    C0_aligned = (N0[None, ...] @ C0)
    C1_aligned = (N1[None, ...] @ C1)

    return C0_aligned.astype(np.float32), C1_aligned.astype(np.float32), common_frames, C0.astype(np.float32), C1.astype(np.float32)


## removed local _apply_left_se3_to_gaussians in favor of shared utils.apply_left_se3_to_gaussians

@dataclass
class TwoPartBlendDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: TwoPartBlendDataParser)

    data: Path = Path()
    """Dataset root. E.g., datasets/usb_01"""

    part0_dir: Path = Path()
    part1_dir: Path = Path()
    """Outputs dirs for two parts. Used to read transforms_{train,eval}.json for consistent poses."""
    output_dir: Path = Path()
    # Optionally provide 3DGS PLYs in each part's camera coords
    part0_ply: Optional[Path] = None
    part1_ply: Optional[Path] = None
    use_all_train_images: bool = True
    """Whether to use all the train images"""
    motion_part: Literal[0, 1] = 0
    """Which part's transforms to use for listing frames and c2w."""

    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    train_split_fraction: float = 0.9
    eval_interval: int = 8

    depth_unit_scale_factor: float = 1.0
    mask_color: Optional[Tuple[float, float, float]] = None
    scene_scale: float = 1.0
    camera_model: Literal["perspective", "PINHOLE"] = "PINHOLE"
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    orientation_method: Literal["pca", "up", "none"] = "up"
    """The method to use for orientation."""
    center_poses: bool = True
    """Whether to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""


@dataclass
class TwoPartBlendDataParser(DataParser):
    config: TwoPartBlendDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        cfg = self.config
        assert cfg.data.exists(), f"Data directory {cfg.data} does not exist."
        assert cfg.part0_dir.exists() and cfg.part1_dir.exists(), "Both part dirs must exist"

        # Load all frames for both parts, align both to Identity at first frame
        frames0_all = _load_all_frames_for_part(cfg.part0_dir)
        frames1_all = _load_all_frames_for_part(cfg.part1_dir)
        C0_aligned, C1_aligned, frames_ref, C0_common_orig, C1_common_orig = _align_two_pose_lists_by_first_frame(frames0_all, frames1_all)
        # Save aligned transforms.json for both parts
        # _save_aligned_transforms_json(cfg.part0_dir, frames_ref, C0_aligned)
        # _save_aligned_transforms_json(cfg.part1_dir, frames_ref, C1_aligned)

        # Convert camera poses to OpenGL convention (cv -> gl)
        C0_gl = (C0_aligned @ CV_TO_GL)
        C1_gl = (C1_aligned @ CV_TO_GL)
        
        # Apply camera normalization (auto orient and center)
        poses = torch.from_numpy(C0_gl)
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=cfg.orientation_method,
            center_poses=cfg.center_poses,
        )
        
        # Scale poses
        scale_factor = 1.0
        if cfg.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= cfg.scale_factor
        
        poses[:, :3, 3] *= scale_factor
        
        # Apply the same transform to C1
        # transform_matrix is [3,4], convert to [4,4]
        T_norm = torch.eye(4, dtype=torch.float32)
        T_norm[:3, :] = transform_matrix
        C1_gl_torch = torch.from_numpy(C1_gl)
        C1_gl_torch = T_norm[None, :, :] @ C1_gl_torch
        C1_gl_torch[:, :3, 3] *= scale_factor
        
        # Convert [N, 3, 4] back to [N, 4, 4] for metadata (model expects full 4x4)
        N = poses.shape[0]
        C0_gl = torch.zeros((N, 4, 4), dtype=torch.float32)
        C0_gl[:, :3, :] = poses
        C0_gl[:, 3, 3] = 1.0
        C0_gl = C0_gl.numpy()
        
        C1_gl_full = torch.zeros((N, 4, 4), dtype=torch.float32)
        C1_gl_full[:, :3, :] = C1_gl_torch[:, :3, :]
        C1_gl_full[:, 3, 3] = 1.0
        C1_gl = C1_gl_full.numpy()

        # Choose listing source and split indices based on aligned frames
        image_filenames: List[Path] = []
        mask_filenames: List[Path] = []
        depth_filenames: List[Path] = []
        poses_np = C0_gl if cfg.motion_part == 1 else C1_gl

        # intrinsics: read from cam_K.txt or infer from first image
        camK_path = cfg.data / "cam_K.txt"
        if camK_path.exists():
            fx, fy, cx, cy = read_camK(camK_path)
        else:
            CONSOLE.log(f"[yellow] cam_K.txt not found in {cfg.data}, will use default center intrinsics")
            fx = fy = cx = cy = None  # type: ignore

        # build filenames & collect masks/depths following frames_ref order
        for fr in frames_ref:
            rel = Path(fr["file_path"])  # e.g., datasets/usb_01/images/00000.png
            if "images" in rel.parts:
                idx = rel.parts.index("images")
                img_rel = Path(*rel.parts[idx:])
                img_path = cfg.data / img_rel.relative_to("images") if img_rel.is_absolute() else cfg.data / img_rel
            else:
                img_path = cfg.data / rel
            if not img_path.exists():
                img_path = cfg.data / Path(fr["file_path"]).name
            image_filenames.append(img_path)
            # mask
            m_rel = Path(str(fr["file_path"]).replace("images", "masks"))
            mask_path = cfg.data / (Path("masks") / Path(m_rel).name)
            if mask_path.exists():
                mask_filenames.append(mask_path)
            # depth
            d_rel = Path(str(fr["file_path"]).replace("images", "depth").replace(".png", ".npy"))
            depth_path = cfg.data / (Path("depth") / Path(d_rel).name)
            if not depth_path.exists():
                d_rel = Path(str(fr["file_path"]).replace("images", "depth").replace(".png", ".npz"))
                depth_path = cfg.data / (Path("depth") / Path(d_rel).name)
            if depth_path.exists():
                depth_filenames.append(depth_path)

        if len(image_filenames) == 0:
            raise RuntimeError("No images found")

        # image size
        with Image.open(image_filenames[0]) as im:
            width0, height0 = im.size

        # Intrinsics tensors
        fx_t = torch.full((len(image_filenames),), float(fx) if fx is not None else 1.0, dtype=torch.float32)
        fy_t = torch.full((len(image_filenames),), float(fy) if fy is not None else 1.0, dtype=torch.float32)
        cx_t = torch.full((len(image_filenames),), float(cx) if cx is not None else (width0 / 2.0), dtype=torch.float32)
        cy_t = torch.full((len(image_filenames),), float(cy) if cy is not None else (height0 / 2.0), dtype=torch.float32)
        height_t = torch.full((len(image_filenames),), int(height0), dtype=torch.int32)
        width_t = torch.full((len(image_filenames),), int(width0), dtype=torch.int32)
        distortion_params = torch.zeros((len(image_filenames), 6), dtype=torch.float32)

        # Split indices
        if cfg.eval_mode == "fraction":
            i_train, i_eval = get_train_eval_split_fraction(image_filenames, cfg.train_split_fraction, cfg.use_all_train_images)
        elif cfg.eval_mode == "filename":
            i_train, i_eval = get_train_eval_split_filename(image_filenames)
        elif cfg.eval_mode == "interval":
            i_train, i_eval = get_train_eval_split_interval(image_filenames, cfg.eval_interval)
        elif cfg.eval_mode == "all":
            i_train, i_eval = get_train_eval_split_all(image_filenames)
        else:
            raise ValueError(f"Unknown eval mode {cfg.eval_mode}")
        indices = i_train if split == "train" else i_eval
        # Slice by split
        image_filenames = [image_filenames[i] for i in indices]
        poses_np = poses_np[indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) == len(frames_ref) else mask_filenames
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) == len(frames_ref) else depth_filenames
        fx_t = fx_t[indices]
        fy_t = fy_t[indices]
        cx_t = cx_t[indices]
        cy_t = cy_t[indices]
        height_t = height_t[indices]
        width_t = width_t[indices]
        distortion_params = distortion_params[indices]

        # Prepare aligned c2w for this split to pass to the model via metadata
        c2w0_split = C0_gl[indices].tolist()
        c2w1_split = C1_gl[indices].tolist()
        frame_paths_split = [frames_ref[i]["file_path"] for i in indices]

        # scene box
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        # camera type
        camera_type = CameraType.PERSPECTIVE

        # frame indices metadata - use GLOBAL indices from the original frame list
        # NOT local indices within this split. This ensures eval frames can correctly
        # index into _T_obj_base which is sized based on all common frames.
        frame_indices = torch.tensor(indices, dtype=torch.long).unsqueeze(-1)
        cam_indices = torch.arange(len(image_filenames), dtype=torch.long)
        cameras = Cameras(
            fx=fx_t,
            fy=fy_t,
            cx=cx_t,
            cy=cy_t,
            distortion_params=distortion_params,
            height=height_t,
            width=width_t,
            camera_to_worlds=torch.from_numpy(poses_np)[:, :3, :4],
            camera_type=camera_type,
            metadata={"frame_index": frame_indices, "cam_idx": cam_indices},
        )

        # Optionally precompute fused gaussians from PLYs using aligned poses and save to disk
        metadata_extra = {
            "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
            "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            "mask_color": self.config.mask_color,
            # pass aligned c2w (full 4x4) for this split
            "c2w_aligned_part0": c2w0_split,
            "c2w_aligned_part1": c2w1_split,
            "frame_file_paths": frame_paths_split,
        }
        if self.config.part0_ply is None:
            self.config.part0_ply = cfg.part0_dir / "object_3dgs.ply"
        if self.config.part1_ply is None:
            self.config.part1_ply = cfg.part1_dir / "object_3dgs.ply"
        if self.config.part0_ply and self.config.part1_ply:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Load both parts' 3DGS using shared loader
            gp0 = load_3dgs_ply(Path(self.config.part0_ply), device)
            gp1 = load_3dgs_ply(Path(self.config.part1_ply), device)
            # Transform both parts' gaussians:
            # 1. Normalize to first frame = I
            # 2. Apply camera normalization transform
            # 3. Apply scale
            N0 = torch.from_numpy(np.linalg.inv(C0_common_orig[0])).to(device)
            N1 = torch.from_numpy(np.linalg.inv(C1_common_orig[0])).to(device)
            
            gp0_canon = apply_left_se3_to_gaussians(gp0, N0, device)
            gp1_canon = apply_left_se3_to_gaussians(gp1, N1, device)
            
            # Apply the same normalization transform as cameras
            T_norm_full = torch.eye(4, dtype=torch.float32)
            T_norm_full[:3, :] = transform_matrix
            T_norm_full = T_norm_full.to(device)
            
            gp0_canon = apply_left_se3_to_gaussians(gp0_canon, T_norm_full, device)
            gp1_canon = apply_left_se3_to_gaussians(gp1_canon, T_norm_full, device)
            
            gp0_canon["means"] = gp0_canon["means"] * scale_factor
            gp1_canon["means"] = gp1_canon["means"] * scale_factor
            scale_log = torch.log(torch.tensor(float(scale_factor), dtype=gp0_canon["scales"].dtype, device=device))
            gp0_canon["scales"] = gp0_canon["scales"] + scale_log
            gp1_canon["scales"] = gp1_canon["scales"] + scale_log
            fused = {k: torch.cat([gp0_canon[k], gp1_canon[k]], dim=0).cpu() for k in ("means", "scales", "quats", "features_dc", "features_rest", "opacities")}
            fused_counts = {"n0": int(gp0["means"].shape[0]), "n1": int(gp1["means"].shape[0])}
            # save to debug folder
            debug_dir = self.config.output_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            fused_ply_path = debug_dir / "twopart_fused_gaussians.ply"
            if not fused_ply_path.exists():
                save_gaussians_to_ply(fused, fused_ply_path)
                CONSOLE.log(f"[green] Saved fused gaussians PLY to {fused_ply_path}")
            
            metadata_extra.update({
                # In-memory fused gaussians for direct model consumption (avoid disk dependency)
                "fused_gaussians": {
                    "means": fused["means"],
                    "scales": fused["scales"],
                    "quats": fused["quats"],
                    "features_dc": fused["features_dc"],
                    "features_rest": fused["features_rest"],
                    "opacities": fused["opacities"],
                },
                "fused_counts": fused_counts,
                "aligned_transforms_paths": {
                    "part0": str(self.config.part0_dir / "transforms_aligned.json"),
                    "part1": str(self.config.part1_dir / "transforms_aligned.json"),
                },
                # Save original first-frame c2w for transforming gaussians if needed downstream
                "original_first_c2w": {
                    "part0": C0_common_orig[0].tolist(),
                    "part1": C1_common_orig[0].tolist(),
                },
                "motion_part": int(cfg.motion_part),
                "output_dir": str(self.config.output_dir),
            })

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata=metadata_extra,
        )
        return dataparser_outputs






def _save_aligned_transforms_json(part_dir: Path, frames_ref: List[dict], c2w_aligned: np.ndarray):
    """Save transforms_aligned.json with given aligned c2w list (3x4 per frame)."""
    out_frames = []
    for i, fr in enumerate(frames_ref):
        T = c2w_aligned[i]
        out_frames.append({"file_path": fr["file_path"], "transform": T[:3, :].tolist()})
    meta = {"frames": out_frames}
    out_path = part_dir / "transforms_aligned.json"
    with out_path.open("w") as f:
        json.dump(meta, f, indent=4)
    CONSOLE.log(f"[green] Saved aligned transforms to {out_path}")
