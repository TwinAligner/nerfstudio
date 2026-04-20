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
""" Data parser utils for nerfstudio datasets. """

import math
import os
from typing import List, Tuple, Any
from pathlib import Path
import torch
import numpy as np
from plyfile import PlyElement, PlyData
import warnings
warnings.filterwarnings('ignore')


def get_train_eval_split_fraction(image_filenames: List, train_split_fraction: float, use_all_train_images: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split fraction based on the number of images and the train split fraction.

    Args:
        image_filenames: list of image filenames
        train_split_fraction: fraction of images to use for training
    """

    # filter image_filenames and poses based on train/eval split percentage
    num_images = len(image_filenames)
    num_train_images = math.ceil(num_images * train_split_fraction)
    num_eval_images = num_images - num_train_images
    i_all = np.arange(num_images)
    i_train = np.linspace(
        0, num_images - 1, num_train_images, dtype=int
    )  # equally spaced training images starting and ending at 0 and num_images-1
    i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
    assert len(i_eval) == num_eval_images
    if use_all_train_images:
        i_train = i_all.copy()
    return i_train, i_eval


def get_train_eval_split_filename(image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on the filename of the images.

    Args:
        image_filenames: list of image filenames
    """

    num_images = len(image_filenames)
    basenames = [os.path.basename(image_filename) for image_filename in image_filenames]
    i_all = np.arange(num_images)
    i_train = []
    i_eval = []
    for idx, basename in zip(i_all, basenames):
        # check the frame index
        if "train" in basename:
            i_train.append(idx)
        elif "eval" in basename:
            i_eval.append(idx)
        else:
            raise ValueError("frame should contain train/eval in its name to use this eval-frame-index eval mode")

    return np.array(i_train), np.array(i_eval)


def get_train_eval_split_interval(image_filenames: List, eval_interval: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on the interval of the images.

    Args:
        image_filenames: list of image filenames
        eval_interval: interval of images to use for eval
    """

    num_images = len(image_filenames)
    all_indices = np.arange(num_images)
    train_indices = all_indices[all_indices % eval_interval != 0]
    eval_indices = all_indices[all_indices % eval_interval == 0]
    i_train = train_indices
    i_eval = eval_indices

    return i_train, i_eval


def get_train_eval_split_all(image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split where all indices are used for both train and eval.

    Args:
        image_filenames: list of image filenames
    """
    num_images = len(image_filenames)
    i_all = np.arange(num_images)
    i_train = i_all
    i_eval = i_all
    return i_train, i_eval


# -------------------------
# Shared dataparsers helpers
# -------------------------

# OpenCV (x right, y down, z forward) → OpenGL (x right, y up, z backward)
CV_TO_GL = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def to4x4(T: np.ndarray) -> np.ndarray:
    """Ensure homogeneous [4,4] from [3,4] or [4,4]."""
    T = np.array(T, dtype=np.float32)
    if T.shape == (3, 4):
        T = np.vstack([T, np.array([0, 0, 0, 1], dtype=np.float32)])
    return T


def read_camK(camK_path: "Path") -> Tuple[float, float, float, float]:
    """Read simple pinhole intrinsics from cam_K.txt (fx, fy, cx, cy)."""
    with camK_path.open("r") as f:
        vals = [list(map(float, line.strip().split())) for line in f if line.strip()]
    fx = vals[0][0]
    fy = vals[1][1]
    cx = vals[0][2]
    cy = vals[1][2]
    return fx, fy, cx, cy


def safe_get(props: dict, names: List[str], default=None):
    """Return the first present key's value among names; else default.

    This helper remains for robust PLY parsing across varied field names.
    """
    for n in names:
        if n in props:
            return props[n]
    return default


def load_3dgs_ply(
    ply_path: Path,
    device: Any,
    rest_format: str = "auto",
) -> dict:
    """Load a 3DGS PLY file into tensors.

    Returns keys:
    - means [N,3]
    - scales or scales_log [N,3] (always log-scale returned under key 'scales')
    - quats [N,4] (xyzw)
    - opacities [N,1]
    - features_dc [N,3]
    - features_rest: shape depends on input format
      - If PLY uses grouped SH (f_rest_* triplets), returns [N,K,3]
      - If PLY uses flat SH (f_rest_* as single channel), returns [N,M]
    """

    ply = PlyData.read(str(ply_path))
    v = ply.elements[0].data
    import torch  # local import to avoid circulars

    # Make all fields contiguous and cast to float32 to avoid stride issues
    props = {}
    for name in v.dtype.names:
        arr = np.array(v[name], copy=True)  # force contiguous copy
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        props[name] = torch.tensor(arr, dtype=torch.float32, device=device)

    # positions
    x = safe_get(props, ["x"]) ; y = safe_get(props, ["y"]) ; z = safe_get(props, ["z"]) 
    assert x is not None and y is not None and z is not None, "PLY missing x/y/z"
    means = torch.stack([x, y, z], dim=-1)

    # opacity
    opacity = safe_get(props, ["opacity", "opacities"])
    assert opacity is not None, "PLY missing opacity"
    opacities = opacity.reshape(-1, 1)

    # scales → log-scale
    s0 = safe_get(props, ["scale_0", "scale0", "s0"]) ; s1 = safe_get(props, ["scale_1", "scale1", "s1"]) ; s2 = safe_get(props, ["scale_2", "scale2", "s2"]) 
    assert s0 is not None and s1 is not None and s2 is not None, "PLY missing scale_0/1/2"
    S = torch.stack([s0, s1, s2], dim=-1)
    ratio_nonpos = (S <= 0).float().mean()
    if ratio_nonpos > 0.5:
        scales_log = S
    else:
        scales_log = torch.log(S.clamp_min(1e-8))

    # orientation quats (convert wxyz → xyzw)
    r0 = safe_get(props, ["rot_0", "q0"]) ; r1 = safe_get(props, ["rot_1", "q1"]) ; r2 = safe_get(props, ["rot_2", "q2"]) ; r3 = safe_get(props, ["rot_3", "q3"]) 
    assert r0 is not None and r1 is not None and r2 is not None and r3 is not None, "PLY missing rot_0..3"
    quats_xyzw = torch.stack([r1, r2, r3, r0], dim=-1)
    quats_xyzw = quats_xyzw / quats_xyzw.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # features (DC + rest)
    fdc0 = safe_get(props, ["f_dc_0"]) ; fdc1 = safe_get(props, ["f_dc_1"]) ; fdc2 = safe_get(props, ["f_dc_2"]) 
    if fdc0 is not None and fdc1 is not None and fdc2 is not None:
        features_dc = torch.stack([fdc0, fdc1, fdc2], dim=-1)
        rest_keys = sorted([k for k in props.keys() if str(k).startswith("f_rest_")], key=lambda x: int(str(x).split("_")[-1]))
        if rest_keys:
            rest_tensor = torch.stack([props[k] for k in rest_keys], dim=-1)
            if rest_format == "grouped":
                # expect triplets
                K = rest_tensor.shape[-1] // 3
                features_rest = rest_tensor.reshape(means.shape[0], K, 3)
            elif rest_format == "flat":
                features_rest = rest_tensor
            else:  # auto
                features_rest = rest_tensor if rest_tensor.shape[-1] % 3 != 0 else rest_tensor.reshape(means.shape[0], rest_tensor.shape[-1] // 3, 3)
        else:
            # use 3D empty tensor for compatibility with rendering code expecting [..., K, 3]
            features_rest = torch.zeros((means.shape[0], 0, 3), dtype=means.dtype, device=device)
    else:
        raise NotImplementedError("No features_dc found in PLY file")
        # fallback RGB/colors → DC


    return {
        "means": means,
        "scales": scales_log,
        "quats": quats_xyzw,
        "opacities": opacities,
        "features_dc": features_dc,
        "features_rest": features_rest,
    }


def apply_left_se3_to_gaussians(gauss: dict, T: "Any", device: Any) -> dict:
    """Apply a left-multiplication SE(3) transform to gaussian parameters.

    - T: [4,4] transform (torch.Tensor)
    - gauss requires keys: means [N,3], scales (log) [N,3], quats (xyzw) [N,4]
    - scales unchanged; means rotated and translated; orientations left-multiplied
    """
    from pytorch3d.transforms import matrix_to_quaternion  # type: ignore
    from pytorch3d.transforms import quaternion_raw_multiply  # type: ignore[attr-defined]

    R = T[:3, :3]
    t = T[:3, 3]

    means = gauss["means"].to(device)
    means_t = (R @ means.T).T + t[None, :]

    scales_log = gauss["scales"].to(device)
    quats = gauss["quats"].to(device)

    q_R_wxyz = matrix_to_quaternion(R[None, ...])  # [1,4] wxyz
    quats_wxyz = torch.stack(
        [quats[..., 3], quats[..., 0], quats[..., 1], quats[..., 2]], dim=-1
    )
    quats_t_wxyz = quaternion_raw_multiply(q_R_wxyz.repeat(quats.shape[0], 1), quats_wxyz)
    quats_t = torch.stack(
        [quats_t_wxyz[..., 1], quats_t_wxyz[..., 2], quats_t_wxyz[..., 3], quats_t_wxyz[..., 0]],
        dim=-1,
    )
    quats_t = quats_t / quats_t.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    return {
        "means": means_t,
        "scales": scales_log,
        "quats": quats_t,
        "features_dc": gauss["features_dc"].to(device),
        "features_rest": gauss["features_rest"].to(device) if hasattr(gauss["features_rest"], "to") else gauss["features_rest"],
        "opacities": gauss["opacities"].to(device),
    }

def save_gaussians_to_ply(gauss: dict, out_path: Path) -> None:
        """Save gaussians to a PLY file with fields used by 3DGS: x/y/z, opacity,
        scale_0..2, rot_0..3 (wxyz), f_dc_0..2 and f_rest_*.
        Input gauss keys: means [N,3], scales (log) [N,3], quats (xyzw) [N,4],
        features_dc [N,3], features_rest [N,M], opacities [N,1].
        """

        # to numpy float32
        def to_np(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            return x.astype(np.float32)

        means = to_np(gauss["means"])  # [N,3]
        scales = to_np(gauss["scales"])  # [N,3] convert from log-scale to linear for 3DGS PLY
        quats_xyzw = to_np(gauss["quats"])  # [N,4]
        # normalize & convert to wxyz
        qn = np.linalg.norm(quats_xyzw, axis=1, keepdims=True)
        quats_xyzw = quats_xyzw / np.clip(qn, 1e-8, None)
        rot_wxyz = np.stack([quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]], axis=1)

        features_dc = to_np(gauss["features_dc"])  # [N,3]
        features_rest = to_np(gauss.get("features_rest")) if gauss.get("features_rest") is not None else np.zeros((means.shape[0], 0), dtype=np.float32)
        # Handle both flat [N,M] and grouped [N,K,3] formats
        if features_rest.ndim == 3:
            # Reshape from [N,K,3] to [N,K*3] for flat storage
            features_rest = features_rest.reshape(features_rest.shape[0], -1)
        opac = gauss["opacities"]
        if isinstance(opac, torch.Tensor):
            opac = torch.sigmoid(opac).detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            opac = 1 / (1 + np.exp(-to_np(gauss["opacities"]).reshape(-1)))

        N = means.shape[0]
        dtype_list = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ]
        rest_names: List[str] = []
        if features_rest.shape[1] > 0:
            for i in range(features_rest.shape[1]):
                name = f"f_rest_{i}"
                dtype_list.append((name, "f4"))
                rest_names.append(name)

        arr = np.empty(N, dtype=dtype_list)
        arr["x"] = means[:, 0]
        arr["y"] = means[:, 1]
        arr["z"] = means[:, 2]
        arr["opacity"] = opac
        arr["scale_0"] = scales[:, 0]
        arr["scale_1"] = scales[:, 1]
        arr["scale_2"] = scales[:, 2]
        arr["rot_0"] = rot_wxyz[:, 0]
        arr["rot_1"] = rot_wxyz[:, 1]
        arr["rot_2"] = rot_wxyz[:, 2]
        arr["rot_3"] = rot_wxyz[:, 3]
        arr["f_dc_0"] = features_dc[:, 0]
        arr["f_dc_1"] = features_dc[:, 1]
        arr["f_dc_2"] = features_dc[:, 2]
        for i, name in enumerate(rest_names):
            arr[name] = features_rest[:, i]
        
        el = PlyElement.describe(arr, "vertex")
        PlyData([el]).write(str(out_path))