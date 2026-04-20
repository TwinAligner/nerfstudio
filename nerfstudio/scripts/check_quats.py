"""Quaternion consistency checker for 3DGS exports.

Usage examples:

  # 1) 检查融合坐标系（训练/渲染实际使用的坐标系）：不要 --world-frame 导出的 PLY
  python -m nerfstudio.scripts.check_quats \
      --load-config /path/to/twopart_blend/config.yml \
      --ply /path/to/twopart_blend/object_3dgs.ply \
      --mode canonical

  # 2) 检查世界坐标系：针对带 --world-frame 导出的 PLY
  python -m nerfstudio.scripts.check_quats \
      --load-config /path/to/twopart_blend/config.yml \
      --ply /path/to/twopart_blend/object_3dgs.ply \
      --mode world

在运行前请确保将本仓库的 nerfstudio 路径加入 PYTHONPATH，例如：
  export PYTHONPATH=/home/daihang/workspace/articulation/mono-artgs/reconstruction/nerfstudio:$PYTHONPATH
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.data.utils.dataparsers_utils import load_3dgs_ply


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, 1e-8, None)


def quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for XYZW quaternions. Supports broadcasting on leading dims.

    q_out = q1 * q2 (apply q1 first, then q2 if used for vector transform as left-multiply)
    """
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    out = np.stack([x, y, z, w], axis=-1)
    return normalize_quat_xyzw(out)


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to WXYZ quaternion.
    Returns [w,x,y,z]. Accepts [...,3,3] and returns [...,4].
    """
    R = np.asarray(R, dtype=np.float32)
    if R.ndim == 2:
        R = R[None, ...]
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22

    w = np.empty_like(trace)
    x = np.empty_like(trace)
    y = np.empty_like(trace)
    z = np.empty_like(trace)

    mask = trace > 0.0
    s = np.zeros_like(trace)
    s[mask] = 0.5 / np.sqrt(trace[mask] + 1.0)
    w[mask] = 0.25 / s[mask]
    x[mask] = (m21[mask] - m12[mask]) * s[mask]
    y[mask] = (m02[mask] - m20[mask]) * s[mask]
    z[mask] = (m10[mask] - m01[mask]) * s[mask]

    mask0 = (~mask) & (m00 > m11) & (m00 > m22)
    s[mask0] = 2.0 * np.sqrt(np.maximum(1.0 + m00[mask0] - m11[mask0] - m22[mask0], 0.0))
    w[mask0] = (m21[mask0] - m12[mask0]) / s[mask0]
    x[mask0] = 0.25 * s[mask0]
    y[mask0] = (m01[mask0] + m10[mask0]) / s[mask0]
    z[mask0] = (m02[mask0] + m20[mask0]) / s[mask0]

    mask1 = (~mask) & (~mask0) & (m11 > m22)
    s[mask1] = 2.0 * np.sqrt(np.maximum(1.0 + m11[mask1] - m00[mask1] - m22[mask1], 0.0))
    w[mask1] = (m02[mask1] - m20[mask1]) / s[mask1]
    x[mask1] = (m01[mask1] + m10[mask1]) / s[mask1]
    y[mask1] = 0.25 * s[mask1]
    z[mask1] = (m12[mask1] + m21[mask1]) / s[mask1]

    mask2 = (~mask) & (~mask0) & (~mask1)
    s[mask2] = 2.0 * np.sqrt(np.maximum(1.0 + m22[mask2] - m00[mask2] - m11[mask2], 0.0))
    w[mask2] = (m10[mask2] - m01[mask2]) / s[mask2]
    x[mask2] = (m02[mask2] + m20[mask2]) / s[mask2]
    y[mask2] = (m12[mask2] + m21[mask2]) / s[mask2]
    z[mask2] = 0.25 * s[mask2]

    q = np.stack([w, x, y, z], axis=-1).astype(np.float32)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    if q.shape[0] == 1:
        q = q[0]
    return q


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    q_wxyz = rotmat_to_quat_wxyz(R)
    if q_wxyz.ndim == 1:
        q_wxyz = q_wxyz[None, ...]
    q_xyzw = np.stack([q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]], axis=1)
    q_xyzw = q_xyzw / (np.linalg.norm(q_xyzw, axis=-1, keepdims=True) + 1e-8)
    if q_xyzw.shape[0] == 1:
        q_xyzw = q_xyzw[0]
    return q_xyzw


def geodesic_deg_from_xyzw(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Angle (deg) between two rotations represented by XYZW quaternions; batch-wise."""
    q_a = normalize_quat_xyzw(q_a)
    q_b = normalize_quat_xyzw(q_b)
    # Convert to rotation matrices
    def quat_to_rot(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
        R[..., 0, 0] = 1 - 2 * (yy + zz)
        R[..., 0, 1] = 2 * (xy - wz)
        R[..., 0, 2] = 2 * (xz + wy)
        R[..., 1, 0] = 2 * (xy + wz)
        R[..., 1, 1] = 1 - 2 * (xx + zz)
        R[..., 1, 2] = 2 * (yz - wx)
        R[..., 2, 0] = 2 * (xz - wy)
        R[..., 2, 1] = 2 * (yz + wx)
        R[..., 2, 2] = 1 - 2 * (xx + yy)
        return R
    Ra = quat_to_rot(q_a)
    Rb = quat_to_rot(q_b)
    # Relative rotation: R_rel = R_a^T @ R_b
    Rrel = np.einsum("...ij,...jk->...ik", Ra.transpose(0, 2, 1), Rb)
    tr = np.clip((np.trace(Rrel, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    ang = np.degrees(np.arccos(tr))
    return ang


def _summarize_scale_consistency(scales_model_log: np.ndarray, scales_ply_log: np.ndarray, label: str) -> None:
    """Print simple stats comparing model vs PLY log-scales (and linear scales).

    Expects shapes [N, 3] in log-domain.
    """
    n = min(scales_model_log.shape[0], scales_ply_log.shape[0])
    sm = scales_model_log[:n]
    sp = scales_ply_log[:n]
    diff_log = np.abs(sm - sp)
    # Linear domain for intuition
    sm_lin = np.exp(sm)
    sp_lin = np.exp(sp)
    # Relative error in linear domain
    rel_lin = np.abs(sm_lin - sp_lin) / (np.maximum(sp_lin, 1e-8))
    print(
        f"{label} scales (log): mean={float(diff_log.mean()):.6f}, median={float(np.median(diff_log)):.6f}, max={float(diff_log.max()):.6f}"
    )
    print(
        f"{label} scales (linear, relative): mean={float(rel_lin.mean()):.6f}, median={float(np.median(rel_lin)):.6f}, max={float(rel_lin.max()):.6f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Check quaternion consistency between model and exported PLY")
    ap.add_argument("--load-config", type=Path, required=True, help="Path to config.yml used by eval_setup")
    ap.add_argument("--ply", type=Path, required=True, help="Path to exported object_3dgs.ply to check")
    ap.add_argument("--mode", type=str, choices=["canonical", "world"], default="canonical",
                    help="canonical: compare in fused/canonical coords; world: map model quats to world then compare")
    ap.add_argument("--disable-per-part", action="store_true",
                    help="In world mode, do not apply per-part first-frame c2w (useful to check old exports).")
    ap.add_argument("--interpret-rot", type=str, choices=["auto", "wxyz", "xyzw"], default="auto",
                    help="How to interpret PLY rot_0..3. auto=use loader (wxyz). wxyz=explicit WXYZ; xyzw=explicit XYZW.")
    ap.add_argument("--diagnose", action="store_true",
                    help="Also estimate the single fixed rotation that best aligns model quats to PLY quats.")
    args = ap.parse_args()

    assert args.load_config.exists(), f"Config not found: {args.load_config}"
    assert args.ply.exists(), f"PLY not found: {args.ply}"

    _, pipeline, _, _ = eval_setup(args.load_config)
    model = pipeline.model

    # Load or parse PLY quaternions
    if args.interpret_rot == "auto":
        # use shared loader: expects WXYZ in file, returns XYZW
        g = load_3dgs_ply(args.ply, device=torch.device("cpu"))
        q_ply_xyzw = g["quats"].cpu().numpy()
    else:
        from plyfile import PlyData
        ply = PlyData.read(str(args.ply))
        v = ply.elements[0].data
        r0 = np.asarray(v["rot_0"], dtype=np.float32)
        r1 = np.asarray(v["rot_1"], dtype=np.float32)
        r2 = np.asarray(v["rot_2"], dtype=np.float32)
        r3 = np.asarray(v["rot_3"], dtype=np.float32)
        if args.interpret_rot == "wxyz":
            # file stores WXYZ → convert to XYZW
            q_ply_xyzw = np.stack([r1, r2, r3, r0], axis=1).astype(np.float32)
        else:  # xyzw
            # file stores XYZW directly
            q_ply_xyzw = np.stack([r0, r1, r2, r3], axis=1).astype(np.float32)
        q_ply_xyzw = normalize_quat_xyzw(q_ply_xyzw)
    n_ply = q_ply_xyzw.shape[0]

    # Model quats in XYZW
    q_model_xyzw = model.quats.detach().cpu().numpy()
    n_model = q_model_xyzw.shape[0]

    print(f"PLY gaussians: {n_ply}, Model gaussians: {n_model}")
    if n_ply != n_model:
        print("[Warning] Count mismatch; results may be unreliable.")

    if args.mode == "canonical":
        # Direct comparison in fused/canonical coordinates
        nq = min(n_ply, n_model)
        qm = q_model_xyzw[:nq]
        qp = q_ply_xyzw[:nq]
        ang = geodesic_deg_from_xyzw(qm, qp)
        print("Canonical/fused coord export: mean={:.6f}°, median={:.6f}°, max={:.6f}°".format(
            float(np.mean(ang)), float(np.median(ang)), float(np.max(ang))
        ))
        # Scale consistency in canonical space (compare log-scales)
        try:
            g_scales = load_3dgs_ply(args.ply, device=torch.device("cpu"))
            scales_ply_log = g_scales["scales"].cpu().numpy()
            scales_model_log = model.scales.detach().cpu().numpy()
            _summarize_scale_consistency(scales_model_log, scales_ply_log, label="Canonical")
        except Exception as e:
            print(f"[Warning] Failed to check scales (canonical): {e}")
        if args.diagnose:
            # Estimate fixed rotation R such that qp ≈ R * qm
            def quat_to_rot(q: np.ndarray) -> np.ndarray:
                x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
                xx, yy, zz = x * x, y * y, z * z
                xy, xz, yz = x * y, x * z, y * z
                wx, wy, wz = w * x, w * y, w * z
                R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
                R[..., 0, 0] = 1 - 2 * (yy + zz)
                R[..., 0, 1] = 2 * (xy - wz)
                R[..., 0, 2] = 2 * (xz + wy)
                R[..., 1, 0] = 2 * (xy + wz)
                R[..., 1, 1] = 1 - 2 * (xx + zz)
                R[..., 1, 2] = 2 * (yz - wx)
                R[..., 2, 0] = 2 * (xz - wy)
                R[..., 2, 1] = 2 * (yz + wx)
                R[..., 2, 2] = 1 - 2 * (xx + yy)
                return R
            Rm = quat_to_rot(qm)
            Rp = quat_to_rot(qp)
            # Orthogonal Procrustes: find R that minimizes sum ||R Rm_i - Rp_i||^2
            M = np.zeros((3, 3), dtype=np.float32)
            for i in range(Rm.shape[0]):
                M += Rp[i] @ Rm[i].T
            U, S, Vt = np.linalg.svd(M)
            Rbest = U @ Vt
            if np.linalg.det(Rbest) < 0:
                U[:, -1] *= -1
                Rbest = U @ Vt
            # Axis-angle
            cos_theta = (np.trace(Rbest) - 1.0) * 0.5
            cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
            theta = float(np.degrees(np.arccos(cos_theta)))
            axis = np.array([
                Rbest[2,1] - Rbest[1,2],
                Rbest[0,2] - Rbest[2,0],
                Rbest[1,0] - Rbest[0,1],
            ], dtype=np.float32)
            n = float(np.linalg.norm(axis))
            axis = (axis / n) if n > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
            print("Diagnose: fixed R aligning model->PLY: angle={:.6f}°, axis={}".format(theta, axis.tolist()))
        return

    # World-frame: map model quats from normalized/canonical coords to original world frame
    dp_outputs = pipeline.datamanager.train_dataparser_outputs
    R_dp = dp_outputs.dataparser_transform[:3, :3].detach().cpu().numpy().astype(np.float32)
    R_inv = np.linalg.inv(R_dp)
    q_Rinv_xyzw = rotmat_to_quat_xyzw(R_inv)

    # Left-multiply by R_inv for all gaussians
    q_world = quat_mul_xyzw(np.broadcast_to(q_Rinv_xyzw, q_model_xyzw.shape), q_model_xyzw)

    # If TwoPartBlend metadata is present, also left-multiply per-part first-frame c2w
    if not args.disable_per_part:
        meta = getattr(dp_outputs, "metadata", None)
        if isinstance(meta, dict) and ("original_first_c2w" in meta) and ("fused_counts" in meta):
            n0 = int(meta["fused_counts"].get("n0", 0))
            try:
                c2w0 = np.asarray(meta["original_first_c2w"]["part0"], dtype=np.float32)
                c2w1 = np.asarray(meta["original_first_c2w"]["part1"], dtype=np.float32)
                R0 = c2w0[:3, :3].astype(np.float32)
                R1 = c2w1[:3, :3].astype(np.float32)
                qR0_xyzw = rotmat_to_quat_xyzw(R0)
                qR1_xyzw = rotmat_to_quat_xyzw(R1)
                if n0 > 0:
                    q_world[:n0] = quat_mul_xyzw(np.broadcast_to(qR0_xyzw, q_world[:n0].shape), q_world[:n0])
                if n0 < q_world.shape[0]:
                    q_world[n0:] = quat_mul_xyzw(np.broadcast_to(qR1_xyzw, q_world[n0:].shape), q_world[n0:])
            except Exception as e:
                print(f"[Warning] Failed to apply per-part first-frame c2w: {e}")

    nq = min(n_ply, q_world.shape[0])
    qm = q_world[:nq]
    qp = q_ply_xyzw[:nq]
    ang = geodesic_deg_from_xyzw(qm, qp)
    print("World-frame export: mean={:.6f}°, median={:.6f}°, max={:.6f}°".format(
        float(np.mean(ang)), float(np.median(ang)), float(np.max(ang))
    ))
    # Scale consistency in world space: subtract ln(s) from model's log-scales
    try:
        dp_outputs = pipeline.datamanager.train_dataparser_outputs
        s = float(dp_outputs.dataparser_scale)
        # Load PLY log-scales (loader returns log-domain)
        g_scales = load_3dgs_ply(args.ply, device=torch.device("cpu"))
        scales_ply_log = g_scales["scales"].cpu().numpy()
        # Map model log-scales to world by removing global scale
        scales_model_log_world = model.scales.detach().cpu().numpy() - np.log(max(s, 1e-12))
        _summarize_scale_consistency(scales_model_log_world, scales_ply_log, label="World")
    except Exception as e:
        print(f"[Warning] Failed to check scales (world): {e}")
    if args.diagnose:
        # Same diagnosis in world mode
        def quat_to_rot(q: np.ndarray) -> np.ndarray:
            x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            wx, wy, wz = w * x, w * y, w * z
            R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
            R[..., 0, 0] = 1 - 2 * (yy + zz)
            R[..., 0, 1] = 2 * (xy - wz)
            R[..., 0, 2] = 2 * (xz + wy)
            R[..., 1, 0] = 2 * (xy + wz)
            R[..., 1, 1] = 1 - 2 * (xx + zz)
            R[..., 1, 2] = 2 * (yz - wx)
            R[..., 2, 0] = 2 * (xz - wy)
            R[..., 2, 1] = 2 * (yz + wx)
            R[..., 2, 2] = 1 - 2 * (xx + yy)
            return R
        Rm = quat_to_rot(qm)
        Rp = quat_to_rot(qp)
        M = np.zeros((3, 3), dtype=np.float32)
        for i in range(Rm.shape[0]):
            M += Rp[i] @ Rm[i].T
        U, S, Vt = np.linalg.svd(M)
        Rbest = U @ Vt
        if np.linalg.det(Rbest) < 0:
            U[:, -1] *= -1
            Rbest = U @ Vt
        cos_theta = (np.trace(Rbest) - 1.0) * 0.5
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        theta = float(np.degrees(np.arccos(cos_theta)))
        axis = np.array([
            Rbest[2,1] - Rbest[1,2],
            Rbest[0,2] - Rbest[2,0],
            Rbest[1,0] - Rbest[0,1],
        ], dtype=np.float32)
        n = float(np.linalg.norm(axis))
        axis = (axis / n) if n > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        print("Diagnose: fixed R aligning model->PLY (world): angle={:.6f}°, axis={}".format(theta, axis.tolist()))


if __name__ == "__main__":
    main()


