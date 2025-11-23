#!/usr/bin/env python3
"""
Post-process a soft voxel volume (sigma, rgb) into a hardened, more solid field.

Usage:

  python harden_recon_volume.py \
      --npz video_voxel_out/recon_volume.npz \
      --out-dir video_voxel_out \
      --percentile 90 \
      --solid-sigma 5.0 \
      --morph-iters 2

This will create:
  - video_voxel_out/recon_volume_hardened.npz
  - video_voxel_out/recon_voxels_hardened.ply
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------- core hardening logic ----------

def harden_sigma(
    sigma: np.ndarray,
    percentile: float = 90.0,
    solid_sigma: float = 5.0,
    morph_iters: int = 2,
) -> np.ndarray:
    """
    Turn a soft sigma field into a crisp, almost-binary one.

    sigma: [D,H,W] numpy array (float32)
    returns: sigma_hard: [D,H,W] numpy array (float32)
    """
    assert sigma.ndim == 3, "sigma should be [D,H,W]"

    nonzero = sigma[sigma > 0]
    if nonzero.size == 0:
        print("[WARN] sigma is all zeros, nothing to harden.")
        return sigma.copy()

    # Percentile over non-zero sigmas → threshold for "solid" voxels
    tau = float(np.percentile(nonzero, percentile))
    print(f"[HARDEN] percentile={percentile} → tau={tau:.6f}")

    mask = sigma >= tau  # boolean occupancy

    # Optional 3D morphological cleanup (dilate then erode)
    if morph_iters > 0:
        print(f"[HARDEN] running {morph_iters} dilate/erode iterations")
        m = torch.from_numpy(mask.astype(np.float32))[None, None]  # [1,1,D,H,W]
        k = torch.ones(1, 1, 3, 3, 3, dtype=torch.float32)

        # dilation: any neighbor → 1
        for _ in range(morph_iters):
            m = (F.conv3d(m, k, padding=1) > 0).float()

        # erosion: all neighbors → 1  (27 voxels in 3x3x3)
        for _ in range(morph_iters):
            m = (F.conv3d(m, k, padding=1) == 27).float()

        mask = m[0, 0].cpu().numpy().astype(bool)

    num_on = int(mask.sum())
    print(f"[HARDEN] active voxels after mask = {num_on}")

    sigma_hard = np.zeros_like(sigma, dtype=np.float32)
    sigma_hard[mask] = solid_sigma  # high, almost-opaque density

    return sigma_hard


# ---------- PLY export (RGBA, but opaque) ----------

def export_voxels_as_ply_rgba(
    sigma: np.ndarray,
    rgb: np.ndarray,
    out_path: Path,
    thresh: float,
    opaque: bool = True,
):
    """
    sigma: [D,H,W]       density
    rgb:   [D,H,W,3]     in [0,1]
    Writes a vertex PLY with RGBA.
    If opaque=True, alpha=255 for all voxels above thresh.
    """
    D, H, W = sigma.shape
    assert rgb.shape == (D, H, W, 3), "rgb must be [D,H,W,3]"

    mask = sigma > thresh
    idxs = np.argwhere(mask)

    if len(idxs) == 0:
        print("[PLY] No voxels above threshold, skipping export.")
        return

    zs = np.linspace(-1, 1, D)
    ys = np.linspace(-1, 1, H)
    xs = np.linspace(-1, 1, W)

    sigma_norm = sigma / (sigma.max() + 1e-8)

    verts = []
    colors = []

    for z_i, y_i, x_i in idxs:
        x = xs[x_i]
        y = ys[y_i]
        z = zs[z_i]
        verts.append((x, y, z))

        c_rgb = (rgb[z_i, y_i, x_i] * 255.0).astype(np.uint8)

        if opaque:
            a = 255
        else:
            a = int(np.clip(sigma_norm[z_i, y_i, x_i] * 255.0, 0, 255))

        colors.append((int(c_rgb[0]), int(c_rgb[1]), int(c_rgb[2]), a))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        n = len(verts)
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b, a) in zip(verts, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {a}\n")

    print(f"[PLY] wrote {n} vertices to {out_path}")


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Harden a recon_volume.npz into a crisper, more solid volume."
    )
    parser.add_argument(
        "--npz", type=str, required=True,
        help="Path to recon_volume.npz (with sigma and rgb arrays).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: same as npz).",
    )
    parser.add_argument(
        "--percentile", type=float, default=90.0,
        help="Percentile over non-zero sigma used as threshold (default: 90).",
    )
    parser.add_argument(
        "--solid-sigma", type=float, default=5.0,
        help="Sigma value to assign to solid voxels (default: 5.0).",
    )
    parser.add_argument(
        "--morph-iters", type=int, default=2,
        help="Number of 3D dilate/erode iterations (default: 2).",
    )
    parser.add_argument(
        "--opaque-ply", action="store_true",
        help="If set, PLY alpha=255 for all voxels above threshold.",
    )

    args = parser.parse_args()

    npz_path = Path(args.npz)
    assert npz_path.exists(), f"{npz_path} does not exist"

    out_dir = Path(args.out_dir) if args.out_dir is not None else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {npz_path}")
    data = np.load(npz_path)
    sigma = data["sigma"]  # [D,H,W]
    rgb   = data["rgb"]    # [D,H,W,3]

    print(f"[LOAD] sigma shape={sigma.shape}, rgb shape={rgb.shape}")
    sigma_hard = harden_sigma(
        sigma,
        percentile=args.percentile,
        solid_sigma=args.solid_sigma,
        morph_iters=args.morph_iters,
    )

    # Save hardened volume
    hardened_npz = out_dir / (npz_path.stem + "_hardened.npz")
    np.savez_compressed(hardened_npz, sigma=sigma_hard, rgb=rgb)
    print(f"[SAVE] hardened volume → {hardened_npz}")

    # Export PLY (threshold at half of solid_sigma since it's basically binary)
    ply_path = out_dir / "recon_voxels_hardened.ply"
    thresh = args.solid_sigma * 0.5
    export_voxels_as_ply_rgba(
        sigma_hard,
        rgb,
        ply_path,
        thresh=thresh,
        opaque=args.opaque_ply,
    )


if __name__ == "__main__":
    main()

