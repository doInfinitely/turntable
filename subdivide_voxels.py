#!/usr/bin/env python3
"""
Subdivide voxel grid by splitting each voxel into multiple smaller voxels.

Takes an input NPZ file with sigma and rgb arrays, subdivides the grid,
and outputs a new NPZ file where each original voxel becomes N³ voxels
(where N is the subdivision factor).

All subdivided voxels inherit the RGBA values from their parent voxel.

Usage:
    # Subdivide 32³ → 64³ (subdivision factor = 2)
    python subdivide_voxels.py --npz input.npz --subdivision 2 --out output.npz
    
    # Subdivide 64³ → 192³ (subdivision factor = 3)
    python subdivide_voxels.py --npz input.npz --subdivision 3 --out output.npz
"""

import argparse
from pathlib import Path

import numpy as np


def subdivide_voxels(sigma, rgb, subdivision):
    """
    Subdivide voxel grid by repeating each voxel.
    
    Args:
        sigma: [D, H, W] density array
        rgb: [D, H, W, 3] color array
        subdivision: int, subdivision factor (e.g., 2 means each voxel becomes 2³=8 voxels)
    
    Returns:
        sigma_sub: [D*subdivision, H*subdivision, W*subdivision] subdivided density
        rgb_sub: [D*subdivision, H*subdivision, W*subdivision, 3] subdivided color
    """
    D, H, W = sigma.shape
    
    # Method: repeat each voxel along all 3 dimensions
    # np.repeat works on a single axis at a time
    
    # Subdivide sigma
    # First repeat along depth (axis 0)
    sigma_sub = np.repeat(sigma, subdivision, axis=0)  # [D*sub, H, W]
    # Then repeat along height (axis 1)
    sigma_sub = np.repeat(sigma_sub, subdivision, axis=1)  # [D*sub, H*sub, W]
    # Finally repeat along width (axis 2)
    sigma_sub = np.repeat(sigma_sub, subdivision, axis=2)  # [D*sub, H*sub, W*sub]
    
    # Subdivide rgb (same process but with 4D array)
    rgb_sub = np.repeat(rgb, subdivision, axis=0)    # [D*sub, H, W, 3]
    rgb_sub = np.repeat(rgb_sub, subdivision, axis=1)  # [D*sub, H*sub, W, 3]
    rgb_sub = np.repeat(rgb_sub, subdivision, axis=2)  # [D*sub, H*sub, W*sub, 3]
    
    return sigma_sub, rgb_sub


def main():
    parser = argparse.ArgumentParser(
        description="Subdivide voxel grid by splitting each voxel into N³ smaller voxels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Double resolution: 32³ → 64³
  python subdivide_voxels.py --npz input.npz --subdivision 2
  
  # Triple resolution: 32³ → 96³
  python subdivide_voxels.py --npz input.npz --subdivision 3
  
  # Custom output path
  python subdivide_voxels.py --npz input.npz --subdivision 2 --out output_64.npz
  
  # Very high resolution: 64³ → 256³
  python subdivide_voxels.py --npz input.npz --subdivision 4
        """
    )
    
    parser.add_argument(
        "--npz", type=str, required=True,
        help="Path to input .npz file (with sigma and rgb arrays)"
    )
    parser.add_argument(
        "--subdivision", type=int, required=True,
        help="Subdivision factor (e.g., 2 means each voxel becomes 2³=8 voxels)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path for subdivided .npz file (default: input_subN.npz)"
    )
    
    args = parser.parse_args()
    
    # Validate subdivision factor
    if args.subdivision < 1:
        raise ValueError(f"Subdivision factor must be >= 1, got {args.subdivision}")
    
    if args.subdivision == 1:
        print("[WARN] Subdivision factor is 1, output will be identical to input")
    
    # Load input
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"Input file not found: {npz_path}")
    
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    
    if "sigma" not in data or "rgb" not in data:
        raise ValueError("Input .npz must contain 'sigma' and 'rgb' arrays")
    
    sigma = data["sigma"]  # [D, H, W]
    rgb = data["rgb"]      # [D, H, W, 3]
    
    D, H, W = sigma.shape
    print(f"Input shape: {D}×{H}×{W}")
    print(f"  Sigma: {sigma.shape}, dtype={sigma.dtype}")
    print(f"  RGB: {rgb.shape}, dtype={rgb.dtype}")
    
    # Verify shapes
    if rgb.shape[:3] != (D, H, W):
        raise ValueError(f"Shape mismatch: sigma {sigma.shape}, rgb {rgb.shape}")
    if rgb.shape[3] != 3:
        raise ValueError(f"RGB must have 3 channels, got shape {rgb.shape}")
    
    # Subdivide
    print(f"\nSubdividing with factor {args.subdivision}...")
    print(f"  Each voxel will become {args.subdivision}³ = {args.subdivision**3} voxels")
    
    sigma_sub, rgb_sub = subdivide_voxels(sigma, rgb, args.subdivision)
    
    D_new, H_new, W_new = sigma_sub.shape
    print(f"\nOutput shape: {D_new}×{H_new}×{W_new}")
    print(f"  Sigma: {sigma_sub.shape}")
    print(f"  RGB: {rgb_sub.shape}")
    print(f"  Resolution increased by {args.subdivision}× in each dimension")
    print(f"  Total voxels: {D*H*W:,} → {D_new*H_new*W_new:,} ({args.subdivision**3}× more)")
    
    # Determine output path
    if args.out is None:
        out_path = npz_path.parent / f"{npz_path.stem}_sub{args.subdivision}.npz"
    else:
        out_path = Path(args.out)
    
    # Save
    print(f"\nSaving to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, sigma=sigma_sub, rgb=rgb_sub)
    
    # Report file sizes
    input_size_mb = npz_path.stat().st_size / 1e6
    output_size_mb = out_path.stat().st_size / 1e6
    print(f"\nFile sizes:")
    print(f"  Input:  {input_size_mb:.2f} MB")
    print(f"  Output: {output_size_mb:.2f} MB ({output_size_mb/input_size_mb:.1f}× larger)")
    
    print(f"\nDone! Subdivided voxel grid saved to {out_path}")


if __name__ == "__main__":
    main()

