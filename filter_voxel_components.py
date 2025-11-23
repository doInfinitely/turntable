#!/usr/bin/env python3
"""
Filter out small connected components from a voxel volume.
Keeps only the largest component(s) and zeros out noise/artifacts.
"""

import numpy as np
import sys
from analyze_connected_components import label_connected_components_3d


def filter_components(npz_path, output_path, sigma_threshold=0.5, min_size_percent=5.0):
    """
    Filter voxel volume to keep only large connected components.
    
    Args:
        npz_path: Input NPZ file path
        output_path: Output NPZ file path
        sigma_threshold: Threshold for considering a voxel occupied
        min_size_percent: Minimum component size as % of total occupied voxels
    
    Returns:
        Number of components kept, number filtered out
    """
    print(f"Loading volume from: {npz_path}")
    data = np.load(npz_path)
    sigma = data['sigma'].copy()  # Shape: [D, H, W]
    rgb = data['rgb'].copy()      # Shape: [D, H, W, 3]
    
    print(f"Volume shape: {sigma.shape}")
    print(f"Sigma threshold: {sigma_threshold}")
    print(f"Minimum component size: {min_size_percent}% of occupied voxels")
    print()
    
    # Create binary occupancy mask
    occupied = sigma > sigma_threshold
    total_occupied = occupied.sum()
    print(f"Total occupied voxels: {total_occupied:,}")
    
    if total_occupied == 0:
        print("No occupied voxels found!")
        return 0, 0
    
    # Find connected components
    print("Finding connected components...")
    labeled, num_components = label_connected_components_3d(occupied)
    print(f"Found {num_components} connected components")
    print()
    
    # Compute component sizes
    min_size = int((min_size_percent / 100.0) * total_occupied)
    print(f"Minimum voxels to keep: {min_size:,}")
    print()
    
    components_to_keep = []
    components_to_remove = []
    
    for comp_id in range(1, num_components + 1):
        comp_mask = (labeled == comp_id)
        size = comp_mask.sum()
        
        if size >= min_size:
            components_to_keep.append((comp_id, size))
        else:
            components_to_remove.append((comp_id, size))
    
    print(f"Components to keep: {len(components_to_keep)}")
    for comp_id, size in components_to_keep:
        pct = 100.0 * size / total_occupied
        print(f"  Component #{comp_id}: {size:,} voxels ({pct:.2f}%)")
    
    print()
    print(f"Components to remove: {len(components_to_remove)}")
    if len(components_to_remove) <= 10:
        for comp_id, size in components_to_remove:
            pct = 100.0 * size / total_occupied
            print(f"  Component #{comp_id}: {size:,} voxels ({pct:.2f}%)")
    else:
        total_removed = sum(size for _, size in components_to_remove)
        pct = 100.0 * total_removed / total_occupied
        print(f"  {len(components_to_remove)} small components totaling {total_removed:,} voxels ({pct:.2f}%)")
    print()
    
    # Create mask of voxels to keep
    keep_mask = np.zeros_like(labeled, dtype=bool)
    for comp_id, _ in components_to_keep:
        keep_mask |= (labeled == comp_id)
    
    # Filter sigma and rgb
    print("Filtering volume...")
    sigma_filtered = sigma.copy()
    rgb_filtered = rgb.copy()
    
    # Zero out small components
    sigma_filtered[~keep_mask] = 0.0
    rgb_filtered[~keep_mask] = 0.0
    
    voxels_removed = total_occupied - keep_mask.sum()
    print(f"Zeroed out {voxels_removed:,} voxels ({100.0 * voxels_removed / total_occupied:.2f}% of occupied volume)")
    print()
    
    # Save filtered volume
    print(f"Saving filtered volume to: {output_path}")
    np.savez_compressed(
        output_path,
        sigma=sigma_filtered,
        rgb=rgb_filtered
    )
    
    print("Done!")
    print()
    print(f"View the filtered volume:")
    print(f"  python voxel_volume_viewer.py {output_path}")
    
    return len(components_to_keep), len(components_to_remove)


def main():
    if len(sys.argv) < 2:
        print("Usage: python filter_voxel_components.py <input.npz> [output.npz] [min_size_percent]")
        print()
        print("Arguments:")
        print("  input.npz         Input NPZ file to filter")
        print("  output.npz        Output NPZ file (default: input_filtered.npz)")
        print("  min_size_percent  Minimum component size as % of occupied voxels (default: 5.0)")
        print()
        print("Examples:")
        print("  # Keep components >= 5% of occupied volume (default)")
        print("  python filter_voxel_components.py video_voxel_out/recon_volume.npz")
        print()
        print("  # Keep only components >= 50% (basically just the largest)")
        print("  python filter_voxel_components.py video_voxel_out/recon_volume.npz output.npz 50.0")
        print()
        print("  # Keep components >= 1% (more permissive)")
        print("  python filter_voxel_components.py video_voxel_out/recon_volume.npz output.npz 1.0")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Default output path: add _filtered before extension
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        if input_path.endswith('.npz'):
            output_path = input_path[:-4] + '_filtered.npz'
        else:
            output_path = input_path + '_filtered.npz'
    
    min_size_percent = float(sys.argv[3]) if len(sys.argv) >= 4 else 5.0
    
    try:
        filter_components(input_path, output_path, sigma_threshold=0.5, min_size_percent=min_size_percent)
    except FileNotFoundError:
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Expected key {e} not found in NPZ file")
        print("Make sure the file contains 'sigma' and 'rgb' arrays")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

