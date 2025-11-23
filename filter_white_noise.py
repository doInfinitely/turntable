#!/usr/bin/env python3
"""
Filter out small white/bright components from a voxel volume.
Removes only components that are both small AND nearly white (artifacts).
"""

import numpy as np
import sys
from analyze_connected_components import label_connected_components_3d


def filter_white_noise(npz_path, output_path, sigma_threshold=0.5, 
                       max_size_voxels=200, white_threshold=240):
    """
    Filter voxel volume to remove small white/bright components.
    
    Args:
        npz_path: Input NPZ file path
        output_path: Output NPZ file path
        sigma_threshold: Threshold for considering a voxel occupied
        max_size_voxels: Maximum size (in voxels) to consider for removal
        white_threshold: RGB threshold for considering a component "white" (0-255)
    
    Returns:
        Number of components kept, number filtered out
    """
    print(f"Loading volume from: {npz_path}")
    data = np.load(npz_path)
    sigma = data['sigma'].copy()  # Shape: [D, H, W]
    rgb = data['rgb'].copy()      # Shape: [D, H, W, 3]
    
    print(f"Volume shape: {sigma.shape}")
    print(f"Sigma threshold: {sigma_threshold}")
    print(f"Filtering criteria:")
    print(f"  - Size <= {max_size_voxels} voxels")
    print(f"  - AND average RGB >= {white_threshold}/255 (nearly white)")
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
    
    # Analyze each component
    components_to_keep = []
    components_to_remove = []
    
    for comp_id in range(1, num_components + 1):
        comp_mask = (labeled == comp_id)
        size = comp_mask.sum()
        
        # Compute average color (weighted by sigma)
        comp_sigma = sigma[comp_mask]
        comp_rgb = rgb[comp_mask]  # [N, 3]
        weights = comp_sigma / (comp_sigma.sum() + 1e-8)
        avg_color = (comp_rgb.T @ weights).T  # Weighted average
        avg_color_255 = (avg_color * 255)
        
        # Check if this is a small white component
        is_small = size <= max_size_voxels
        is_white = np.all(avg_color_255 >= white_threshold)
        
        if is_small and is_white:
            components_to_remove.append((comp_id, size, avg_color_255))
        else:
            components_to_keep.append((comp_id, size, avg_color_255))
    
    print(f"Components to KEEP: {len(components_to_keep)}")
    # Show first 10
    for i, (comp_id, size, color) in enumerate(components_to_keep[:10]):
        pct = 100.0 * size / total_occupied
        print(f"  #{comp_id}: {size:,} voxels ({pct:.2f}%), RGB=({int(color[0])}, {int(color[1])}, {int(color[2])})")
    if len(components_to_keep) > 10:
        print(f"  ... and {len(components_to_keep) - 10} more")
    
    print()
    print(f"Components to REMOVE (small white noise): {len(components_to_remove)}")
    # Show first 20
    for i, (comp_id, size, color) in enumerate(components_to_remove[:20]):
        pct = 100.0 * size / total_occupied
        print(f"  #{comp_id}: {size:,} voxels ({pct:.2f}%), RGB=({int(color[0])}, {int(color[1])}, {int(color[2])})")
    if len(components_to_remove) > 20:
        total_removed = sum(size for _, size, _ in components_to_remove)
        pct = 100.0 * total_removed / total_occupied
        print(f"  ... and {len(components_to_remove) - 20} more")
        print(f"  Total to remove: {total_removed:,} voxels ({pct:.2f}%)")
    else:
        total_removed = sum(size for _, size, _ in components_to_remove)
        pct = 100.0 * total_removed / total_occupied
        print(f"  Total: {total_removed:,} voxels ({pct:.2f}%)")
    print()
    
    # Create mask of voxels to keep
    keep_mask = np.zeros_like(labeled, dtype=bool)
    for comp_id, _, _ in components_to_keep:
        keep_mask |= (labeled == comp_id)
    
    # Filter sigma and rgb
    print("Filtering volume...")
    sigma_filtered = sigma.copy()
    rgb_filtered = rgb.copy()
    
    # Zero out white noise components
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
        print("Usage: python filter_white_noise.py <input.npz> [output.npz] [max_size] [white_threshold]")
        print()
        print("Arguments:")
        print("  input.npz        Input NPZ file to filter")
        print("  output.npz       Output NPZ file (default: input_filtered.npz)")
        print("  max_size         Max size in voxels to consider for removal (default: 200)")
        print("  white_threshold  RGB threshold for 'white' 0-255 (default: 240)")
        print()
        print("Examples:")
        print("  # Remove small white components (default)")
        print("  python filter_white_noise.py video_voxel_out/recon_volume.npz")
        print()
        print("  # Remove larger white components (up to 500 voxels)")
        print("  python filter_white_noise.py video_voxel_out/recon_volume.npz output.npz 500")
        print()
        print("  # More aggressive white detection (>= 220)")
        print("  python filter_white_noise.py video_voxel_out/recon_volume.npz output.npz 200 220")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Default output path
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        if input_path.endswith('.npz'):
            output_path = input_path[:-4] + '_filtered.npz'
        else:
            output_path = input_path + '_filtered.npz'
    
    max_size_voxels = int(sys.argv[3]) if len(sys.argv) >= 4 else 200
    white_threshold = int(sys.argv[4]) if len(sys.argv) >= 5 else 240
    
    try:
        filter_white_noise(input_path, output_path, 
                          sigma_threshold=0.5,
                          max_size_voxels=max_size_voxels,
                          white_threshold=white_threshold)
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

