#!/usr/bin/env python3
"""
Analyze connected components in reconstructed voxel volume.

Finds connected components, computes their size and average color,
and displays results sorted by size.
"""

import numpy as np
import sys
from collections import deque


def label_connected_components_3d(binary_volume, rgb=None, color_tolerance=np.inf):
    """
    Label connected components in a 3D binary volume using flood-fill (BFS).
    Uses 26-connectivity (3x3x3 neighborhood).
    
    Args:
        binary_volume: 3D boolean array
        rgb: Optional [D, H, W, 3] RGB array for color-based connectivity
        color_tolerance: Maximum color distance (L2 norm in RGB space) to consider
                        voxels as connected. Default: inf (ignore color, spatial only)
    
    Returns:
        labeled: 3D int array with component labels (0=background, 1...N=components)
        num_components: Number of components found
    """
    shape = binary_volume.shape
    labeled = np.zeros(shape, dtype=np.int32)
    component_id = 0
    
    # 26-connectivity: all neighbors in 3x3x3 cube
    deltas = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz != 0 or dy != 0 or dx != 0:  # Skip center
                    deltas.append((dz, dy, dx))
    
    # Check if using color-based connectivity
    use_color = rgb is not None and np.isfinite(color_tolerance)
    
    # Flood-fill from each unlabeled occupied voxel
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                # If this voxel is occupied and not yet labeled
                if binary_volume[z, y, x] and labeled[z, y, x] == 0:
                    # Start new component
                    component_id += 1
                    
                    # BFS flood-fill
                    queue = deque([(z, y, x)])
                    labeled[z, y, x] = component_id
                    
                    # Get seed color for this component (if using color connectivity)
                    if use_color:
                        seed_color = rgb[z, y, x].astype(np.float32)
                    
                    while queue:
                        cz, cy, cx = queue.popleft()
                        
                        # Get current voxel color
                        if use_color:
                            current_color = rgb[cz, cy, cx].astype(np.float32)
                        
                        # Check all 26 neighbors
                        for dz, dy, dx in deltas:
                            nz, ny, nx = cz + dz, cy + dy, cx + dx
                            
                            # Check bounds
                            if (0 <= nz < shape[0] and 
                                0 <= ny < shape[1] and 
                                0 <= nx < shape[2]):
                                
                                # If neighbor is occupied and unlabeled
                                if binary_volume[nz, ny, nx] and labeled[nz, ny, nx] == 0:
                                    # Check color similarity if using color connectivity
                                    if use_color:
                                        neighbor_color = rgb[nz, ny, nx].astype(np.float32)
                                        # Compute L2 color distance from seed color
                                        color_dist = np.linalg.norm(neighbor_color - seed_color)
                                        
                                        if color_dist > color_tolerance:
                                            # Color too different, don't connect
                                            continue
                                    
                                    # Add to component
                                    labeled[nz, ny, nx] = component_id
                                    queue.append((nz, ny, nx))
    
    return labeled, component_id


def analyze_connected_components(npz_path, sigma_threshold=0.5, color_tolerance=0.1):
    """
    Analyze connected components in a voxel volume.
    
    Args:
        npz_path: Path to .npz file containing 'sigma' and 'rgb' arrays
        sigma_threshold: Threshold for considering a voxel occupied
        color_tolerance: Maximum color distance (L2 norm in [0,1] RGB space) to 
                        consider voxels as connected. Default: inf (spatial only)
    
    Returns:
        List of (component_id, size, avg_color, bbox) tuples
    """
    # Load volume
    print(f"Loading volume from: {npz_path}")
    data = np.load(npz_path)
    sigma = data['sigma']  # Shape: [D, H, W]
    rgb = data['rgb']      # Shape: [D, H, W, 3]
    
    print(f"Volume shape: {sigma.shape}")
    print(f"Sigma range: [{sigma.min():.3f}, {sigma.max():.3f}]")
    print()
    
    # Create binary occupancy mask
    occupied = sigma > sigma_threshold
    total_occupied = occupied.sum()
    
    print(f"Sigma threshold: {sigma_threshold}")
    print(f"Occupied voxels: {total_occupied:,} ({100 * total_occupied / sigma.size:.2f}% of volume)")
    if np.isfinite(color_tolerance):
        print(f"Color tolerance: {color_tolerance:.3f} (using color-based connectivity)")
    else:
        print(f"Color tolerance: infinite (spatial connectivity only)")
    print()
    
    if total_occupied == 0:
        print("No occupied voxels found! Try lowering the threshold.")
        return []
    
    # Find connected components (26-connectivity using flood-fill)
    print("Finding connected components...")
    labeled, num_components = label_connected_components_3d(occupied, rgb=rgb, color_tolerance=color_tolerance)
    
    print(f"Found {num_components} connected components")
    print()
    
    # Analyze each component
    components = []
    
    for comp_id in range(1, num_components + 1):
        # Get mask for this component
        comp_mask = (labeled == comp_id)
        size = comp_mask.sum()
        
        # Compute average color (weighted by sigma for more accurate color)
        comp_sigma = sigma[comp_mask]
        comp_rgb = rgb[comp_mask]  # [N, 3]
        
        # Weight by sigma (denser voxels contribute more to color)
        weights = comp_sigma / (comp_sigma.sum() + 1e-8)
        avg_color = (comp_rgb.T @ weights).T  # Weighted average
        avg_color_255 = (avg_color * 255).astype(int)
        
        # Compute bounding box
        coords = np.argwhere(comp_mask)
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        bbox = (bbox_min, bbox_max)
        
        components.append({
            'id': comp_id,
            'size': int(size),
            'avg_color': avg_color_255,
            'bbox': bbox,
            'percentage': 100.0 * size / total_occupied
        })
    
    # Sort by size (descending)
    components.sort(key=lambda x: x['size'], reverse=True)
    
    return components, total_occupied


def print_components_table(components, total_occupied):
    """Print formatted table of components."""
    print("=" * 100)
    print("Connected Components Analysis")
    print("=" * 100)
    print()
    
    # Header
    print(f"{'Component':<12} {'Size':<12} {'% Total':<10} {'Avg Color (RGB)':<20} {'Bounding Box'}")
    print(f"{'-'*12} {'-'*12} {'-'*10} {'-'*20} {'-'*30}")
    
    # Components
    for comp in components:
        comp_id = comp['id']
        size = comp['size']
        pct = comp['percentage']
        color = comp['avg_color']
        bbox_min, bbox_max = comp['bbox']
        
        # Format color as (R, G, B)
        color_str = f"({color[0]:3d}, {color[1]:3d}, {color[2]:3d})"
        
        # Format bbox
        bbox_str = f"[{bbox_min[0]}:{bbox_max[0]}, {bbox_min[1]}:{bbox_max[1]}, {bbox_min[2]}:{bbox_max[2]}]"
        
        print(f"{comp_id:<12} {size:<12,} {pct:<10.2f} {color_str:<20} {bbox_str}")
    
    print()
    print(f"Total occupied voxels: {total_occupied:,}")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_connected_components.py <npz_file> [sigma_threshold] [color_tolerance]")
        print()
        print("Arguments:")
        print("  npz_file: Path to .npz file with 'sigma' and 'rgb' arrays")
        print("  sigma_threshold: Threshold for occupied voxels (default: 0.5)")
        print("  color_tolerance: Max color distance for connectivity (default: 0.1)")
        print("                   Color distance is L2 norm in [0,1] RGB space")
        print("                   Use 'inf' for spatial-only connectivity")
        print()
        print("Examples:")
        print("  # Default (spatial + color with tolerance 0.1)")
        print("  python analyze_connected_components.py video_voxel_out/recon_volume.npz")
        print()
        print("  # Spatial connectivity only")
        print("  python analyze_connected_components.py video_voxel_out/recon_volume.npz 0.5 inf")
        print()
        print("  # Strict color matching (tolerance 0.05)")
        print("  python analyze_connected_components.py video_voxel_out/recon_volume.npz 0.3 0.05")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    sigma_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    color_tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    
    try:
        components, total_occupied = analyze_connected_components(npz_path, sigma_threshold, color_tolerance)
        
        if components:
            print_components_table(components, total_occupied)
            
            # Summary statistics
            if len(components) > 1:
                largest = components[0]
                print("Summary:")
                print(f"  Largest component: #{largest['id']} with {largest['size']:,} voxels ({largest['percentage']:.1f}%)")
                print(f"  Number of small components (<1%): {sum(1 for c in components if c['percentage'] < 1.0)}")
                
                # Check for significant spurious components
                spurious = [c for c in components[1:] if c['percentage'] > 5.0]
                if spurious:
                    print(f"  WARNING: {len(spurious)} significant spurious component(s) detected (>5% of volume)")
                    for c in spurious:
                        print(f"    - Component #{c['id']}: {c['size']:,} voxels ({c['percentage']:.1f}%)")
    
    except FileNotFoundError:
        print(f"Error: File not found: {npz_path}")
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

