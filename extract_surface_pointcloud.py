#!/usr/bin/env python3
"""
Extract surface point cloud from voxel volume using volumetric ray-casting.

Casts rays from a sphere through the volume and finds where cumulative
transmittance drops below a threshold (surface hit point).
"""

import numpy as np
import sys
import torch
import torch.nn.functional as F


def fibonacci_sphere(n_points):
    """
    Generate uniformly distributed points on a unit sphere using Fibonacci spiral.
    
    Returns: (n_points, 3) array of unit vectors
    """
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y from 1 to -1
        radius = np.sqrt(1 - y * y)
        
        theta = phi * i
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        points.append([x, y, z])
    
    return np.array(points)


def world_to_grid_coords(pts_world, scene_radius=1.5):
    """
    Convert world coordinates to grid sample coordinates.
    
    Args:
        pts_world: [..., 3] points in world space [-scene_radius, scene_radius]
        scene_radius: Radius of the scene
    
    Returns:
        grid_coords: [..., 3] in [-1, 1] for F.grid_sample
    """
    # Normalize to [-1, 1] for grid_sample
    grid_coords = pts_world / scene_radius
    # Clamp to valid range
    grid_coords = np.clip(grid_coords, -1.0, 1.0)
    return grid_coords


def sample_volume_at_points(sigma_tensor, rgb_tensor, pts_world, scene_radius=1.5):
    """
    Sample sigma and RGB at world-space points using trilinear interpolation.
    
    Args:
        sigma_tensor: [1, 1, D, H, W] tensor
        rgb_tensor: [1, 3, D, H, W] tensor
        pts_world: [N, 3] numpy array in world space
        scene_radius: Scene radius
    
    Returns:
        sigma_samples: [N] numpy array
        rgb_samples: [N, 3] numpy array
    """
    # Convert to grid coordinates
    grid_coords = world_to_grid_coords(pts_world, scene_radius)
    
    # Reshape for grid_sample: [1, N, 1, 1, 3]
    grid_coords_torch = torch.from_numpy(grid_coords).float().reshape(1, -1, 1, 1, 3)
    
    # Sample
    with torch.no_grad():
        sigma_sampled = F.grid_sample(
            sigma_tensor, grid_coords_torch,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )  # [1, 1, N, 1, 1]
        
        rgb_sampled = F.grid_sample(
            rgb_tensor, grid_coords_torch,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )  # [1, 3, N, 1, 1]
    
    # Extract and convert to numpy
    sigma_out = sigma_sampled.squeeze().cpu().numpy()  # [N]
    rgb_out = rgb_sampled.squeeze(0).squeeze(-1).squeeze(-1).permute(1, 0).cpu().numpy()  # [N, 3]
    
    return sigma_out, rgb_out


def cast_ray_find_surface(sigma_tensor, rgb_tensor, ray_origin, ray_direction,
                          n_samples=128, near=0.1, far=4.0, scene_radius=1.5,
                          transmittance_threshold=0.5):
    """
    Cast a single ray and find the surface hit point.
    
    Args:
        sigma_tensor: [1, 1, D, H, W] tensor
        rgb_tensor: [1, 3, D, H, W] tensor  
        ray_origin: [3] numpy array
        ray_direction: [3] unit vector
        n_samples: Number of samples along ray
        near, far: Near and far planes
        scene_radius: Scene radius
        transmittance_threshold: T threshold for surface (e.g., 0.5 = 50% light blocked)
    
    Returns:
        surface_point: [3] world position or None if no hit
        surface_color: [3] RGB color [0,1] or None
    """
    # Generate sample points along ray
    t_vals = np.linspace(near, far, n_samples)
    pts = ray_origin[None, :] + t_vals[:, None] * ray_direction[None, :]  # [n_samples, 3]
    
    # Sample volume
    sigma_samples, rgb_samples = sample_volume_at_points(
        sigma_tensor, rgb_tensor, pts, scene_radius
    )  # [n_samples], [n_samples, 3]
    
    # Compute transmittance along ray
    delta = (far - near) / n_samples
    
    # T(t) = exp(-∑σ_i * δ)
    transmittance = np.ones(n_samples)
    for i in range(1, n_samples):
        # Cumulative transmittance up to this point
        transmittance[i] = transmittance[i-1] * np.exp(-sigma_samples[i-1] * delta)
    
    # Find first point where transmittance drops below threshold
    hit_indices = np.where(transmittance < transmittance_threshold)[0]
    
    if len(hit_indices) > 0:
        hit_idx = hit_indices[0]
        # Interpolate between samples for sub-voxel precision
        if hit_idx > 0:
            # Linear interpolation between hit_idx-1 and hit_idx
            t0, t1 = transmittance[hit_idx-1], transmittance[hit_idx]
            alpha = (transmittance_threshold - t1) / (t0 - t1 + 1e-8)
            alpha = np.clip(alpha, 0, 1)
            
            surface_point = pts[hit_idx-1] * alpha + pts[hit_idx] * (1 - alpha)
            surface_color = rgb_samples[hit_idx-1] * alpha + rgb_samples[hit_idx] * (1 - alpha)
        else:
            surface_point = pts[hit_idx]
            surface_color = rgb_samples[hit_idx]
        
        return surface_point, surface_color
    
    return None, None


def extract_surface_pointcloud(npz_path, n_rays=10000, n_samples=128,
                               transmittance_threshold=0.5, sphere_radius=2.5,
                               scene_radius=1.5):
    """
    Extract surface point cloud by casting rays from a sphere.
    
    Args:
        npz_path: Path to volume NPZ
        n_rays: Number of rays to cast (more = denser point cloud)
        n_samples: Samples per ray
        transmittance_threshold: T threshold for surface detection (0.5 = 50% blocked)
        sphere_radius: Radius of ray origin sphere
        scene_radius: Voxel grid scene radius
    
    Returns:
        points: [N, 3] surface points
        colors: [N, 3] surface colors
    """
    print(f"Loading volume from: {npz_path}")
    data = np.load(npz_path)
    sigma = data['sigma']  # [D, H, W]
    rgb = data['rgb']      # [D, H, W, 3]
    
    print(f"Volume shape: {sigma.shape}")
    print(f"Sigma range: [{sigma.min():.3f}, {sigma.max():.3f}]")
    print()
    
    # Convert to torch tensors
    sigma_tensor = torch.from_numpy(sigma).float().unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    rgb_tensor = torch.from_numpy(rgb).float().permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, D, H, W]
    
    print(f"Generating {n_rays} rays from sphere (radius={sphere_radius})...")
    
    # Generate uniformly distributed ray origins on sphere
    ray_directions = fibonacci_sphere(n_rays)
    ray_origins = ray_directions * sphere_radius  # Place on sphere surface
    
    # For rays across diameter: origin on one side, direction toward opposite side
    # Ray points from +dir to -dir (across diameter)
    ray_directions = -ray_directions  # Point inward toward origin
    
    print(f"Casting rays (transmittance threshold={transmittance_threshold})...")
    
    surface_points = []
    surface_colors = []
    
    for i in range(n_rays):
        if i % 500 == 0:
            print(f"  Ray {i}/{n_rays} ({100*i/n_rays:.1f}%) - {len(surface_points)} hits so far")
        
        point, color = cast_ray_find_surface(
            sigma_tensor, rgb_tensor,
            ray_origins[i], ray_directions[i],
            n_samples=n_samples,
            near=0.1,
            far=sphere_radius * 2,  # Diameter
            scene_radius=scene_radius,
            transmittance_threshold=transmittance_threshold
        )
        
        if point is not None:
            surface_points.append(point)
            surface_colors.append(color)
    
    if len(surface_points) == 0:
        print("No surface hits found!")
        return None, None
    
    surface_points = np.array(surface_points)
    surface_colors = np.array(surface_colors)
    
    print()
    print(f"Extracted {len(surface_points)} surface points ({100*len(surface_points)/n_rays:.1f}% hit rate)")
    
    return surface_points, surface_colors


def save_pointcloud_ply(points, colors, output_path):
    """Save point cloud as PLY file."""
    n = len(points)
    
    # Convert colors to 0-255
    colors_255 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    
    print(f"Saving point cloud to: {output_path}")
    
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Data
        for i in range(n):
            x, y, z = points[i]
            r, g, b = colors_255[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
    
    print(f"Saved {n} points")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_surface_pointcloud.py <input.npz> [output.ply] [n_rays] [threshold]")
        print()
        print("Arguments:")
        print("  input.npz   Input volume NPZ file")
        print("  output.ply  Output PLY file (default: input_surface.ply)")
        print("  n_rays      Number of rays to cast (default: 10000)")
        print("  threshold   Transmittance threshold for surface (default: 0.5)")
        print()
        print("Examples:")
        print("  # Default settings")
        print("  python extract_surface_pointcloud.py video_voxel_out/recon_volume.npz")
        print()
        print("  # More rays for denser point cloud")
        print("  python extract_surface_pointcloud.py video_voxel_out/recon_volume.npz surface.ply 50000")
        print()
        print("  # More aggressive surface detection (less light blocked)")
        print("  python extract_surface_pointcloud.py video_voxel_out/recon_volume.npz surface.ply 10000 0.3")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Default output path
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        if input_path.endswith('.npz'):
            output_path = input_path[:-4] + '_surface.ply'
        else:
            output_path = input_path + '_surface.ply'
    
    n_rays = int(sys.argv[3]) if len(sys.argv) >= 4 else 10000
    threshold = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.5
    
    try:
        print("=" * 60)
        print("Surface Point Cloud Extraction")
        print("=" * 60)
        print()
        
        points, colors = extract_surface_pointcloud(
            input_path,
            n_rays=n_rays,
            n_samples=128,
            transmittance_threshold=threshold,
            sphere_radius=2.5,
            scene_radius=1.5
        )
        
        if points is not None:
            print()
            save_pointcloud_ply(points, colors, output_path)
            print()
            print("Done! View with:")
            print(f"  meshlab {output_path}")
            print(f"  # or drag into CloudCompare, Blender, etc.")
    
    except FileNotFoundError:
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

