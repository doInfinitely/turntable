#!/usr/bin/env python3
"""
Differentiable hardening via density flow optimization.

Instead of optimizing voxel densities directly, this script:
1. Computes derivatives of error w.r.t. density FLOW between neighboring voxels
2. Rewards high variance (penalizes uniformity) by subtracting variance from loss
3. Updates voxels via gradient descent on flow parameters

The goal is to move density toward the outer shell of the model, creating
a crisp surface representation with sharp boundaries (high variance = 
concentrated vs empty voxels, not diffuse medium-density everywhere).

Usage:
    python differentiable_hardening.py \
        --npz video_voxel_out/recon_volume.npz \
        --out-dir video_voxel_out \
        --n-iters 1000 \
        --flow-lr 0.01 \
        --variance-penalty 0.1
"""

import argparse
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import cv2


# ==================== Camera & Rendering Utilities ====================

def make_intrinsics(h, w, fov_y_deg=45.0, device="cpu"):
    """Create camera intrinsics matrix."""
    fov_y = math.radians(fov_y_deg)
    fy = 0.5 * h / math.tan(0.5 * fov_y)
    fx = fy
    cx = w / 2.0
    cy = h / 2.0
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=torch.float32, device=device)
    return K


def build_orbit_pose(theta, radius=2.5, device="cpu"):
    """Build camera pose for circular orbit at angle theta."""
    C = np.array([radius * math.cos(theta),
                  0.0,
                  radius * math.sin(theta)], dtype=np.float32)
    
    look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    forward = (look_at - C)
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    
    true_up = np.cross(right, forward)
    
    R = np.stack([right, true_up, forward], axis=0)
    t = -R @ C[:, None]
    
    R_t = torch.from_numpy(R).to(device)
    t_t = torch.from_numpy(t).to(device)
    return R_t, t_t


def world_to_grid(pts_world, scene_radius=1.5):
    """Map world coordinates to grid_sample coordinates [-1, 1]."""
    return pts_world / scene_radius


def generate_rays(h, w, K, R, t, n_samples=64, near=0.1, far=5.0, device="cpu"):
    """Generate 3D sample points along rays."""
    ys, xs = torch.meshgrid(
        torch.linspace(0, h - 1, h, device=device),
        torch.linspace(0, w - 1, w, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=-1)
    
    K_inv = torch.inverse(K.to(device))
    dirs_cam = (K_inv @ pix.reshape(-1, 3).T).T
    dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)
    
    R = R.to(device)
    t = t.to(device)
    
    dirs_world = (R.transpose(0, 1) @ dirs_cam.T).T
    C = -(R.transpose(0, 1) @ t).reshape(1, 3)
    
    ts = torch.linspace(near, far, n_samples, device=device).view(-1, 1, 1)
    dirs_world = dirs_world.reshape(1, h, w, 3)
    C_exp = C.view(1, 1, 1, 3)
    
    pts = C_exp + ts[..., None] * dirs_world
    pts = pts.unsqueeze(0)
    return pts


def sample_volume(sigma, rgb, pts_world, scene_radius=1.5):
    """Sample volume at world points."""
    pts_grid = world_to_grid(pts_world, scene_radius)
    _, S, H, W, _ = pts_grid.shape
    grid = pts_grid.view(1, S, H, W, 3)
    
    sigma_samples = F.grid_sample(
        sigma, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    rgb_samples = F.grid_sample(
        rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    
    sigma_samples = sigma_samples.squeeze(1)
    rgb_samples = rgb_samples.permute(0, 2, 3, 4, 1)
    return sigma_samples, rgb_samples


def volume_render(sigma_samples, rgb_samples, n_samples):
    """NeRF-style volume rendering."""
    delta = 1.0 / n_samples
    alpha = 1.0 - torch.exp(-sigma_samples * delta)
    
    alpha_shifted = torch.cat(
        [torch.zeros_like(alpha[:, :1]), alpha[:, :-1]], dim=1
    )
    T = torch.cumprod(1.0 - alpha_shifted + 1e-10, dim=1)
    
    weights = T * alpha
    rgb_out = (weights.unsqueeze(-1) * rgb_samples).sum(dim=1)
    rgb_out = rgb_out.permute(0, 3, 1, 2)
    return rgb_out


def render_volume(sigma, rgb, K, poses, img_size=(64, 64),
                  n_samples=64, scene_radius=1.5, device="cpu"):
    """Render volume from multiple poses."""
    H, W = img_size
    images = []
    for (R, t) in poses:
        pts = generate_rays(H, W, K, R, t,
                            n_samples=n_samples,
                            near=0.1, far=5.0,
                            device=device)
        sigma_s, rgb_s = sample_volume(sigma, rgb, pts, scene_radius=scene_radius)
        rgb_img = volume_render(sigma_s, rgb_s, n_samples)
        images.append(rgb_img[0])
    return images


# ==================== Flow-Based Voxel Volume ====================

class FlowBasedVoxelVolume(nn.Module):
    """
    Voxel volume parameterized by density FLOWS between neighbors.
    
    Instead of directly optimizing voxel densities, we optimize:
    - Base density field (sigma_base)
    - Flow parameters for each of 6 neighbor directions (flow_x+, flow_x-, etc.)
    
    The actual density is computed by:
        sigma[i] = sigma_base[i] + sum of incoming flows - sum of outgoing flows
    """
    
    def __init__(self, initial_sigma, initial_rgb, flow_scale=1.0):
        """
        Args:
            initial_sigma: [D, H, W] initial density field
            initial_rgb: [D, H, W, 3] initial color field
            flow_scale: scaling factor for flow magnitudes
        """
        super().__init__()
        
        D, H, W = initial_sigma.shape
        self.grid_size = (D, H, W)
        self.flow_scale = flow_scale
        
        # Base density (starting point)
        self.sigma_base = nn.Parameter(
            torch.from_numpy(initial_sigma).float().unsqueeze(0).unsqueeze(0)
        )  # [1, 1, D, H, W]
        
        # Flow parameters for 6 directions: +x, -x, +y, -y, +z, -z
        # Each flow[i,j,k] represents density flowing FROM voxel to its neighbor
        # Initialize to zero (no flow initially)
        self.flow_xp = nn.Parameter(torch.zeros(1, 1, D, H, W))  # flow to x+1
        self.flow_xn = nn.Parameter(torch.zeros(1, 1, D, H, W))  # flow to x-1
        self.flow_yp = nn.Parameter(torch.zeros(1, 1, D, H, W))  # flow to y+1
        self.flow_yn = nn.Parameter(torch.zeros(1, 1, D, H, W))  # flow to y-1
        self.flow_zp = nn.Parameter(torch.zeros(1, 1, D, H, W))  # flow to z+1
        self.flow_zn = nn.Parameter(torch.zeros(1, 1, D, H, W))  # flow to z-1
        
        # Color field (optimized directly via regular per-voxel gradients)
        # Convert from [0, 1] to logits for proper optimization
        rgb_tensor = torch.from_numpy(initial_rgb).float().permute(3, 0, 1, 2).unsqueeze(0)
        # Clamp to avoid inf in logit (very small epsilon to minimize distortion)
        rgb_tensor = torch.clamp(rgb_tensor, 1e-7, 1 - 1e-7)
        rgb_logits = torch.logit(rgb_tensor)  # Inverse sigmoid
        self.rgb = nn.Parameter(rgb_logits)  # [1, 3, D, H, W]
    
    def compute_sigma_from_flows(self):
        """
        Compute actual density field from base + flows.
        
        For each voxel, density = base + incoming flows - outgoing flows
        """
        D, H, W = self.grid_size
        
        # Start with base density
        sigma = self.sigma_base.clone()
        
        # Apply flows (scaled)
        # Flow in +x direction: remove from current, add to x+1
        flow_xp_scaled = self.flow_xp * self.flow_scale
        sigma[:, :, :, :, :-1] -= flow_xp_scaled[:, :, :, :, :-1]  # outgoing
        sigma[:, :, :, :, 1:] += flow_xp_scaled[:, :, :, :, :-1]   # incoming
        
        # Flow in -x direction: remove from current, add to x-1
        flow_xn_scaled = self.flow_xn * self.flow_scale
        sigma[:, :, :, :, 1:] -= flow_xn_scaled[:, :, :, :, 1:]    # outgoing
        sigma[:, :, :, :, :-1] += flow_xn_scaled[:, :, :, :, 1:]   # incoming
        
        # Flow in +y direction
        flow_yp_scaled = self.flow_yp * self.flow_scale
        sigma[:, :, :, :-1, :] -= flow_yp_scaled[:, :, :, :-1, :]
        sigma[:, :, :, 1:, :] += flow_yp_scaled[:, :, :, :-1, :]
        
        # Flow in -y direction
        flow_yn_scaled = self.flow_yn * self.flow_scale
        sigma[:, :, :, 1:, :] -= flow_yn_scaled[:, :, :, 1:, :]
        sigma[:, :, :, :-1, :] += flow_yn_scaled[:, :, :, 1:, :]
        
        # Flow in +z direction
        flow_zp_scaled = self.flow_zp * self.flow_scale
        sigma[:, :, :-1, :, :] -= flow_zp_scaled[:, :, :-1, :, :]
        sigma[:, :, 1:, :, :] += flow_zp_scaled[:, :, :-1, :, :]
        
        # Flow in -z direction
        flow_zn_scaled = self.flow_zn * self.flow_scale
        sigma[:, :, 1:, :, :] -= flow_zn_scaled[:, :, 1:, :, :]
        sigma[:, :, :-1, :, :] += flow_zn_scaled[:, :, 1:, :, :]
        
        # Ensure non-negative density
        sigma = F.relu(sigma)
        
        return sigma
    
    def forward(self):
        """Return current density and color fields."""
        sigma = self.compute_sigma_from_flows()
        rgb = torch.sigmoid(self.rgb)  # Apply sigmoid to logits → [0, 1] range
        return sigma, rgb
    
    def get_total_flow_magnitude(self):
        """Compute total magnitude of all flows (for regularization)."""
        return (
            self.flow_xp.abs().sum() +
            self.flow_xn.abs().sum() +
            self.flow_yp.abs().sum() +
            self.flow_yn.abs().sum() +
            self.flow_zp.abs().sum() +
            self.flow_zn.abs().sum()
        )


# ==================== Variance Penalty ====================

def compute_local_variance_penalty(sigma):
    """
    Compute local variance to encourage sharp boundaries.
    
    For each voxel, compute the variance of density in its local neighborhood.
    High variance = sharp boundaries (empty vs filled) - GOOD!
    Low variance = diffuse/uniform - BAD!
    
    We SUBTRACT this from the loss, so higher variance → lower loss.
    This penalizes uniformity and encourages concentrated density.
    
    Args:
        sigma: [1, 1, D, H, W] density field
    
    Returns:
        variance: scalar, mean variance across all voxels (higher is better)
    """
    # Compute local mean using 3x3x3 convolution
    kernel_size = 3
    padding = kernel_size // 2
    
    # Create averaging kernel
    kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size,
                       device=sigma.device, dtype=sigma.dtype) / (kernel_size ** 3)
    
    # Local mean
    local_mean = F.conv3d(sigma, kernel, padding=padding)
    
    # Local variance: E[(x - μ)²] = E[x²] - μ²
    sigma_squared = sigma ** 2
    local_mean_squared = F.conv3d(sigma_squared, kernel, padding=padding)
    local_variance = local_mean_squared - local_mean ** 2
    
    # Return mean variance (we want to minimize this)
    return local_variance.mean()


def compute_global_variance_penalty(sigma):
    """
    Compute global variance of non-zero voxels.
    
    High variance = good (voxels are either empty OR filled, not in-between)
    Low variance = bad (all voxels at same medium density = diffuse)
    
    We SUBTRACT this from the loss to encourage binary-like density distribution.
    """
    # Only consider non-zero voxels
    mask = sigma > 0.01
    if mask.sum() == 0:
        return torch.tensor(0.0, device=sigma.device)
    
    sigma_nonzero = sigma[mask]
    variance = sigma_nonzero.var()
    return variance


# ==================== Training Loop ====================

def differentiable_harden(
    npz_path,
    out_dir,
    n_iters=1000,
    flow_lr=0.01,
    variance_penalty_weight=0.1,
    flow_regularization_weight=0.001,
    n_views=8,
    img_res=(64, 64),
    n_samples=64,
    scene_radius=1.5,
    fov_y_deg=45.0,
    use_local_variance=True,
    use_global_variance=True,
):
    """
    Main training loop for differentiable hardening.
    
    Args:
        npz_path: Path to input .npz file with sigma and rgb
        out_dir: Output directory
        n_iters: Number of optimization iterations
        flow_lr: Learning rate for flow parameters
        variance_penalty_weight: Weight for variance penalty term
        flow_regularization_weight: Weight for flow magnitude regularization
        n_views: Number of views to render for reconstruction loss
        img_res: Image resolution for rendering
        n_samples: Number of samples per ray
        scene_radius: Scene radius in world units
        fov_y_deg: Field of view in degrees
        use_local_variance: Whether to use local variance penalty
        use_global_variance: Whether to use global variance penalty
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load initial volume
    print(f"Loading volume from {npz_path}")
    data = np.load(npz_path)
    sigma_init = data["sigma"]  # [D, H, W]
    rgb_init = data["rgb"]      # [D, H, W, 3]
    print(f"Volume shape: {sigma_init.shape}, RGB shape: {rgb_init.shape}")
    
    # Create flow-based volume
    flow_vol = FlowBasedVoxelVolume(
        sigma_init, rgb_init, flow_scale=0.1
    ).to(device)
    
    # Optimizer - optimizes BOTH flow parameters (for density) AND rgb logits (for color)
    # Flow parameters control density distribution, RGB uses standard per-voxel gradients
    optimizer = optim.Adam(flow_vol.parameters(), lr=flow_lr)
    
    # Setup camera poses for rendering
    D, H_vol, W_vol = sigma_init.shape
    H_img, W_img = img_res
    K = make_intrinsics(H_img, W_img, fov_y_deg=fov_y_deg, device=device)
    
    # Create circular orbit of camera poses
    poses = []
    for i in range(n_views):
        theta = 2 * math.pi * i / n_views
        poses.append(build_orbit_pose(theta, radius=2.5, device=device))
    
    # Render initial volume to get target images
    print("Rendering initial volume as target...")
    with torch.no_grad():
        sigma_init_t, rgb_init_t = flow_vol()
        target_images = render_volume(
            sigma_init_t, rgb_init_t, K, poses,
            img_size=img_res, n_samples=n_samples,
            scene_radius=scene_radius, device=device
        )
        target_stack = torch.stack(target_images, dim=0)  # [V, 3, H, W]
    
    print(f"\nStarting optimization for {n_iters} iterations...")
    print(f"  Flow learning rate: {flow_lr}")
    print(f"  Variance reward weight: {variance_penalty_weight} (higher variance = better)")
    print(f"  Flow regularization weight: {flow_regularization_weight}")
    print(f"  Using local variance: {use_local_variance}")
    print(f"  Using global variance: {use_global_variance}")
    print()
    
    # Training loop
    for it in range(n_iters):
        optimizer.zero_grad()
        
        # Forward pass
        sigma, rgb = flow_vol()
        
        # Render current volume
        pred_images = render_volume(
            sigma, rgb, K, poses,
            img_size=img_res, n_samples=n_samples,
            scene_radius=scene_radius, device=device
        )
        pred_stack = torch.stack(pred_images, dim=0)
        
        # Reconstruction loss (MSE)
        loss_recon = F.mse_loss(pred_stack, target_stack)
        
        # Variance reward (encourage HIGH variance = concentrated density)
        variance_reward = 0.0
        if use_local_variance:
            loss_local_var = compute_local_variance_penalty(sigma)
            variance_reward += loss_local_var
        if use_global_variance:
            loss_global_var = compute_global_variance_penalty(sigma)
            variance_reward += loss_global_var
        
        # Flow regularization (prevent excessive flow)
        loss_flow_reg = flow_vol.get_total_flow_magnitude() / sigma.numel()
        
        # Total loss
        # NOTE: We SUBTRACT variance to REWARD high variance (penalize uniformity)
        # High variance = sharp boundaries (empty vs filled voxels)
        # Low variance = diffuse/uniform density (what we want to avoid)
        loss = (
            loss_recon
            - variance_penalty_weight * variance_reward  # Subtract to encourage HIGH variance
            + flow_regularization_weight * loss_flow_reg
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent RGB from going crazy
        torch.nn.utils.clip_grad_norm_(flow_vol.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Logging
        if it % 10 == 0 or it == 0:
            var_val = variance_reward if isinstance(variance_reward, float) else variance_reward.item()
            print(
                f"[{it:4d}/{n_iters}] "
                f"loss={loss.item():.6f} "
                f"(recon={loss_recon.item():.6f}, "
                f"variance={var_val:.6f}, "
                f"flow_reg={loss_flow_reg.item():.6f})"
            )
        
        # Save intermediate results
        if it % 100 == 0 or it == n_iters - 1:
            save_iteration(flow_vol, out_dir, it, scene_radius)
    
    print("\nOptimization complete!")
    
    # Save final results
    print("Saving final results...")
    save_final_results(flow_vol, out_dir, poses, K, img_res, n_samples, scene_radius, device)
    
    print(f"Done! Results saved to {out_dir}")


def save_iteration(flow_vol, out_dir, iteration, scene_radius):
    """Save intermediate results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    with torch.no_grad():
        sigma, rgb = flow_vol()
        
        # Convert to numpy
        sigma_np = sigma[0, 0].cpu().numpy()
        rgb_np = rgb[0].cpu().numpy().transpose(1, 2, 3, 0)  # [D, H, W, 3]
        
        # Save as npz
        npz_path = out_dir / f"hardened_iter_{iteration:04d}.npz"
        np.savez_compressed(npz_path, sigma=sigma_np, rgb=rgb_np)
        
        print(f"  Saved iteration {iteration} to {npz_path}")


def save_final_results(flow_vol, out_dir, poses, K, img_res, n_samples, scene_radius, device):
    """Save final hardened volume and rendered views."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    with torch.no_grad():
        sigma, rgb = flow_vol()
        
        # Convert to numpy
        sigma_np = sigma[0, 0].cpu().numpy()
        rgb_np = rgb[0].cpu().numpy().transpose(1, 2, 3, 0)
        
        # Save final volume
        final_npz = out_dir / "hardened_final.npz"
        np.savez_compressed(final_npz, sigma=sigma_np, rgb=rgb_np)
        print(f"Saved final volume to {final_npz}")
        
        # Render and save views
        print("Rendering final views...")
        pred_images = render_volume(
            sigma, rgb, K, poses,
            img_size=img_res, n_samples=n_samples,
            scene_radius=scene_radius, device=device
        )
        
        for i, img in enumerate(pred_images):
            img_np = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_path = out_dir / f"hardened_view_{i:03d}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        
        print(f"Saved {len(pred_images)} rendered views")
        
        # Export as PLY
        export_ply(sigma_np, rgb_np, out_dir / "hardened_voxels.ply", thresh=0.5)


def export_ply(sigma, rgb, out_path, thresh=0.5):
    """Export voxels as PLY file."""
    D, H, W = sigma.shape
    mask = sigma > thresh
    idxs = np.argwhere(mask)
    
    if len(idxs) == 0:
        print("No voxels above threshold, skipping PLY export")
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
        a = int(np.clip(sigma_norm[z_i, y_i, x_i] * 255.0, 0, 255))
        colors.append((int(c_rgb[0]), int(c_rgb[1]), int(c_rgb[2]), a))
    
    with open(out_path, "w") as f:
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
    
    print(f"Exported {n} voxels to {out_path}")


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description="Differentiable hardening via density flow optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python differentiable_hardening.py --npz video_voxel_out/recon_volume.npz
  
  # High-quality hardening with more iterations
  python differentiable_hardening.py --npz video_voxel_out/recon_volume.npz \\
      --n-iters 2000 --flow-lr 0.005 --variance-penalty 0.2
  
  # Focus on local variance (sharper boundaries)
  python differentiable_hardening.py --npz video_voxel_out/recon_volume.npz \\
      --use-local-variance --no-global-variance --variance-penalty 0.5
        """
    )
    
    parser.add_argument(
        "--npz", type=str, required=True,
        help="Path to input .npz file (with sigma and rgb arrays)"
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: same as input npz)"
    )
    parser.add_argument(
        "--n-iters", type=int, default=1000,
        help="Number of optimization iterations (default: 1000)"
    )
    parser.add_argument(
        "--flow-lr", type=float, default=0.01,
        help="Learning rate for flow parameters (default: 0.01)"
    )
    parser.add_argument(
        "--variance-penalty", type=float, default=0.1,
        help="Weight for variance reward (penalizes uniformity, rewards high variance) (default: 0.1)"
    )
    parser.add_argument(
        "--flow-reg", type=float, default=0.001,
        help="Weight for flow regularization (default: 0.001)"
    )
    parser.add_argument(
        "--n-views", type=int, default=8,
        help="Number of views for reconstruction loss (default: 8)"
    )
    parser.add_argument(
        "--img-res", type=int, nargs=2, default=[64, 64],
        help="Image resolution for rendering (default: 64 64)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=64,
        help="Samples per ray (default: 64)"
    )
    parser.add_argument(
        "--scene-radius", type=float, default=1.5,
        help="Scene radius (default: 1.5)"
    )
    parser.add_argument(
        "--fov", type=float, default=45.0,
        help="Field of view in degrees (default: 45.0)"
    )
    parser.add_argument(
        "--use-local-variance", action="store_true", default=True,
        help="Use local variance penalty (default: True)"
    )
    parser.add_argument(
        "--no-local-variance", action="store_false", dest="use_local_variance",
        help="Disable local variance penalty"
    )
    parser.add_argument(
        "--use-global-variance", action="store_true", default=True,
        help="Use global variance penalty (default: True)"
    )
    parser.add_argument(
        "--no-global-variance", action="store_false", dest="use_global_variance",
        help="Disable global variance penalty"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.out_dir is None:
        args.out_dir = Path(args.npz).parent
    
    # Print configuration
    print("=" * 60)
    print("DIFFERENTIABLE HARDENING VIA DENSITY FLOW")
    print("=" * 60)
    print(f"Input: {args.npz}")
    print(f"Output: {args.out_dir}")
    print(f"Iterations: {args.n_iters}")
    print(f"Flow learning rate: {args.flow_lr}")
    print(f"Variance reward weight: {args.variance_penalty}")
    print(f"Flow regularization: {args.flow_reg}")
    print(f"Views: {args.n_views}")
    print(f"Image resolution: {args.img_res[0]}×{args.img_res[1]}")
    print(f"Samples per ray: {args.n_samples}")
    print(f"Local variance: {args.use_local_variance}")
    print(f"Global variance: {args.use_global_variance}")
    print("=" * 60)
    print()
    
    # Run optimization
    differentiable_harden(
        npz_path=args.npz,
        out_dir=args.out_dir,
        n_iters=args.n_iters,
        flow_lr=args.flow_lr,
        variance_penalty_weight=args.variance_penalty,
        flow_regularization_weight=args.flow_reg,
        n_views=args.n_views,
        img_res=tuple(args.img_res),
        n_samples=args.n_samples,
        scene_radius=args.scene_radius,
        fov_y_deg=args.fov,
        use_local_variance=args.use_local_variance,
        use_global_variance=args.use_global_variance,
    )


if __name__ == "__main__":
    main()

