# voxel_turntable.py
# Synthetic voxel NeRF demo (fixed):
# 1) Analytic colored sphere as GT voxel grid
# 2) Render images from known orbit camera poses
# 3) Optimize a new voxel grid to reconstruct GT from those images

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim


# ----------------- Camera utilities -----------------

def make_intrinsics(h, w, fov_y_deg=45.0, device="cpu"):
    fov_y = math.radians(fov_y_deg)
    fy = 0.5 * h / math.tan(0.5 * fov_y)
    fx = fy
    cx = w / 2.0
    cy = h / 2.0
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=torch.float32, device=device)
    return K


def build_orbit_poses(n_views, radius=2.5, device="cpu"):
    """
    Simple circular orbit around origin, y-up.
    Returns list of (R [3,3], t [3,1]) on given device.
    """
    poses = []
    for i in range(n_views):
        theta = 2.0 * math.pi * i / n_views
        C = np.array([radius * math.cos(theta), 0.0, radius * math.sin(theta)],
                     dtype=np.float32)
        look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        z_cam = C - look_at
        z_cam /= np.linalg.norm(z_cam) + 1e-8
        x_cam = np.cross(up, z_cam)
        x_cam /= np.linalg.norm(x_cam) + 1e-8
        y_cam = np.cross(z_cam, x_cam)

        R = np.stack([x_cam, y_cam, z_cam], axis=0)  # world->cam
        t = -R @ C[:, None]

        R_t = torch.from_numpy(R).to(device)
        t_t = torch.from_numpy(t).to(device)
        poses.append((R_t, t_t))
    return poses


def generate_rays(h, w, K, R, t, n_samples=64,
                  near=1.0, far=3.5, device="cpu"):
    """
    Generate 3D sample points along rays for a single camera.
    Returns pts_world: [1, n_samples, H, W, 3]
    """
    ys, xs = torch.meshgrid(
        torch.linspace(0, h - 1, h, device=device),
        torch.linspace(0, w - 1, w, device=device),
        indexing="ij"
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=-1)  # HxWx3

    K_inv = torch.inverse(K)
    dirs_cam = (K_inv @ pix.reshape(-1, 3).T).T  # (H*W)x3
    dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)

    R = R.to(device)
    t = t.to(device)

    dirs_world = (R.transpose(0, 1) @ dirs_cam.T).T  # (H*W)x3
    C = -(R.transpose(0, 1) @ t).reshape(1, 3)       # 1x3

    ts = torch.linspace(near, far, n_samples, device=device).view(-1, 1, 1)
    dirs_world = dirs_world.reshape(1, h, w, 3)
    C_exp = C.view(1, 1, 1, 3)

    pts = C_exp + ts[..., None] * dirs_world  # SxHxWx3 (after broadcast)
    pts = pts.unsqueeze(0)  # 1xSxHxWx3
    return pts


def world_to_grid(pts_world, scene_radius=3.5):
    """
    Map world coords to [-1,1]^3 for grid_sample.

    We assume all points lie roughly inside a sphere of radius <= scene_radius.
    """
    return pts_world / scene_radius  # now in [-1,1] approx


# ----------------- Voxel volume + renderer -----------------

class VoxelVolume(nn.Module):
    def __init__(self, grid_size=32):
        super().__init__()
        self.grid_size = grid_size
        # density logits and RGB per voxel
        self.density = nn.Parameter(torch.randn(1, 1, grid_size, grid_size, grid_size) * 0.1)
        self.color = nn.Parameter(torch.rand(1, 3, grid_size, grid_size, grid_size))

    def forward(self):
        # density: [1,1,D,H,W], positive via softplus
        sigma = F.softplus(self.density)
        rgb = torch.sigmoid(self.color)  # [1,3,D,H,W]
        return sigma, rgb


def sample_volume(sigma, rgb, pts_world, scene_radius=3.5):
    """
    sigma: [1,1,D,H,W]
    rgb:   [1,3,D,H,W]
    pts_world: [1,S,H,W,3] world coords
    Returns:
        sigma_samples: [1,S,H,W]
        rgb_samples:   [1,S,H,W,3]
    """
    pts_grid = world_to_grid(pts_world, scene_radius)  # in [-1,1]^3

    _, S, H, W, _ = pts_grid.shape
    grid = pts_grid.view(1, S, H, W, 3)

    sigma_samples = F.grid_sample(
        sigma, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # [1,1,S,H,W]
    rgb_samples = F.grid_sample(
        rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # [1,3,S,H,W]

    sigma_samples = sigma_samples.squeeze(1)           # [1,S,H,W]
    rgb_samples = rgb_samples.permute(0, 2, 3, 4, 1)   # [1,S,H,W,3]
    return sigma_samples, rgb_samples


def volume_render(sigma_samples, rgb_samples, n_samples):
    """
    NeRF-style volume rendering along each ray.
    sigma_samples: [1,S,H,W] (density)
    rgb_samples:   [1,S,H,W,3]
    Returns:
        rgb_out: [1,3,H,W]
    """
    delta = 1.0 / n_samples

    alpha = 1.0 - torch.exp(-sigma_samples * delta)  # [1,S,H,W]

    alpha_shifted = torch.cat(
        [torch.zeros_like(alpha[:, :1]), alpha[:, :-1]], dim=1
    )  # [1,S,H,W]
    T = torch.cumprod(1.0 - alpha_shifted + 1e-10, dim=1)  # [1,S,H,W]

    weights = T * alpha  # [1,S,H,W]

    rgb_out = (weights.unsqueeze(-1) * rgb_samples).sum(dim=1)  # [1,H,W,3]
    rgb_out = rgb_out.permute(0, 3, 1, 2)  # [1,3,H,W]
    return rgb_out


def render_volume(sigma, rgb, K, poses, img_size=(64, 64),
                  n_samples=64, scene_radius=3.5, device="cpu"):
    """
    Render all views from a given volume.
    Returns list of [3,H,W] tensors in [0,1].
    """
    H, W = img_size
    images = []
    for i, (R, t) in enumerate(poses):
        pts = generate_rays(H, W, K, R, t,
                            n_samples=n_samples,
                            near=1.0, far=3.5,
                            device=device)
        sigma_s, rgb_s = sample_volume(sigma, rgb, pts, scene_radius=scene_radius)
        rgb_img = volume_render(sigma_s, rgb_s, n_samples)  # [1,3,H,W]
        images.append(rgb_img[0])  # [3,H,W]
    return images


# ----------------- Analytic GT volume (sphere) -----------------

def make_sphere_gt_volume(grid_size=32, device="cpu"):
    """
    Create an analytic GT volume: colored sphere centered at origin.

    - sigma is high inside radius r<=0.6, 0 outside.
    - color varies smoothly with position for a nontrivial texture.
    """
    D = H = W = grid_size

    zs = torch.linspace(-1, 1, D, device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)

    z, y, x = torch.meshgrid(zs, ys, xs, indexing="ij")  # D,H,W

    r = torch.sqrt(x * x + y * y + z * z)

    sigma = torch.zeros(1, 1, D, H, W, device=device)
    rgb = torch.zeros(1, 3, D, H, W, device=device)

    inside = r <= 0.6

    # Smooth density inside sphere
    sigma[0, 0][inside] = 50  # higher towards center

    # Color: simple function of position (normalize to 0..1)
    rgb_x = (x + 1.0) / 2.0
    rgb_y = (y + 1.0) / 2.0
    rgb_z = (z + 1.0) / 2.0
    rgb[0, 0] = rgb_x
    rgb[0, 1] = rgb_y
    rgb[0, 2] = rgb_z

    return sigma, rgb


# ----------------- Training reconstruction -----------------

def train_reconstruction(device="cuda" if torch.cuda.is_available() else "cpu"):
    # Config
    grid_size = 32
    img_size = (64, 64)
    n_views = 12
    n_samples = 64
    n_iters = 500
    scene_radius = 3.5
    out_dir = Path("voxel_turntable_out")
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1) Analytic GT volume (sphere)
    sigma_gt, rgb_gt = make_sphere_gt_volume(grid_size=grid_size, device=device)

    # 2) Camera setup
    H, W = img_size
    K = make_intrinsics(H, W, fov_y_deg=45.0, device=device)
    poses = build_orbit_poses(n_views, radius=2.5, device=device)

    # 3) Render GT images
    with torch.no_grad():
        gt_images = render_volume(sigma_gt, rgb_gt, K, poses,
                                  img_size=img_size,
                                  n_samples=n_samples,
                                  scene_radius=scene_radius,
                                  device=device)

    # Save GT images for inspection
    for i, img in enumerate(gt_images):
        img_np = (img.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"gt_{i:03d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # 4) Learnable volume to reconstruct
    recon_vol = VoxelVolume(grid_size=grid_size).to(device)
    with torch.no_grad():
        # make recon start clearly different from GT: low density, random color
        recon_vol.density[:] = torch.randn_like(recon_vol.density) * 0.1
        recon_vol.color[:] = torch.rand_like(recon_vol.color)

    optim_ = optim.Adam(recon_vol.parameters(), lr=1e-2)

    # Precompute stacked GT for faster loss
    gt_stack = torch.stack(gt_images, dim=0)  # [V,3,H,W]

    # 5) Train
    for it in range(n_iters):
        sigma_rec, rgb_rec = recon_vol()
        pred_images = render_volume(sigma_rec, rgb_rec, K, poses,
                                    img_size=img_size,
                                    n_samples=n_samples,
                                    scene_radius=scene_radius,
                                    device=device)

        pred_stack = torch.stack(pred_images, dim=0)

        loss = F.mse_loss(pred_stack, gt_stack)

        optim_.zero_grad()
        loss.backward()
        optim_.step()

        if it % 10 == 0 or it == 0:
            print(f"[{it}/{n_iters}] loss={loss.item():.6e}")

    # 6) Save a few recon images and voxel point cloud
    sigma_rec, rgb_rec = recon_vol()
    rec_images = render_volume(sigma_rec, rgb_rec, K, poses,
                               img_size=img_size,
                               n_samples=n_samples,
                               scene_radius=scene_radius,
                               device=device)
    for i, img in enumerate(rec_images):
        img_np = (img.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"recon_{i:03d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    export_voxels_as_ply(
        sigma_rec[0, 0].cpu().detach().numpy(),
        rgb_rec[0].permute(1, 2, 3, 0).cpu().detach().numpy(),
        out_dir / "recon_voxels.ply"
    )

    print("Done. Check:", out_dir)


def export_voxels_as_ply(sigma, rgb, out_path, thresh=0.5):
    """
    sigma: [D,H,W]
    rgb:   [D,H,W,3] in [0,1]
    """
    D, H, W = sigma.shape
    mask = sigma > thresh
    idxs = np.argwhere(mask)

    if len(idxs) == 0:
        print("No voxels above threshold.")
        return

    zs = np.linspace(-1, 1, D)
    ys = np.linspace(-1, 1, H)
    xs = np.linspace(-1, 1, W)

    verts = []
    colors = []

    for z_i, y_i, x_i in idxs:
        x = xs[x_i]
        y = ys[y_i]
        z = zs[z_i]
        verts.append((x, y, z))
        c = (rgb[z_i, y_i, x_i] * 255.0).astype(np.uint8)
        colors.append((int(c[0]), int(c[1]), int(c[2])))

    verts = np.array(verts, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)

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
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(verts, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


if __name__ == "__main__":
    train_reconstruction()
