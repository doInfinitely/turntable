# video_orbit_voxel_recon.py
# Reconstruct a 3D voxel grid from an orbiting video, assuming:
# - circular orbit of radius 1
# - constant angular speed
# - camera always looks at the scene center (0,0,0)
#
# You provide:
#   video_path, orbit_period_frames, direction (+1 or -1), start_frame

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim


# ----------------- Camera & volume utilities -----------------

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


def build_orbit_pose(theta, radius=1.0, device="cpu"):
    """
    Single orbit pose at angle theta (radians) in XZ-plane.
    Camera center: (r cos θ, 0, r sin θ), looking at origin, y-up.
    Returns (R [3,3], t [3,1]) as torch tensors on device.
    """
    # Camera center in world coords
    C = np.array([radius * math.cos(theta),
                  0.0,
                  radius * math.sin(theta)], dtype=np.float32)

    look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up      = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    forward = (look_at - C)
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)

    true_up = np.cross(right, forward)

    # world->camera rotation
    R = np.stack([right, true_up, -forward], axis=0)
    t = -R @ C[:, None]

    R_t = torch.from_numpy(R).to(device)
    t_t = torch.from_numpy(t).to(device)
    return R_t, t_t


def build_orbit_poses_from_period(
    frame_indices,
    orbit_period_frames,
    direction,
    radius=1.0,
    device="cpu",
    theta0=0.0,
):
    """
    Given:
      - frame_indices: list of video frame indices (ints)
      - orbit_period_frames: frames per 2π rotation
      - direction: +1 (CCW) or -1 (CW)
    Returns:
      poses: list of (R, t) for each frame index in frame_indices
    """
    dtheta = direction * 2.0 * math.pi / float(orbit_period_frames)
    poses = []
    for k in frame_indices:
        theta = theta0 + k * dtheta
        poses.append(build_orbit_pose(theta, radius=radius, device=device))
    return poses


class VoxelVolume(nn.Module):
    def __init__(self, grid_size=32):
        super().__init__()
        self.grid_size = grid_size
        # density logits and RGB per voxel
        self.density = nn.Parameter(
            torch.randn(1, 1, grid_size, grid_size, grid_size) * 0.1
        )
        self.color = nn.Parameter(
            torch.rand(1, 3, grid_size, grid_size, grid_size)
        )

    def forward(self):
        sigma = F.softplus(self.density)           # [1,1,D,H,W]
        rgb   = torch.sigmoid(self.color)          # [1,3,D,H,W]
        return sigma, rgb


def world_to_grid(pts_world, scene_radius=1.5):
    """
    Map world coordinates to [-1,1]^3.
    If your orbit radius=1, scene_radius ~1.5 leaves margin around object.
    """
    return pts_world / scene_radius


def generate_rays(h, w, K, R, t, n_samples=64,
                  near=0.5, far=2.5, device="cpu"):
    """
    Generate 3D sample points along rays for a single camera.
    Returns pts_world: [1, S, H, W, 3]
    """
    ys, xs = torch.meshgrid(
        torch.linspace(0, h - 1, h, device=device),
        torch.linspace(0, w - 1, w, device=device),
        indexing="ij",
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
    pts = pts.unsqueeze(0)                    # 1xSxHxWx3
    return pts


def sample_volume(sigma, rgb, pts_world, scene_radius=1.5):
    """
    sigma: [1,1,D,H,W]
    rgb:   [1,3,D,H,W]
    pts_world: [1,S,H,W,3]
    Returns:
        sigma_samples: [1,S,H,W]
        rgb_samples:   [1,S,H,W,3]
    """
    pts_grid = world_to_grid(pts_world, scene_radius)  # [-1,1]^3
    _, S, H, W, _ = pts_grid.shape
    grid = pts_grid.view(1, S, H, W, 3)

    sigma_samples = F.grid_sample(
        sigma, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # [1,1,S,H,W]
    rgb_samples = F.grid_sample(
        rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # [1,3,S,H,W]

    sigma_samples = sigma_samples.squeeze(1)           # [1,S,H,W]
    rgb_samples   = rgb_samples.permute(0, 2, 3, 4, 1) # [1,S,H,W,3]
    return sigma_samples, rgb_samples


def volume_render(sigma_samples, rgb_samples, n_samples):
    """
    NeRF-style compositing.
    sigma_samples: [1,S,H,W]
    rgb_samples:   [1,S,H,W,3]
    Returns:
      rgb_out: [1,3,H,W]
    """
    delta = 1.0 / n_samples
    alpha = 1.0 - torch.exp(-sigma_samples * delta)   # [1,S,H,W]

    alpha_shifted = torch.cat(
        [torch.zeros_like(alpha[:, :1]), alpha[:, :-1]], dim=1
    )
    T = torch.cumprod(1.0 - alpha_shifted + 1e-10, dim=1)  # [1,S,H,W]

    weights = T * alpha
    rgb_out = (weights.unsqueeze(-1) * rgb_samples).sum(dim=1)  # [1,H,W,3]
    rgb_out = rgb_out.permute(0, 3, 1, 2)                      # [1,3,H,W]
    return rgb_out


def render_volume(sigma, rgb, K, poses, img_size=(64, 64),
                  n_samples=64, scene_radius=1.5, device="cpu"):
    H, W = img_size
    images = []
    for (R, t) in poses:
        pts = generate_rays(H, W, K, R, t,
                            n_samples=n_samples,
                            near=0.5, far=2.5,
                            device=device)
        sigma_s, rgb_s = sample_volume(sigma, rgb, pts, scene_radius=scene_radius)
        rgb_img = volume_render(sigma_s, rgb_s, n_samples)  # [1,3,H,W]
        images.append(rgb_img[0])
    return images


# ----------------- Video loading -----------------

def load_frames_as_tensors(video_path, frame_indices, img_res=(64, 64), device="cpu"):
    """
    Load specific frames from a video, resize, return tensor [V,3,H,W] in [0,1].

    More robust version:
      - clamps requested indices to valid [0, frame_count)
      - reads sequentially instead of random seeking for each frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Keep only indices that are actually in-range
    requested = sorted(set(frame_indices))
    valid_indices = [i for i in requested if 0 <= i < frame_count]

    if not valid_indices:
        cap.release()
        raise RuntimeError(
            f"No valid frame indices; requested {requested}, "
            f"video has {frame_count} frames."
        )

    dropped = set(requested) - set(valid_indices)
    if dropped:
        print(
            f"[WARN] Dropping out-of-range frame indices {sorted(dropped)}; "
            f"video has {frame_count} frames."
        )

    Ht, Wt = img_res
    frames = []
    wanted = set(valid_indices)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx in wanted:
            # BGR -> RGB, resize, convert to tensor
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (Wt, Ht), interpolation=cv2.INTER_AREA)
            frame_f = torch.from_numpy(frame).float() / 255.0  # HxWx3
            frame_f = frame_f.permute(2, 0, 1)                  # 3xHxW
            frames.append(frame_f)

            if len(frames) == len(valid_indices):
                # We’ve collected all requested frames
                break

        idx += 1

    cap.release()

    if len(frames) != len(valid_indices):
        raise RuntimeError(
            f"Requested {len(valid_indices)} valid frames, "
            f"but only read {len(frames)} from the stream."
        )

    stack = torch.stack(frames, dim=0).to(device)  # Vx3xHxW
    return stack, valid_indices


# ----------------- Training from video frames -----------------

def train_from_video(
    video_path,
    orbit_period_frames,
    direction,
    start_frame=0,
    frame_step=4,
    grid_size=32,
    img_res=(64,64),
    n_samples=64,
    n_iters=500,
    scene_radius=1.5,
    fov_y_deg=45.0,
    out_dir="video_voxel_out",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1) Choose which frames to use as views
    frame_indices = list(range(start_frame,
                               start_frame + orbit_period_frames,
                               frame_step))
    print("Requested frames:", frame_indices)

    # 2) Load frames as gt images (note the new return: gt_stack, used_indices)
    gt_stack, used_indices = load_frames_as_tensors(
        video_path, frame_indices,
        img_res=img_res, device=device
    )
    print("Actually using frames:", used_indices)

    V, C, H, W = gt_stack.shape
    print(f"Loaded {V} frames at {H}x{W}")

    # 3) Camera intrinsics + poses from orbit assumption
    K = make_intrinsics(H, W, fov_y_deg=fov_y_deg, device=device)
    poses = build_orbit_poses_from_period(
        used_indices,                    # <-- use the actual indices we got
        orbit_period_frames=orbit_period_frames,
        direction=direction,
        radius=1.0,
        device=device,
        theta0=0.0,
    )

    # 4) Learnable volume
    recon_vol = VoxelVolume(grid_size=grid_size).to(device)
    with torch.no_grad():
        recon_vol.density[:] = torch.randn_like(recon_vol.density) * 0.1
        recon_vol.color[:]   = torch.rand_like(recon_vol.color)

    opt = optim.Adam(recon_vol.parameters(), lr=1e-2)

    # 5) Train loop
    for it in range(n_iters):
        sigma_rec, rgb_rec = recon_vol()

        pred_images = render_volume(
            sigma_rec, rgb_rec, K, poses,
            img_size=img_res,
            n_samples=n_samples,
            scene_radius=scene_radius,
            device=device,
        )  # list of V [3,H,W]

        pred_stack = torch.stack(pred_images, dim=0)  # Vx3xHxW

        loss = F.mse_loss(pred_stack, gt_stack)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 10 == 0 or it == 0:
            print(f"[{it}/{n_iters}] loss={loss.item():.6e}")

    # 6) Save recon views and voxel cloud
    sigma_rec, rgb_rec = recon_vol()
    rec_images = render_volume(
        sigma_rec, rgb_rec, K, poses,
        img_size=img_res,
        n_samples=n_samples,
        scene_radius=scene_radius,
        device=device,
    )

    for i, img in enumerate(rec_images):
        img_np = (img.clamp(0,1).permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"recon_{i:03d}.png"),
                    cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # Export voxels as PLY (optional)
    export_voxels_as_ply(
        sigma_rec[0,0].detach().cpu().numpy(),
        rgb_rec[0].permute(1,2,3,0).detach().cpu().numpy(),
        out_dir / "recon_voxels.ply",
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
    import sys

    if len(sys.argv) < 5:
        print("Usage: python video_orbit_voxel_recon.py "
              "<video_path> <orbit_period_frames> <direction(+1|-1)> <start_frame>")
        sys.exit(1)

    video_path = sys.argv[1]
    orbit_period_frames = int(sys.argv[2])
    direction = int(sys.argv[3])
    start_frame = int(sys.argv[4])

    train_from_video(
        video_path=video_path,
        orbit_period_frames=orbit_period_frames,
        direction=direction,
        start_frame=start_frame,
        frame_step=4,          # you can tweak
        grid_size=32,
        img_res=(64,64),
        n_samples=64,
        n_iters=500,
        scene_radius=1.5,
        fov_y_deg=45.0,
        out_dir="video_voxel_out",
    )

