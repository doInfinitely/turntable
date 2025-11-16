# video_orbit_voxel_recon.py
# Reconstruct a 3D voxel grid from an orbiting video, assuming:
# - circular orbit of radius 1
# - constant angular speed
# - camera always looks at the scene center (0,0,0)
#
# You can either:
#   1) Let it auto-estimate orbit_period_frames & direction from the video, or
#   2) Pass them explicitly as before.

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

# Optional visualization imports
try:
    import pygame
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("[WARN] pygame not available, real-time visualization disabled")


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
    # FIXED: Camera looks down +Z, so +Z should align with forward direction
    R = np.stack([right, true_up, forward], axis=0)
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
      - direction: +1 (CCW) or -1 (CW) in our convention
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
    def __init__(self, grid_size=32, sigma_scale=1.0, init_logit=-5.0):
        super().__init__()
        self.grid_size = grid_size
        self.sigma_scale = sigma_scale

        # start very negative → sigma ≈ 0 everywhere
        self.density = nn.Parameter(
            torch.full((1, 1, grid_size, grid_size, grid_size),
                       init_logit, dtype=torch.float32)
        )
        # random colors is fine
        self.color = nn.Parameter(
            torch.rand(1, 3, grid_size, grid_size, grid_size)
        )

    def forward(self):
        # softplus(-5) ≈ 0.0067; with sigma_scale you can tune effective opacity
        sigma = F.softplus(self.density) * self.sigma_scale   # [1,1,D,H,W]
        rgb   = torch.sigmoid(self.color)                     # [1,3,D,H,W]
        return sigma, rgb
    

def world_to_grid(pts_world, scene_radius=1.5):
    """
    Map world coordinates in [-scene_radius, +scene_radius] to grid_sample coordinates in [-1, +1].
    F.grid_sample expects coordinates in [-1, 1] which map to the full extent of the voxel grid.
    """
    return pts_world / scene_radius  # This already does the right thing!


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
        # FIXED: far must be large enough to traverse entire volume
        # Camera at ~2.5, volume extends ±1.5, so far should be at least 2.5 + 1.5 = 4.0
        pts = generate_rays(H, W, K, R, t,
                            n_samples=n_samples,
                            near=0.1, far=5.0,
                            device=device)
        sigma_s, rgb_s = sample_volume(sigma, rgb, pts, scene_radius=scene_radius)
        rgb_img = volume_render(sigma_s, rgb_s, n_samples)  # [1,3,H,W]
        images.append(rgb_img[0])
    return images


# ----------------- Video helpers -----------------

def estimate_background_frame(video_path, sample_step=5, max_samples=200):
    """
    Estimate static background by taking a temporal median over sampled frames.
    Assumes background is static and subject moves / is orbited.

    Returns: bg_rgb (H,W,3 uint8)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_step == 0:
            frames.append(frame.astype(np.uint8))
            if len(frames) >= max_samples:
                break
        idx += 1

    cap.release()

    if not frames:
        raise RuntimeError("No frames read while estimating background")

    stack = np.stack(frames, axis=0)   # (N,H,W,3)
    bg = np.median(stack, axis=0).astype(np.uint8)
    return bg


def foreground_mask_from_background(frame_bgr, bg_bgr,
                                    color_thresh=25,
                                    morph_kernel=5):
    """
    frame_bgr, bg_bgr: (H,W,3) uint8
    Returns mask (H,W) uint8 in {0,255} where 255 = foreground.
    """
    diff = cv2.absdiff(frame_bgr, bg_bgr)
    dist = np.linalg.norm(diff.astype(np.float32), axis=2)  # (H,W)

    mask = (dist > color_thresh).astype(np.uint8) * 255

    if morph_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return mask


def load_frames_as_tensors(video_path, frame_indices, img_res=(64, 64), device="cpu"):
    """
    Load specific frames from a video, resize, return:
      gt_stack: [V,3,H,W] in [0,1]
      mask_stack: [V,1,H,W] in {0,1}  (1 = foreground)
      used_indices: list[int]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

    # --- estimate background using full-res frames ---
    bg_bgr = estimate_background_frame(video_path)

    Ht, Wt = img_res
    frames = []
    masks  = []
    wanted = set(valid_indices)

    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if idx in wanted:
            # compute mask BEFORE resize, to keep bg estimate aligned
            mask = foreground_mask_from_background(frame_bgr, bg_bgr)

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # resize both frame and mask to training resolution
            frame_rgb = cv2.resize(frame_rgb, (Wt, Ht), interpolation=cv2.INTER_AREA)
            mask_r = cv2.resize(mask, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

            frame_f = torch.from_numpy(frame_rgb).float() / 255.0  # HxWx3
            frame_f = frame_f.permute(2, 0, 1)                      # 3xHxW

            mask_f = torch.from_numpy(mask_r.astype(np.float32) / 255.0)  # HxW in [0,1]
            mask_f = mask_f.unsqueeze(0)  # 1xHxW

            frames.append(frame_f)
            masks.append(mask_f)

            if len(frames) == len(valid_indices):
                break

        idx += 1

    cap.release()

    if len(frames) != len(valid_indices):
        raise RuntimeError(
            f"Requested {len(valid_indices)} valid frames, "
            f"but only read {len(frames)} from the stream."
        )

    gt_stack   = torch.stack(frames, dim=0).to(device)  # [V,3,H,W]
    mask_stack = torch.stack(masks,  dim=0).to(device)  # [V,1,H,W]
    return gt_stack, mask_stack, valid_indices


# ---------- Orbit period + direction estimators (pinhole-ish) ----------

def estimate_orbit_period(video_path, start_frame=0, min_lag=10, max_frames=240):
    """
    Heuristic orbit period estimator:
      - compares each later frame to the start_frame
      - finds the first strong minimum in MSE after `min_lag`
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, f0 = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Could not read start_frame {start_frame}")

    # Downscale + grayscale to make MSE cheaper & smoother
    h0, w0 = f0.shape[:2]
    scale = 160.0 / max(w0, 160.0)
    f0_small = cv2.resize(f0, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    f0_gray = cv2.cvtColor(f0_small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diffs = []
    for k in range(1, max_frames):
        ret, fk = cap.read()
        if not ret:
            break
        fk_small = cv2.resize(fk, (f0_small.shape[1], f0_small.shape[0]), interpolation=cv2.INTER_AREA)
        fk_gray = cv2.cvtColor(fk_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = fk_gray - f0_gray
        mse = float(np.mean(diff * diff))
        diffs.append(mse)

    cap.release()

    if len(diffs) <= min_lag + 2:
        raise RuntimeError("Not enough frames to estimate period")

    diffs_arr = np.array(diffs)
    # Search for minimum after min_lag
    search = diffs_arr[min_lag:]
    offset = int(np.argmin(search))
    period_frames = offset + min_lag + 1  # +1 because diffs[k] is frame start_frame + k

    print(f"[AUTO] Estimated orbit period: {period_frames} frames, fps={fps:.2f}")
    return period_frames, fps


def estimate_orbit_direction(video_path, start_frame=0):
    """
    Very simple direction estimator:
      - compute optical flow between start_frame and start_frame+1
      - take sign of mean horizontal flow over high-magnitude pixels
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, f0 = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Could not read frame {start_frame}")
    ret, f1 = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Could not read frame {start_frame + 1}")

    # Downscale for speed
    h, w = f0.shape[:2]
    scale = 160.0 / max(w, 160.0)
    size = (int(w * scale), int(h * scale))
    g0 = cv2.cvtColor(cv2.resize(f0, size, interpolation=cv2.INTER_AREA),
                      cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cv2.resize(f1, size, interpolation=cv2.INTER_AREA),
                      cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        g0, g1,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    u = flow[..., 0]
    v = flow[..., 1]
    mag = np.sqrt(u * u + v * v)

    # Focus on high-motion regions
    thresh = np.percentile(mag, 70)
    mask = mag > thresh
    if not np.any(mask):
        mean_u = float(np.mean(u))
    else:
        mean_u = float(np.mean(u[mask]))

    direction = 1 if mean_u >= 0 else -1
    print(f"[AUTO] Estimated orbit direction from optical flow: mean_u={mean_u:.4f} → direction={direction}")
    cap.release()
    return direction


# ----------------- Regularization helpers -----------------

def make_radius_volume(grid_size, device):
    """Return r(x,y,z) on [-1,1]^3 as [1,1,D,H,W]."""
    zs = torch.linspace(-1.0, 1.0, grid_size, device=device)
    ys = torch.linspace(-1.0, 1.0, grid_size, device=device)
    xs = torch.linspace(-1.0, 1.0, grid_size, device=device)
    z, y, x = torch.meshgrid(zs, ys, xs, indexing="ij")
    r = torch.sqrt(x * x + y * y + z * z)
    return r[None, None, ...]  # [1,1,D,H,W]


def make_distance_volume(grid_size=32, scene_radius=1.5, device="cpu"):
    """
    Returns:
      dist_norm: [1,1,D,H,W] with values in [0, ~1], 0 at center, 1 at scene_radius.
    """
    D = H = W = grid_size

    zs = torch.linspace(-scene_radius, scene_radius, D, device=device)
    ys = torch.linspace(-scene_radius, scene_radius, H, device=device)
    xs = torch.linspace(-scene_radius, scene_radius, W, device=device)

    z, y, x = torch.meshgrid(zs, ys, xs, indexing="ij")  # D,H,W
    dist = torch.sqrt(x * x + y * y + z * z)            # D,H,W

    dist_norm = dist / (scene_radius + 1e-8)            # ~[0,1]
    dist_norm = dist_norm.unsqueeze(0).unsqueeze(0)     # [1,1,D,H,W]
    return dist_norm


def tv3d(x):
    """
    3D total variation regularization for a 5D tensor [B,C,D,H,W].
    Returns a scalar.
    """
    # differences along depth, height, width
    dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

    return (dx.abs().mean() + dy.abs().mean() + dz.abs().mean())


def compute_neighbor_density_sum(sigma, kernel_size=3):
    """
    For each voxel, compute the sum of density in its 3D neighborhood.
    sigma: [1,1,D,H,W]
    Returns: neighbor_sum [1,1,D,H,W]
    """
    # Create a 3D averaging kernel (all ones)
    k = kernel_size
    padding = k // 2
    
    # Use conv3d to sum neighbors
    # kernel shape: [out_channels, in_channels, kD, kH, kW]
    kernel = torch.ones(1, 1, k, k, k, device=sigma.device, dtype=sigma.dtype)
    
    neighbor_sum = F.conv3d(sigma, kernel, padding=padding)
    return neighbor_sum


def render_voxels_pygame(sigma_np, rgb_np, angle_y, angle_x, radius, 
                         img_size=(256, 256), scene_radius=1.5, 
                         fov_y_deg=45.0, thresh_factor=0.2):
    """
    Minecraft-style cube voxel renderer with actual cube faces.
    Returns: uint8 image [H,W,3]
    """
    H_img, W_img = img_size
    
    # Camera position (spherical)
    cx = radius * math.cos(angle_x) * math.cos(angle_y)
    cy = radius * math.sin(angle_x)
    cz = radius * math.cos(angle_x) * math.sin(angle_y)
    eye = np.array([cx, cy, cz], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    # Camera intrinsics
    fov_y = math.radians(fov_y_deg)
    fy = 0.5 * H_img / math.tan(0.5 * fov_y)
    fx = fy
    cx_i = W_img / 2.0
    cy_i = H_img / 2.0
    
    # Build camera R,t
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8
    true_up = np.cross(right, forward)
    
    R = np.stack([right, true_up, forward], axis=0)
    t = -R @ eye
    
    D, H, W = sigma_np.shape
    
    # Threshold voxels
    max_sigma = float(sigma_np.max()) if sigma_np.size > 0 else 0.0
    if max_sigma <= 0:
        return np.zeros((H_img, W_img, 3), dtype=np.uint8)
    
    sigma_thresh = max_sigma * thresh_factor
    mask = sigma_np > sigma_thresh
    idxs = np.argwhere(mask)
    
    if idxs.shape[0] == 0:
        return np.zeros((H_img, W_img, 3), dtype=np.uint8)
    
    # World space voxel grid
    zs = np.linspace(-1, 1, D) * scene_radius
    ys = np.linspace(-1, 1, H) * scene_radius
    xs = np.linspace(-1, 1, W) * scene_radius
    voxel_size = 2.0 * scene_radius / D  # Size of one voxel
    
    # Cube vertices (8 corners of unit cube centered at origin)
    half_size = voxel_size * 0.5
    cube_verts = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # back face
        [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1],   # front face
    ], dtype=np.float32) * half_size
    
    # Cube faces (6 faces, each defined by 4 vertex indices forming 2 triangles)
    # Each face: [v0, v1, v2, v3] where we'll render triangles (v0,v1,v2) and (v0,v2,v3)
    cube_faces = np.array([
        [0, 1, 2, 3],  # back (-Z)
        [4, 7, 6, 5],  # front (+Z)
        [0, 4, 5, 1],  # bottom (-Y)
        [3, 2, 6, 7],  # top (+Y)
        [0, 3, 7, 4],  # left (-X)
        [1, 5, 6, 2],  # right (+X)
    ], dtype=int)
    
    # Face normals for lighting
    face_normals = np.array([
        [0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]
    ], dtype=np.float32)
    
    # Create depth buffer and color buffer
    z_buffer = np.full((H_img, W_img), np.inf, dtype=np.float32)
    img = np.zeros((H_img, W_img, 3), dtype=np.uint8)
    
    # Render each voxel as a cube
    for idx in idxs:
        z_i, y_i, x_i = idx
        voxel_center = np.array([xs[x_i], ys[y_i], zs[z_i]], dtype=np.float32)
        color = (rgb_np[z_i, y_i, x_i] * 255.0).astype(np.uint8)
        
        # Transform cube vertices to world space
        verts_world = cube_verts + voxel_center
        
        # Transform to camera space
        verts_cam = (R @ verts_world.T).T + t
        
        # Render each face
        for face_idx, face in enumerate(cube_faces):
            v0, v1, v2, v3 = verts_cam[face]
            
            # Backface culling: check if face normal points towards camera
            face_normal_world = face_normals[face_idx]
            face_normal_cam = R @ face_normal_world
            if face_normal_cam[2] >= 0:  # Face pointing away from camera
                continue
            
            # Check if all vertices are in front of camera
            if v0[2] <= 0.01 or v1[2] <= 0.01 or v2[2] <= 0.01 or v3[2] <= 0.01:
                continue
            
            # Project vertices to screen
            def project(v):
                u = fx * (v[0] / v[2]) + cx_i
                v_p = fy * (v[1] / v[2]) + cy_i
                return np.array([u, v_p]), v[2]
            
            p0, z0 = project(v0)
            p1, z1 = project(v1)
            p2, z2 = project(v2)
            p3, z3 = project(v3)
            
            # Simple lighting: darken faces based on angle to camera
            light_dir = -forward  # Light from camera
            face_brightness = max(0.3, abs(np.dot(face_normal_world, light_dir)))
            lit_color = (color * face_brightness).astype(np.uint8)
            
            # Rasterize two triangles for this face
            for tri_verts, tri_depths in [([p0, p1, p2], [z0, z1, z2]), 
                                           ([p0, p2, p3], [z0, z2, z3])]:
                rasterize_triangle(img, z_buffer, tri_verts, tri_depths, lit_color, H_img, W_img)
    
    return img


def rasterize_triangle(img, z_buffer, verts_2d, depths, color, H, W):
    """
    Rasterize a single triangle with depth testing.
    verts_2d: list of 3 (u, v) screen coordinates
    depths: list of 3 depth values
    color: RGB uint8 tuple
    """
    # Get bounding box
    us = [v[0] for v in verts_2d]
    vs = [v[1] for v in verts_2d]
    
    min_u = max(0, int(np.floor(min(us))))
    max_u = min(W - 1, int(np.ceil(max(us))))
    min_v = max(0, int(np.floor(min(vs))))
    max_v = min(H - 1, int(np.ceil(max(vs))))
    
    if min_u > max_u or min_v > max_v:
        return
    
    p0, p1, p2 = verts_2d
    z0, z1, z2 = depths
    
    # Precompute triangle edge functions for barycentric coordinates
    def edge_function(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
    
    area = edge_function(p0, p1, p2)
    if abs(area) < 1e-6:  # Degenerate triangle
        return
    
    # Scan over bounding box
    for v in range(min_v, max_v + 1):
        for u in range(min_u, max_u + 1):
            p = np.array([u + 0.5, v + 0.5])  # Pixel center
            
            # Compute barycentric coordinates
            w0 = edge_function(p1, p2, p)
            w1 = edge_function(p2, p0, p)
            w2 = edge_function(p0, p1, p)
            
            # Check if point is inside triangle
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                # Normalize barycentric coords
                w0 /= area
                w1 /= area
                w2 /= area
                
                # Interpolate depth
                z = w0 * z0 + w1 * z1 + w2 * z2
                
                # Depth test
                if z < z_buffer[v, u]:
                    z_buffer[v, u] = z
                    img[v, u] = color


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
    use_neighbor_growth=False,  # True: growing from center, False: hard core
    enable_viewer=False,  # True: show live pygame viewer
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1) Choose which frames to use as views  
    # Sample frames evenly throughout the orbit using frame_step
    frame_indices = list(range(start_frame, start_frame + orbit_period_frames, frame_step))
    print("Requested frames:", frame_indices)

    # 2) Load frames as gt images
    gt_stack, mask_stack, used_indices = load_frames_as_tensors(
        video_path, frame_indices,
        img_res=img_res, device=device
    )
    print("Actually using frames:", used_indices)

    V, C, H, W = gt_stack.shape
    print(f"Loaded {V} frames at {H}x{W}")

    # 3) Camera intrinsics + poses from orbit assumption
    K = make_intrinsics(H, W, fov_y_deg=fov_y_deg, device=device)
    poses = build_orbit_poses_from_period(
        used_indices,
        orbit_period_frames=orbit_period_frames,
        direction=direction,
        radius=2.5,  # Just outside voxel grid diagonal (~2.6)
        device=device,
        theta0=0.0,
    )

    # 4) Learnable volume
    recon_vol = VoxelVolume(grid_size=grid_size, sigma_scale=1.0, init_logit=-5.0).to(device)    
    with torch.no_grad():
        # Random color initialization
        recon_vol.color.uniform_(-0.5, 0.5)  # Logits around 0 → colors around 0.5

    # Optimize BOTH density and colors (need color gradients!)
    opt = optim.Adam(recon_vol.parameters(), lr=5e-2)  # Optimize both shape and colors

    # --- Distance volume (used by both approaches) ---
    dist_vol = make_distance_volume(
        grid_size=grid_size,
        scene_radius=scene_radius,
        device=device,
    )  # [1,1,D,H,W]

    if use_neighbor_growth:
        # ===== NEIGHBOR GROWTH APPROACH =====
        # Initialize center voxels with Gaussian density
        print("[NEIGHBOR GROWTH MODE] Initializing center seed...")
        
        with torch.no_grad():
            # Create Gaussian initialization centered at the grid center
            center_z = grid_size / 2.0
            center_y = grid_size / 2.0
            center_x = grid_size / 2.0
            
            # Gaussian parameters
            peak_sigma = 200.0  # Peak density at center (alpha ≈ 0.95)
            gaussian_std = 2.0  # Standard deviation in voxels (sweet spot!)
            
            # Create coordinate grids
            z_coords = torch.arange(grid_size, dtype=torch.float32, device=device)
            y_coords = torch.arange(grid_size, dtype=torch.float32, device=device)
            x_coords = torch.arange(grid_size, dtype=torch.float32, device=device)
            
            zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
            
            # Compute squared distance from center
            dist_sq = (zz - center_z)**2 + (yy - center_y)**2 + (xx - center_x)**2
            
            # Gaussian: sigma = peak * exp(-dist^2 / (2 * std^2))
            gaussian = peak_sigma * torch.exp(-dist_sq / (2 * gaussian_std**2))
            
            # Set density (convert sigma to logit, but for large values logit ≈ sigma)
            recon_vol.density[0, 0] = gaussian
            
            # Count how many voxels are above threshold
            num_above_thresh = (gaussian > 1.0).sum().item()
            max_sigma = gaussian.max().item()
            
            print(f"  → Initialized Gaussian seed centered at [{center_z:.1f}, {center_y:.1f}, {center_x:.1f}]")
            print(f"  → Peak σ≈{max_sigma:.1f}, std={gaussian_std:.1f} voxels")
            print(f"  → {num_above_thresh} voxels with σ>1.0")
        
        # Hyperparams for neighbor growth (disable for now, just test rendering)
        target_alpha = 0.3
        neighbor_threshold = -math.log(1 - target_alpha) * n_samples
        neighbor_kernel_size = 3    # 3x3x3 neighborhood
        bg_sigma = 0.01             # fixed background density
        print(f"  → Neighbor threshold set to {neighbor_threshold:.2f} (corresponds to alpha≈{target_alpha})")
        
    else:
        # ===== HARD CORE APPROACH =====
        print("[HARD CORE MODE] Using expanding core constraint...")
        # Keep density at near-zero everywhere initially
        r_core_min = 0.3      # start with a small central ball
        r_core_max = 0.6      # optional: allow it to expand a bit
        bg_sigma   = 0.01     # tiny background density outside

    # Initialize live viewer if enabled
    viewer_state = None
    if enable_viewer and VISUALIZATION_AVAILABLE:
        pygame.init()
        win_size = 512
        screen = pygame.display.set_mode((win_size, win_size))
        pygame.display.set_caption("Voxel Training (C=cubes, V=volume, arrows=orbit, +/-=zoom)")
        clock = pygame.time.Clock()
        
        viewer_state = {
            'screen': screen,
            'clock': clock,
            'win_size': win_size,
            'angle_y': 0.0,
            'angle_x': 0.0,
            'radius': 2.5,  # Match training camera orbit radius
            'auto_rotate': True,  # auto-rotate during training
            'mode': 'cubes',  # 'cubes' or 'volume'
        }
        print("[VIEWER] Pygame window initialized (512x512)")
    elif enable_viewer and not VISUALIZATION_AVAILABLE:
        print("[WARN] Viewer requested but pygame not available")

    # 5) Train loop
    for it in range(n_iters):
        phase = it / float(n_iters)

        # forward
        sigma_raw, rgb_rec = recon_vol()   # sigma_raw: [1,1,D,H,W], already softplus

        if use_neighbor_growth:
            # ===== NEIGHBOR GROWTH APPROACH =====
            # Compute sum of density in neighborhood of each voxel
            neighbor_sum = compute_neighbor_density_sum(sigma_raw, kernel_size=neighbor_kernel_size)
            
            # Only allow gradients where neighbor density is sufficient
            # This makes density "grow" from the seed
            growth_mask = (neighbor_sum >= neighbor_threshold).float()
            
            # Apply growth constraint
            sigma_rec = sigma_raw * growth_mask + bg_sigma * (1.0 - growth_mask)
            
            active_voxels = growth_mask.sum().item()
            max_neighbor_sum = neighbor_sum.max().item()
            mean_sigma_in_mask = (sigma_raw * growth_mask).sum().item() / (active_voxels + 1e-8)
            info_str = f"active_voxels={int(active_voxels)}, max_nbr_sum={max_neighbor_sum:.2f}, mean_σ_in_mask={mean_sigma_in_mask:.3f}"
            
            
        else:
            # ===== HARD CORE APPROACH =====
            # Expanding core radius over time
            r_core = r_core_min + (r_core_max - r_core_min) * phase
            
            # build core mask
            core_mask = (dist_vol <= r_core).float()         # [1,1,D,H,W]
            outer_mask = 1.0 - core_mask

            # apply hard constraint:
            # - inside: trainable sigma_raw
            # - outside: fixed bg_sigma (no gradients there)
            sigma_rec = sigma_raw * core_mask + bg_sigma * outer_mask
            
            info_str = f"r_core={r_core:.3f}"

        pred_images = render_volume(
            sigma_rec, rgb_rec, K, poses,
            img_size=img_res,
            n_samples=n_samples,
            scene_radius=scene_radius,
            device=device,
        )  # list of V [3,H,W]

        pred_stack = torch.stack(pred_images, dim=0)

        # data term - masked MSE (foreground only)
        mask = mask_stack  # [V,1,H,W] in {0,1}
        diff2 = (pred_stack - gt_stack) ** 2 * mask
        denom = mask.sum() * pred_stack.shape[1] + 1e-6  # *channels
        loss_mse = diff2.sum() / denom

        # NO REGULARIZATION - test if photometric gradients exist at all
        loss = loss_mse

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 10 == 0 or it == 0:
            print(
                f"[{it}/{n_iters}] loss={loss.item():.6e} "
                f"(mse={loss_mse.item():.6e}, {info_str})"
            )

        # Update live viewer
        if viewer_state is not None and (it % 5 == 0 or it == 0):
            # Handle pygame events (non-blocking)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    viewer_state = None
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        viewer_state = None
                        break
                    elif event.key == pygame.K_SPACE:
                        viewer_state['auto_rotate'] = not viewer_state['auto_rotate']
                    elif event.key == pygame.K_v:
                        viewer_state['mode'] = 'volume'
                        pygame.display.set_caption("Voxel Training (mode: VOLUME)")
                    elif event.key == pygame.K_c:
                        viewer_state['mode'] = 'cubes'
                        pygame.display.set_caption("Voxel Training (mode: CUBES)")
            
            if viewer_state is not None:
                # Handle continuous key presses for camera control
                keys = pygame.key.get_pressed()
                orbit_speed = 0.05
                zoom_speed = 0.1
                
                if keys[pygame.K_LEFT]:
                    viewer_state['angle_y'] -= orbit_speed
                if keys[pygame.K_RIGHT]:
                    viewer_state['angle_y'] += orbit_speed
                if keys[pygame.K_UP]:
                    viewer_state['angle_x'] = max(viewer_state['angle_x'] - orbit_speed, -math.pi / 2 + 0.1)
                if keys[pygame.K_DOWN]:
                    viewer_state['angle_x'] = min(viewer_state['angle_x'] + orbit_speed, math.pi / 2 - 0.1)
                if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
                    viewer_state['radius'] = max(0.5, viewer_state['radius'] - zoom_speed)
                if keys[pygame.K_MINUS]:
                    viewer_state['radius'] += zoom_speed
                
                # Auto-rotate if enabled
                if viewer_state['auto_rotate']:
                    viewer_state['angle_y'] += 0.01
                
                # Render current voxels based on mode
                if viewer_state['mode'] == 'volume':
                    # Volumetric rendering
                    cx = viewer_state['radius'] * math.cos(viewer_state['angle_x']) * math.cos(viewer_state['angle_y'])
                    cy = viewer_state['radius'] * math.sin(viewer_state['angle_x'])
                    cz = viewer_state['radius'] * math.cos(viewer_state['angle_x']) * math.sin(viewer_state['angle_y'])
                    eye = np.array([cx, cy, cz], dtype=np.float32)
                    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    
                    # Build camera pose
                    K = make_intrinsics(256, 256, fov_y_deg=fov_y_deg, device=device)
                    
                    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    forward = target - eye
                    forward /= np.linalg.norm(forward) + 1e-8
                    right = np.cross(forward, up)
                    right /= np.linalg.norm(right) + 1e-8
                    true_up = np.cross(right, forward)
                    R = np.stack([right, true_up, forward], axis=0)
                    t = -R @ eye
                    R_t = torch.from_numpy(R).to(device)
                    t_t = torch.from_numpy(t).to(device).view(3, 1)
                    
                    # Generate rays and render
                    with torch.no_grad():
                        pts = generate_rays(256, 256, K, R_t, t_t, n_samples=n_samples, 
                                          near=0.5, far=2.5, device=device)
                        sigma_s, rgb_s = sample_volume(sigma_rec, rgb_rec, pts, scene_radius=scene_radius)
                        img_t = volume_render(sigma_s, rgb_s, n_samples)
                        img_np = img_t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                        img = (img_np * 255).astype(np.uint8)
                else:
                    # Cube rendering
                    sigma_np = sigma_rec[0, 0].detach().cpu().numpy()
                    rgb_np = rgb_rec[0].detach().cpu().numpy().transpose(1, 2, 3, 0)  # [D,H,W,3]
                    
                    img = render_voxels_pygame(
                        sigma_np, rgb_np,
                        viewer_state['angle_y'],
                        viewer_state['angle_x'],
                        viewer_state['radius'],
                        img_size=(256, 256),
                        scene_radius=scene_radius,
                        thresh_factor=0.2
                    )
                
                # Display
                surf = pygame.surfarray.make_surface(np.rot90(img, k=1))
                surf = pygame.transform.smoothscale(surf, (viewer_state['win_size'], viewer_state['win_size']))
                viewer_state['screen'].blit(surf, (0, 0))
                pygame.display.flip()
                viewer_state['clock'].tick(30)

    # Clean up viewer
    if viewer_state is not None:
        pygame.quit()
        print("[VIEWER] Closed")

    # 6) Save recon views and voxel cloud
    sigma_raw, rgb_rec = recon_vol()
    
    # Apply final constraint based on mode
    if use_neighbor_growth:
        neighbor_sum = compute_neighbor_density_sum(sigma_raw, kernel_size=neighbor_kernel_size)
        growth_mask = (neighbor_sum >= neighbor_threshold).float()
        sigma_rec = sigma_raw * growth_mask + bg_sigma * (1.0 - growth_mask)
    else:
        core_mask = (dist_vol <= r_core_max).float()
        outer_mask = 1.0 - core_mask
        sigma_rec = sigma_raw * core_mask + bg_sigma * outer_mask
    
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

    # Export voxels as PLY with RGBA
    # sigma_rec: [1,1,D,H,W], rgb_rec: [1,3,D,H,W]
    sigma_np = sigma_rec[0, 0].detach().cpu().numpy()              # [D,H,W]
    rgb_np   = rgb_rec[0].detach().cpu().numpy()                   # [3,D,H,W]
    rgb_np   = np.moveaxis(rgb_np, 0, -1)                          # [D,H,W,3]

    np.savez(out_dir / "recon_volume.npz", sigma=sigma_np, rgb=rgb_np)

    export_voxels_as_ply_rgba(
        sigma_np,
        rgb_np,
        out_dir / "recon_voxels_rgba.ply",
    )

    print("Done. Check:", out_dir)


# ----------------- PLY export with alpha -----------------

def export_voxels_as_ply_rgba(sigma, rgb, out_path, thresh=0.5):
    """
    sigma: [D,H,W]       density
    rgb:   [D,H,W,3]     in [0,1]
    Writes a vertex PLY with RGBA (alpha from normalized sigma).
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

    sigma_norm = sigma / (sigma.max() + 1e-8)

    verts = []
    colors = []

    for z_i, y_i, x_i in idxs:
        x = xs[x_i]
        y = ys[y_i]
        z = zs[z_i]
        verts.append((x, y, z))

        c_rgb = (rgb[z_i, y_i, x_i] * 255.0).astype(np.uint8)
        a     = int(np.clip(sigma_norm[z_i, y_i, x_i] * 255.0, 0, 255))
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


# ----------------- CLI -----------------

if __name__ == "__main__":
    import sys

    args = sys.argv

    if len(args) >= 3:
        # Auto-estimate period + direction:
        # python video_orbit_voxel_recon.py <video_path> <start_frame>
        video_path = args[1]
        start_frame = int(args[2])
        orbit_period_frames, fps = estimate_orbit_period(
            video_path, start_frame=start_frame
        )
        direction = estimate_orbit_direction(
            video_path, start_frame=start_frame
        )
    else:
        print("Usage (auto):   python video_orbit_voxel_recon.py <video_path> <start_frame> [--neighbor-growth] [--viewer]")
        print("Usage (manual): python video_orbit_voxel_recon.py <video_path> <orbit_period_frames> <direction(+1|-1)> <start_frame> [--neighbor-growth] [--viewer]")
        print()
        print("Flags:")
        print("  --neighbor-growth  Use neighbor-based growth from center seed (default: hard core)")
        print("  --viewer           Enable live pygame viewer during training")
        sys.exit(1)

    # Check for optional flags
    use_neighbor_growth = "--neighbor-growth" in sys.argv
    enable_viewer = "--viewer" in sys.argv
    
    if use_neighbor_growth:
        print("=" * 60)
        print("USING NEIGHBOR GROWTH MODE (organic growth from center seed)")
        print("=" * 60)
    else:
        print("=" * 60)
        print("USING HARD CORE MODE (expanding radial constraint)")
        print("=" * 60)
    
    if enable_viewer:
        print("=" * 60)
        print("LIVE VIEWER ENABLED")
        print("  C/V=cubes/volume, arrows=orbit, +/-=zoom, SPACE=pause rotation")
        print("=" * 60)
    
    train_from_video(
        video_path=video_path,
        orbit_period_frames=orbit_period_frames,
        direction=direction,
        start_frame=start_frame,
        frame_step=4,          # tweak as you like
        grid_size=32,
        img_res=(64,64),
        n_samples=64,
        n_iters=2000,          # increased for neighbor growth to have time to expand
        scene_radius=1.5,
        fov_y_deg=45.0,
        out_dir="video_voxel_out",
        use_neighbor_growth=use_neighbor_growth,
        enable_viewer=enable_viewer,
    )

