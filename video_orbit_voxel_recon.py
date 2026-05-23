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
import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch import nn, optim

# Sharded volume for multi-GPU model parallelism
from sharded_voxel_volume import ShardedVoxelVolume

# Optional visualization imports
try:
    import pygame
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("[WARN] pygame not available, real-time visualization disabled")

# Optional OpenAI imports for background removal
try:
    from openai import OpenAI
    from PIL import Image
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


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
                  near=0.5, far=2.5, device="cpu", K_inv=None):
    """
    Generate 3D sample points along rays for a single camera.
    Returns pts_world: [1, S, H, W, 3]

    If K_inv is provided, it will be used directly to avoid repeatedly
    calling torch.inverse(K) (important for multi-GPU / lazy tensors).
    """
    ys, xs = torch.meshgrid(
        torch.linspace(0, h - 1, h, device=device),
        torch.linspace(0, w - 1, w, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=-1)  # HxWx3

    # Only compute inverse if not provided
    if K_inv is None:
        if K.device != torch.device(device):
            K = K.to(device)
        K_inv_local = torch.inverse(K)
    else:
        K_inv_local = K_inv.to(device)

    dirs_cam = (K_inv_local @ pix.reshape(-1, 3).T).T  # (H*W)x3
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


def sample_volume_sharded(sharded_vol, pts_world, scene_radius=1.5):
    """
    Sample from sharded volume (multi-GPU).
    
    sharded_vol: ShardedVoxelVolume instance
    pts_world: [1,S,H,W,3] points in world space
    Returns:
        sigma_samples: [1,S,H,W]
        rgb_samples:   [1,S,H,W,3]
    """
    _, S, H, W, _ = pts_world.shape
    
    # Flatten points for sampling
    pts_flat = pts_world.reshape(-1, 3)  # [S*H*W, 3]
    
    # Sample from sharded volume
    sigma_flat, rgb_flat = sharded_vol.sample_volume(pts_flat, scene_radius)
    
    # Reshape back
    sigma_samples = sigma_flat.reshape(1, S, H, W)
    rgb_samples = rgb_flat.reshape(1, S, H, W, 3)
    
    return sigma_samples, rgb_samples


def volume_render(sigma_samples, rgb_samples, n_samples,
                  return_depth=False, near=0.1, far=5.0):
    """
    NeRF-style compositing.
    sigma_samples: [1,S,H,W]
    rgb_samples:   [1,S,H,W,3]
    Returns:
      rgb_out: [1,3,H,W]
      (optional) depth_out: [1,H,W] expected ray distance
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
    if not return_depth:
        return rgb_out
    S = sigma_samples.shape[1]
    ts = torch.linspace(near, far, S, device=sigma_samples.device)
    ts = ts.view(1, S, 1, 1)
    depth_out = (weights * ts).sum(dim=1)                      # [1,H,W]
    return rgb_out, depth_out


def render_volume(sigma, rgb, K, poses, img_size=(64, 64),
                  n_samples=64, scene_radius=1.5, device="cpu"):
    """Render volume using regular (non-sharded) volume."""
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


def render_volume_with_depth(sigma, rgb, K, poses, img_size=(64, 64),
                             n_samples=64, scene_radius=1.5, device="cpu",
                             near=0.1, far=5.0):
    """Render volume returning both RGB and expected ray-depth per view."""
    H, W = img_size
    images, depths = [], []
    for (R, t) in poses:
        pts = generate_rays(H, W, K, R, t, n_samples=n_samples,
                            near=near, far=far, device=device)
        sigma_s, rgb_s = sample_volume(sigma, rgb, pts, scene_radius=scene_radius)
        rgb_img, depth_img = volume_render(sigma_s, rgb_s, n_samples,
                                           return_depth=True,
                                           near=near, far=far)
        images.append(rgb_img[0])
        depths.append(depth_img[0])
    return images, depths


def render_volume_sharded(sharded_vol, K, poses, img_size=(64, 64),
                          n_samples=64, scene_radius=1.5, device="cuda:0"):
    """Render volume using sharded volume (multi-GPU model parallelism)."""
    H, W = img_size
    images = []
    for (R, t) in poses:
        pts = generate_rays(H, W, K, R, t,
                            n_samples=n_samples,
                            near=0.1, far=5.0,
                            device=device)
        sigma_s, rgb_s = sample_volume_sharded(sharded_vol, pts, scene_radius=scene_radius)
        rgb_img = volume_render(sigma_s, rgb_s, n_samples)  # [1,3,H,W]
        images.append(rgb_img[0])
    return images


def render_volume_multigpu(sigma, rgb, K, poses, img_size=(64, 64),
                            n_samples=64, scene_radius=1.5, n_gpus=1):
    """
    Multi-GPU version: splits poses across GPUs, each GPU renders its subset.
    Returns list of images on GPU 0.
    
    OPTIMIZED: Uses CUDA streams for async parallel execution and computes
    K_inv only once per GPU to avoid repeated inversion of a lazy tensor.
    """
    if n_gpus <= 1:
        return render_volume(sigma, rgb, K, poses, img_size, n_samples, scene_radius, "cuda")
    
    H, W = img_size
    n_views = len(poses)
    
    # Split poses across GPUs
    views_per_gpu = (n_views + n_gpus - 1) // n_gpus
    
    def render_on_gpu(gpu_id, pose_subset):
        """Render a subset of views on a specific GPU"""
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        # Create a CUDA stream for this GPU
        stream = torch.cuda.Stream(device=device)
        
        with torch.cuda.stream(stream):
            # Copy volume and intrinsics to this GPU (non-blocking)
            sigma_gpu = sigma.to(device, non_blocking=True)
            rgb_gpu = rgb.to(device, non_blocking=True)
            K_gpu = K.to(device, non_blocking=True)

            # IMPORTANT: only invert once per GPU (fixes lazy wrapper error)
            K_inv_gpu = torch.inverse(K_gpu)

            images_gpu = []
            for (R, t) in pose_subset:
                R_gpu = R.to(device, non_blocking=True)
                t_gpu = t.to(device, non_blocking=True)
                pts = generate_rays(
                    H, W,
                    K_gpu, R_gpu, t_gpu,
                    n_samples=n_samples,
                    near=0.1, far=5.0,
                    device=device,
                    K_inv=K_inv_gpu,  # reuse precomputed inverse
                )
                sigma_s, rgb_s = sample_volume(sigma_gpu, rgb_gpu, pts, scene_radius=scene_radius)
                rgb_img = volume_render(sigma_s, rgb_s, n_samples)
                images_gpu.append(rgb_img[0])
            
            # Synchronize this stream before returning
            stream.synchronize()
        
        return images_gpu
    
    # Launch rendering on each GPU using threads
    from concurrent.futures import ThreadPoolExecutor
    
    all_images = []
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = []
        for gpu_id in range(n_gpus):
            start_idx = gpu_id * views_per_gpu
            end_idx = min(start_idx + views_per_gpu, n_views)
            if start_idx >= n_views:
                break
            pose_subset = poses[start_idx:end_idx]
            futures.append(executor.submit(render_on_gpu, gpu_id, pose_subset))
        
        # Gather results and move to GPU 0
        for future in futures:
            images_subset = future.result()
            # Move to GPU 0 (non-blocking)
            all_images.extend([img.to("cuda:0", non_blocking=True) for img in images_subset])
    
    # Final sync on GPU 0
    torch.cuda.synchronize("cuda:0")
    
    return all_images


# ----------------- OpenAI Background Removal -----------------

def remove_background_openai(frame_rgb, api_key=None):
    """
    Use OpenAI's GPT-4o Vision to intelligently create a mask for background removal.
    PRESERVES the original image - only generates an improved alpha mask.
    
    Args:
        frame_rgb: numpy array (H, W, 3) in RGB format, uint8
        api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
    
    Returns:
        frame_rgba: numpy array (H, W, 4) in RGBA format, uint8
    """
    if not OPENAI_AVAILABLE:
        print("[WARN] OpenAI not available, skipping background removal")
        # Return original with full alpha
        h, w = frame_rgb.shape[:2]
        alpha = np.ones((h, w, 1), dtype=np.uint8) * 255
        return np.concatenate([frame_rgb, alpha], axis=2)
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(frame_rgb)
        
        # Save to bytes buffer as PNG
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Use GPT-4o Vision to identify what should be kept vs removed
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        vision_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this image and identify the main subject/object versus the background.

I need to create a mask to remove the background. The background includes:
- White or light-colored areas
- Isolated "islands" of white pixels not connected to the main object
- Any areas that are clearly not part of the central object

Please describe:
1. Where is the main object located? (provide approximate pixel coordinates or percentage from edges)
2. What color range does the main object have? (to distinguish from background)
3. Are there isolated white regions that should be removed?
4. What's the approximate bounding box of the object? (top, left, bottom, right as percentages)

Be specific with coordinates and color ranges."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}",
                                "detail": "high"  # Use high detail for accurate mask
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        mask_description = vision_response.choices[0].message.content
        
        # Parse the description to create an improved mask
        # For now, we'll use a heuristic approach based on the description
        # Future: Could use GPT to generate actual pixel coordinates
        
        h, w = frame_rgb.shape[:2]
        
        # Create mask based on color and position analysis
        # Start with a simple color-based approach
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        
        # Check if AI mentioned specific white threshold or bright areas to remove
        if "white" in mask_description.lower() or "bright" in mask_description.lower():
            # More aggressive white removal based on AI feedback
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        else:
            # Standard approach
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Remove small isolated components (the "white islands")
        # Use morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours and keep only the largest connected component
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Keep the largest contour (main object)
            largest_contour = max(contours, key=cv2.contourArea)
            mask_clean = np.zeros_like(mask)
            cv2.drawContours(mask_clean, [largest_contour], -1, 255, -1)
            mask = mask_clean
        
        # Smooth the mask edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Combine original RGB with AI-improved alpha mask
        frame_rgba = np.dstack([frame_rgb, mask])
        
        return frame_rgba
        
    except Exception as e:
        print(f"[ERROR] OpenAI background removal failed: {e}")
        print("[INFO] Falling back to original frame")
        # Return original with full alpha
        h, w = frame_rgb.shape[:2]
        alpha = np.ones((h, w, 1), dtype=np.uint8) * 255
        return np.concatenate([frame_rgb, alpha], axis=2)


def remove_backgrounds_parallel(frames_rgb, api_key=None, max_workers=4):
    """
    Remove backgrounds from multiple frames in parallel using OpenAI.
    Uses GPT-4o Vision to intelligently improve masks while PRESERVING original images.
    
    Args:
        frames_rgb: list of numpy arrays (H, W, 3) in RGB format, uint8
        api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
        max_workers: number of parallel workers
    
    Returns:
        frames_rgba: list of numpy arrays (H, W, 4) in RGBA format, uint8
    """
    print(f"[OpenAI] Removing backgrounds from {len(frames_rgb)} frames in parallel...")
    print(f"[OpenAI] Using GPT-4o Vision to create improved masks")
    print(f"[OpenAI] Original images are preserved (only alpha mask is generated)")
    print(f"[OpenAI] This may take a minute or two and will incur API costs (~$0.01/frame)...")
    
    frames_rgba = [None] * len(frames_rgb)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(remove_background_openai, frame, api_key): idx
            for idx, frame in enumerate(frames_rgb)
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                frames_rgba[idx] = future.result()
                completed += 1
                print(f"[OpenAI] Processed {completed}/{len(frames_rgb)} frames...")
            except Exception as e:
                print(f"[ERROR] Frame {idx} failed: {e}")
                # Fallback: use original with full alpha
                h, w = frames_rgb[idx].shape[:2]
                alpha = np.ones((h, w, 1), dtype=np.uint8) * 255
                frames_rgba[idx] = np.concatenate([frames_rgb[idx], alpha], axis=2)
    
    print(f"[OpenAI] Background removal complete!")
    return frames_rgba


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


def load_frames_as_tensors(video_path, frame_indices, img_res=(64, 64), device="cpu", use_openai_bg_removal=False, openai_api_key=None):
    """
    Load specific frames from a video, resize, return:
      gt_stack: [V,3,H,W] in [0,1]
      mask_stack: [V,1,H,W] in {0,1}  (1 = foreground)
      used_indices: list[int]
      
    If use_openai_bg_removal=True, uses OpenAI DALL-E to remove backgrounds before processing.
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

    # --- Load all requested frames first ---
    Ht, Wt = img_res
    raw_frames_rgb = []  # Store raw RGB frames before processing
    wanted = set(valid_indices)

    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if idx in wanted:
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw_frames_rgb.append(frame_rgb)

            if len(raw_frames_rgb) == len(valid_indices):
                break

        idx += 1

    cap.release()

    if len(raw_frames_rgb) != len(valid_indices):
        raise RuntimeError(
            f"Requested {len(valid_indices)} valid frames, "
            f"but only read {len(raw_frames_rgb)} from the stream."
        )

    # --- Apply OpenAI background removal if requested ---
    if use_openai_bg_removal:
        print("=" * 60)
        print("OPENAI BACKGROUND REMOVAL ENABLED")
        print("Using GPT-4o Vision to create intelligent masks")
        print("Original images preserved - only alpha masks are AI-generated")
        print("=" * 60)
        frames_rgba = remove_backgrounds_parallel(raw_frames_rgb, api_key=openai_api_key, max_workers=100)
        
        # Process RGBA frames (with alpha channel from OpenAI)
        frames = []
        masks  = []
        for frame_rgba in frames_rgba:
            # Resize to training resolution
            frame_rgba_resized = cv2.resize(frame_rgba, (Wt, Ht), interpolation=cv2.INTER_AREA)
            
            # Split RGB and alpha
            frame_rgb = frame_rgba_resized[:, :, :3]
            alpha = frame_rgba_resized[:, :, 3]
            
            # Convert to torch tensors
            frame_f = torch.from_numpy(frame_rgb).float() / 255.0  # HxWx3
            frame_f = frame_f.permute(2, 0, 1)                      # 3xHxW
            
            mask_f = torch.from_numpy(alpha.astype(np.float32) / 255.0)  # HxW in [0,1]
            mask_f = mask_f.unsqueeze(0)  # 1xHxW
            
            frames.append(frame_f)
            masks.append(mask_f)
    else:
        # --- Traditional background subtraction ---
        # Re-open video to estimate background
        bg_bgr = estimate_background_frame(video_path)
        
        cap = cv2.VideoCapture(video_path)
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
    frame_step=1,
    grid_size=32,
    img_res=(64,64),
    n_samples=64,
    n_iters=500,
    scene_radius=1.5,
    fov_y_deg=45.0,
    out_dir="video_voxel_out",
    use_neighbor_growth=False,
    enable_viewer=False,
    use_sharded=False,
    lambda_l1=0.03,          # L1 sparsity weight
    lambda_tv_sigma=0.002,   # TV smoothness weight for density
    lambda_tv_rgb=0.001,     # TV smoothness weight for color
    use_openai_bg_removal=False,  # Use OpenAI to remove backgrounds
    openai_api_key=None,     # OpenAI API key (or use OPENAI_API_KEY env var)
    checkpoint_npz=None,     # Optional: path to checkpoint NPZ to resume from
    use_depth_anything=False,
    lambda_depth=0.5,
    lambda_freespace=0.5,
    depth_anything_model="depth-anything/Depth-Anything-V2-Small-hf",
    reference_height=None,
):

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = "cuda:0" if n_gpus > 0 else "cpu"
    
    print("=" * 60)
    print(f"Primary device: {device}")
    if n_gpus > 0:
        print(f"Available GPUs: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        print(f"CUDA version: {torch.version.cuda}")
        if n_gpus > 1:
            # Decide whether to use multi-GPU rendering based on resolution
            H, W = img_res
            n_pixels = H * W
            if not use_sharded and n_pixels >= 16384:  # 128x128 or higher
                print(f"MULTI-GPU RENDERING ENABLED")
                print(f"  Frame resolution: {H}×{W} = {n_pixels:,} pixels")
                print(f"  Will parallelize rendering across {n_gpus} GPUs")
                print(f"  Expected speedup: ~{n_gpus//2}-{n_gpus}× (rendering dominates volume copy)")
            else:
                if use_sharded:
                    print(f"NOTE: Multi-GPU data parallelism disabled in sharded mode")
                    print(f"  Sharded mode already uses all GPUs for model parallelism")
                else:
                    print(f"NOTE: Multi-GPU rendering disabled (frame resolution too low)")
                    print(f"  Resolution: {H}×{W} = {n_pixels} pixels")
                    print(f"  Multi-GPU helps with: 128×128+ resolution")
    print("=" * 60)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1) Choose which frames to use as views
    # Sample frames evenly throughout the orbit using frame_step
    frame_indices = list(range(start_frame, start_frame + orbit_period_frames, frame_step))
    print("Requested frames:", frame_indices)

    # 2) Load frames as gt images
    gt_stack, mask_stack, used_indices = load_frames_as_tensors(
        video_path, frame_indices,
        img_res=img_res, device=device,
        use_openai_bg_removal=use_openai_bg_removal,
        openai_api_key=openai_api_key
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

    # 3.5) DepthAnything supervision setup
    da_targets = None         # tensor [V, H, W] of per-pixel target depth in world units
    da_masks = None           # tensor [V, H, W] in {0, 1}
    free_space_mask_3d = None # [1, 1, D, H, W] of voxels that should be empty
    if use_depth_anything:
        from depth_anything import DepthAnythingEstimator
        print("[DepthAnything] Loading model...")
        da = DepthAnythingEstimator(model_id=depth_anything_model)
        # Compute per-view depth in pipeline units (1 = scene centre).
        # Convert to world depth from camera using a fixed orbit radius
        # (same convention as build_orbit_poses_from_period).
        cam_radius = 2.5
        # Re-read frames at full resolution to feed DepthAnything, then
        # resize the resulting depth map to (H, W).
        # Use the trainer's foreground mask (already loaded above) so we
        # never feed DepthAnything's nonsense outputs on the white
        # background into the loss.
        fg_mask_np = mask_stack.squeeze(1).cpu().numpy()  # [V, H, W] in {0,1}

        # Per-frame orbit camera distance: if a reference height is
        # supplied, recover it via pinhole geometry from the foreground
        # bbox height in pixels.  Otherwise fall back to the orbit
        # radius assumed by build_orbit_poses_from_period.
        focal_y_px = float(K[1, 1].cpu().numpy())
        per_frame_cam_radius = []
        if reference_height is not None:
            for vi in range(len(used_indices)):
                fg = fg_mask_np[vi] > 0.5
                ys, xs = np.nonzero(fg)
                if len(ys) < 10:
                    per_frame_cam_radius.append(2.5)  # fallback
                    continue
                bbox_h_px = float(ys.max() - ys.min())
                if bbox_h_px <= 0:
                    per_frame_cam_radius.append(2.5)
                    continue
                # Pinhole: bbox_h_px / focal_y_px = reference_height / depth
                d = focal_y_px * reference_height / bbox_h_px
                per_frame_cam_radius.append(float(d))
            mean_d = float(np.mean(per_frame_cam_radius))
            print(f"[DepthAnything] Reference-height calibration: "
                  f"mean recovered cam-distance = {mean_d:.3f} m "
                  f"(per-frame std = {float(np.std(per_frame_cam_radius)):.3f})")
            cam_radius = mean_d  # also used for the freespace mask
        else:
            per_frame_cam_radius = [2.5] * len(used_indices)

        cap = cv2.VideoCapture(video_path)
        depth_targets_np = []
        depth_masks_np = []
        for vi, fi in enumerate(used_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame_bgr = cap.read()
            if not ok:
                depth_targets_np.append(np.ones((H, W), dtype=np.float32) * cam_radius)
                depth_masks_np.append(np.zeros((H, W), dtype=np.float32))
                continue
            d_rel = da.estimate(frame_bgr)                          # (h, w)
            d_rel_resized = cv2.resize(d_rel, (W, H), interpolation=cv2.INTER_LINEAR)

            # Renormalise using only foreground pixels so the white
            # background outliers don't blow up the depth map's range.
            fg = fg_mask_np[vi] > 0.5
            if fg.sum() > 0:
                lo, hi = np.percentile(d_rel_resized[fg], [2, 98])
                d_rel_resized = np.clip(d_rel_resized, lo, hi)
                # Re-anchor to "1.0 = median of foreground"
                med = float(np.median(d_rel_resized[fg]))
                if med > 1e-6:
                    d_rel_resized = d_rel_resized / med

            # Per-frame camera distance from reference-height calibration
            # (or the fallback orbit radius if not supplied)
            cam_d = per_frame_cam_radius[vi]
            d_world = cam_d - (d_rel_resized - 1.0) * cam_d

            # Mask out non-foreground AND any extreme depth values
            valid = fg & (d_rel_resized > 0.3) & (d_rel_resized < 2.5)
            depth_targets_np.append(d_world.astype(np.float32))
            depth_masks_np.append(valid.astype(np.float32))
        cap.release()
        da_targets = torch.from_numpy(np.stack(depth_targets_np)).to(device)  # [V, H, W]
        da_masks = torch.from_numpy(np.stack(depth_masks_np)).to(device)
        n_valid = int(da_masks.sum().item())
        n_total = da_masks.numel()
        print(f"[DepthAnything] Computed {len(used_indices)} depth maps "
              f"(world-distance range [{da_targets.min().item():.2f}, {da_targets.max().item():.2f}] m, "
              f"valid pixels: {n_valid}/{n_total} = {100.0*n_valid/n_total:.1f}%)")

        # Build a global 3D free-space mask: voxels that are CLOSER to
        # any view's camera than that view's depth target says the
        # surface is.
        if lambda_freespace > 0:
            margin = 0.05
            gs = grid_size
            coords = np.linspace(-scene_radius, scene_radius, gs).astype(np.float32)
            zw, yw, xw = np.meshgrid(coords, coords, coords, indexing='ij')
            pts_world = np.stack([xw, yw, zw], axis=-1).reshape(-1, 3).T  # [3, N]
            free_acc = np.zeros(gs * gs * gs, dtype=np.float32)
            K_np = K.cpu().numpy()
            for vi, (Rt, tt) in enumerate(poses):
                R_np = Rt.cpu().numpy()
                t_np = tt.cpu().numpy().reshape(3)
                pts_cam = R_np @ pts_world + t_np[:, None]   # [3, N]
                cam_z = pts_cam[2]
                in_front = cam_z > 0.05
                pix = K_np @ pts_cam
                u = pix[0] / np.maximum(pix[2], 1e-6)
                v = pix[1] / np.maximum(pix[2], 1e-6)
                ui = np.clip(np.round(u).astype(np.int64), 0, W - 1)
                vi_p = np.clip(np.round(v).astype(np.int64), 0, H - 1)
                d_target = depth_targets_np[vi][vi_p, ui]
                # d_mask combines (a) the trainer's foreground mask
                # and (b) sane-depth-range filter, so background-pixel
                # projections are skipped entirely.
                d_mask = depth_masks_np[vi][vi_p, ui]
                in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                is_free = (cam_z < (d_target - margin)) & in_front & in_image & (d_mask > 0.5)
                free_acc += is_free.astype(np.float32)
            free_mask_np = (free_acc >= 1.0).reshape(gs, gs, gs).astype(np.float32)
            free_space_mask_3d = (
                torch.from_numpy(free_mask_np)
                .to(device)
                .unsqueeze(0).unsqueeze(0)
            )
            n_free = int(free_mask_np.sum())
            print(f"[DepthAnything] Free-space mask covers "
                  f"{n_free}/{gs**3} voxels ({100.0*n_free/gs**3:.1f}%)")

    # 4) Load checkpoint if provided
    checkpoint_data = None
    if checkpoint_npz is not None:
        checkpoint_path = Path(checkpoint_npz)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_npz}")
        
        print(f"Loading checkpoint from {checkpoint_npz}...")
        checkpoint_data = np.load(checkpoint_npz)
        
        if "sigma" not in checkpoint_data or "rgb" not in checkpoint_data:
            raise ValueError("Checkpoint must contain 'sigma' and 'rgb' arrays")
        
        checkpoint_sigma = checkpoint_data["sigma"]  # [D, H, W]
        checkpoint_rgb = checkpoint_data["rgb"]      # [D, H, W, 3]
        
        # Validate grid size matches
        D_chk, H_chk, W_chk = checkpoint_sigma.shape
        if (D_chk, H_chk, W_chk) != (grid_size, grid_size, grid_size):
            print(f"[WARN] Checkpoint grid size {D_chk}×{H_chk}×{W_chk} doesn't match requested {grid_size}³")
            print(f"[INFO] Using checkpoint grid size: {D_chk}³")
            grid_size = D_chk
        
        print(f"Checkpoint loaded: {D_chk}×{H_chk}×{W_chk}")
        print(f"  Sigma range: [{checkpoint_sigma.min():.4f}, {checkpoint_sigma.max():.4f}]")
        print(f"  RGB range: [{checkpoint_rgb.min():.4f}, {checkpoint_rgb.max():.4f}]")
        
        # Warn if checkpoint has very high densities (might be from hardening)
        if checkpoint_sigma.max() > 50.0:
            print(f"[WARN] Checkpoint has very high density values (max={checkpoint_sigma.max():.1f})")
            print(f"[WARN] This might be from a hardening step. Consider:")
            print(f"[WARN]   - Using the pre-hardened checkpoint instead")
            print(f"[WARN]   - Reducing regularization weights (lambda_l1, lambda_tv_sigma)")
            print(f"[WARN]   - Using lower learning rate")
    
    # 5) Learnable volume
    if use_sharded:
        # Sharded volume across 8 GPUs for massive resolution (e.g., 1024³)
        assert n_gpus == 8, "Sharded mode requires exactly 8 GPUs"
        print(f"[SHARDED MODE] Creating {grid_size}³ volume distributed across {n_gpus} GPUs...")
        recon_vol = ShardedVoxelVolume(grid_size=grid_size, n_gpus=n_gpus, sigma_scale=1.0, init_logit=-5.0)
        # Note: ShardedVoxelVolume handles device placement internally
        opt = optim.Adam(recon_vol.parameters(), lr=5e-2)
    else:
        # Regular single-GPU volume
        recon_vol = VoxelVolume(grid_size=grid_size, sigma_scale=1.0, init_logit=-5.0).to(device)
        
        if checkpoint_data is not None:
            # Initialize from checkpoint
            with torch.no_grad():
                # Load sigma (convert to logits via inverse softplus)
                # softplus(x) = log(1 + exp(x))
                # Inverse softplus: x = log(exp(sigma) - 1)
                # For numerical stability:
                #   - For large sigma (>10): softplus(x) ≈ x, so x ≈ sigma
                #   - For small sigma: use exact formula
                checkpoint_sigma_t = torch.from_numpy(checkpoint_sigma).float()
                checkpoint_sigma_t = torch.clamp(checkpoint_sigma_t, min=1e-6)
                
                # Scale by sigma_scale
                sigma_scaled = checkpoint_sigma_t / recon_vol.sigma_scale
                
                # Numerically stable inverse softplus
                # For sigma > 10: inverse ≈ sigma (since softplus(x) ≈ x for large x)
                # For sigma <= 10: use log(exp(sigma) - 1)
                sigma_logits = torch.where(
                    sigma_scaled > 10.0,
                    sigma_scaled,  # For large values, inverse is approximately identity
                    torch.log(torch.expm1(sigma_scaled) + 1e-8)  # expm1(x) = exp(x) - 1, more stable
                )
                
                recon_vol.density.copy_(sigma_logits.unsqueeze(0).unsqueeze(0))
                
                # Load RGB (convert to logits via inverse sigmoid)
                checkpoint_rgb_t = torch.from_numpy(checkpoint_rgb).float()
                checkpoint_rgb_t = torch.clamp(checkpoint_rgb_t, 1e-7, 1 - 1e-7)
                rgb_logits = torch.logit(checkpoint_rgb_t)
                recon_vol.color.copy_(rgb_logits.permute(3, 0, 1, 2).unsqueeze(0))
            
            print("[CHECKPOINT] Initialized volume from checkpoint")
            print(f"[CHECKPOINT] Density logits range: [{sigma_logits.min():.4f}, {sigma_logits.max():.4f}]")
            
            # Sanity check: forward pass to verify no NaNs
            with torch.no_grad():
                test_sigma, test_rgb = recon_vol()
                if torch.isnan(test_sigma).any() or torch.isnan(test_rgb).any():
                    print("[ERROR] Checkpoint produced NaN values in forward pass!")
                    print(f"  Sigma has NaN: {torch.isnan(test_sigma).any()}")
                    print(f"  RGB has NaN: {torch.isnan(test_rgb).any()}")
                    raise RuntimeError("Checkpoint loading failed: NaN detected")
                print(f"[CHECKPOINT] Forward pass OK: sigma=[{test_sigma.min():.4f}, {test_sigma.max():.4f}], rgb=[{test_rgb.min():.4f}, {test_rgb.max():.4f}]")
        else:
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
        if checkpoint_data is None:
            # Only initialize seed if NOT loading from checkpoint
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
        else:
            print("[NEIGHBOR GROWTH MODE] Using checkpoint density (skipping seed initialization)")
        
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

        # forward (only for non-sharded volumes)
        if not use_sharded:
            sigma_raw, rgb_rec = recon_vol()   # sigma_raw: [1,1,D,H,W], already softplus

        # Apply constraints (neighbor growth or hard core)
        # Note: These are disabled for sharded mode as they require full volume access
        if use_sharded:
            # Sharded mode: no constraints, volume is too large for these operations
            info_str = "sharded"
        elif use_neighbor_growth:
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

        # Render all views
        import time
        t0 = time.time()
        
        if use_sharded:
            # Sharded rendering (model parallelism across 8 GPUs)
            pred_images = render_volume_sharded(
                recon_vol, K, poses,
                img_size=img_res,
                n_samples=n_samples,
                scene_radius=scene_radius,
                device=device,
            )  # list of V [3,H,W]
        elif n_gpus > 1 and img_res[0] * img_res[1] >= 16384:
            # Multi-GPU data-parallel rendering (for high-resolution frames)
            pred_images = render_volume_multigpu(
                sigma_rec, rgb_rec, K, poses,
                img_size=img_res,
                n_samples=n_samples,
                scene_radius=scene_radius,
                n_gpus=n_gpus,
            )  # list of V [3,H,W]
        else:
            # Single-GPU rendering (optionally with depth for DA supervision)
            if use_depth_anything:
                pred_images, pred_depths = render_volume_with_depth(
                    sigma_rec, rgb_rec, K, poses,
                    img_size=img_res,
                    n_samples=n_samples,
                    scene_radius=scene_radius,
                    device=device,
                )
            else:
                pred_images = render_volume(
                    sigma_rec, rgb_rec, K, poses,
                    img_size=img_res,
                    n_samples=n_samples,
                    scene_radius=scene_radius,
                    device=device,
                )  # list of V [3,H,W]
                pred_depths = None

        t_render = time.time() - t0

        pred_stack = torch.stack(pred_images, dim=0)

        # data term - masked MSE (foreground only)
        mask = mask_stack  # [V,1,H,W] in {0,1}
        diff2 = (pred_stack - gt_stack) ** 2 * mask
        denom = mask.sum() * pred_stack.shape[1] + 1e-6  # *channels
        loss_mse = diff2.sum() / denom

        # Regularization weights (passed as parameters)
        
        if use_sharded:
            # For sharded volumes, compute TV and L1 per-shard and aggregate
            loss_tv_sigma = 0.0
            loss_tv_rgb = 0.0
            loss_l1 = 0.0
            for shard in recon_vol.shards:
                with torch.cuda.device(shard.device):
                    sigma_shard, rgb_shard = shard.forward()
                    # Compute losses on shard's device, then move to main device
                    loss_tv_sigma += tv3d(sigma_shard).to(device)
                    loss_tv_rgb += tv3d(rgb_shard).to(device)
                    loss_l1 += sigma_shard.mean().to(device)
            # Average over shards
            loss_tv_sigma /= len(recon_vol.shards)
            loss_tv_rgb /= len(recon_vol.shards)
            loss_l1 /= len(recon_vol.shards)
            loss_tv = lambda_tv_sigma * loss_tv_sigma + lambda_tv_rgb * loss_tv_rgb
        else:
            # Regular volume
            loss_tv_sigma = tv3d(sigma_rec)
            loss_tv_rgb = tv3d(rgb_rec)
            loss_tv = lambda_tv_sigma * loss_tv_sigma + lambda_tv_rgb * loss_tv_rgb
            
            # L1 Sparsity - penalizes total density to encourage empty space
            loss_l1 = sigma_rec.mean()

        loss = loss_mse + loss_tv + lambda_l1 * loss_l1

        # DepthAnything supervision: per-view L1 depth + global free-space penalty
        loss_depth = torch.tensor(0.0, device=device)
        loss_free = torch.tensor(0.0, device=device)
        if use_depth_anything and pred_depths is not None and lambda_depth > 0:
            depth_stack = torch.stack(pred_depths, dim=0)            # [V, H, W]
            depth_diff = (depth_stack - da_targets).abs() * da_masks
            loss_depth = depth_diff.sum() / (da_masks.sum() + 1e-6)
            loss = loss + lambda_depth * loss_depth
        if (use_depth_anything and free_space_mask_3d is not None
                and lambda_freespace > 0):
            loss_free = (sigma_rec * free_space_mask_3d).sum() / (
                free_space_mask_3d.sum() + 1e-6)
            loss = loss + lambda_freespace * loss_free

        t1 = time.time()
        opt.zero_grad()
        loss.backward()
        opt.step()
        t_backward = time.time() - t1

        if it % 10 == 0 or it == 0:
            extras = ""
            if use_depth_anything:
                extras = f", d={loss_depth.item():.3e}, free={loss_free.item():.3e}"
            print(
                f"[{it}/{n_iters}] loss={loss.item():.6e} "
                f"(mse={loss_mse.item():.6e}, tv={loss_tv.item():.6e}, l1={loss_l1.item():.6e}{extras}, {info_str}) "
                f"[render={t_render:.3f}s, backward={t_backward:.3f}s]"
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
                    K_view = make_intrinsics(256, 256, fov_y_deg=fov_y_deg, device=device)
                    
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
                        pts = generate_rays(
                            256, 256, K_view, R_t, t_t,
                            n_samples=n_samples, near=0.5, far=2.5, device=device
                        )
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
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Voxel-based 3D reconstruction from orbital video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detect orbit)
  python video_orbit_voxel_recon.py video.mp4 0
  
  # High-quality reconstruction with more iterations
  python video_orbit_voxel_recon.py video.mp4 0 --n-iters 16000 --img-res 512 512
  
  # Use OpenAI to improve background masks (preserves original images)
  export OPENAI_API_KEY=your_key_here
  python video_orbit_voxel_recon.py video.mp4 0 --openai-bg-removal
  
  # Sharded mode across 8 GPUs
  python video_orbit_voxel_recon.py video.mp4 0 --sharded --grid-size 512
  
  # Fine-tune regularization
  python video_orbit_voxel_recon.py video.mp4 0 --lambda-l1 0.05 --lambda-tv-sigma 0.003
  
  # Use every 2nd frame for faster iteration
  python video_orbit_voxel_recon.py video.mp4 0 --frame-step 2
        """
    )
    
    # Positional arguments
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("start_frame", type=int, help="Starting frame index (0-based)")
    
    # Volume settings
    volume_group = parser.add_argument_group("volume settings")
    volume_group.add_argument("--grid-size", type=int, default=None,
                             help="Voxel grid resolution (default: 128 standard, 512 sharded)")
    volume_group.add_argument("--scene-radius", type=float, default=1.5,
                             help="Scene radius in world units (default: 1.5)")
    
    # Frame/rendering settings
    frame_group = parser.add_argument_group("frame settings")
    frame_group.add_argument("--img-res", type=int, nargs=2, default=[256, 256],
                            metavar=("WIDTH", "HEIGHT"),
                            help="Frame resolution for training (default: 256 256)")
    frame_group.add_argument("--frame-step", type=int, default=1,
                            help="Use every Nth frame (default: 1 = all frames)")
    frame_group.add_argument("--n-samples", type=int, default=64,
                            help="Samples per ray (default: 64)")
    frame_group.add_argument("--fov", type=float, default=45.0,
                            help="Field of view in degrees (default: 45.0)")
    
    # Training settings
    train_group = parser.add_argument_group("training settings")
    train_group.add_argument("--n-iters", type=int, default=8000,
                            help="Number of training iterations (default: 8000)")
    train_group.add_argument("--lambda-l1", type=float, default=0.03,
                            help="L1 sparsity weight (default: 0.03)")
    train_group.add_argument("--lambda-tv-sigma", type=float, default=0.002,
                            help="TV smoothness weight for density (default: 0.002)")
    train_group.add_argument("--lambda-tv-rgb", type=float, default=0.001,
                            help="TV smoothness weight for color (default: 0.001)")
    
    # Mode flags
    mode_group = parser.add_argument_group("reconstruction modes")
    mode_group.add_argument("--neighbor-growth", action="store_true",
                           help="Use neighbor-based growth mode (default: hard core)")
    mode_group.add_argument("--sharded", action="store_true",
                           help="Use sharded volume across 8 GPUs")
    
    # Visualization
    viz_group = parser.add_argument_group("visualization")
    viz_group.add_argument("--viewer", action="store_true",
                          help="Enable live pygame viewer during training")
    
    # Background removal
    bg_group = parser.add_argument_group("background removal")
    bg_group.add_argument("--openai-bg-removal", action="store_true",
                         help="Use OpenAI DALL-E to remove backgrounds and white artifacts (requires OPENAI_API_KEY)")
    bg_group.add_argument("--openai-api-key", type=str, default=None,
                         help="OpenAI API key (default: uses OPENAI_API_KEY environment variable)")
    
    # Checkpoint
    checkpoint_group = parser.add_argument_group("checkpoint")
    checkpoint_group.add_argument("--checkpoint", type=str, default=None,
                                  help="Resume from checkpoint NPZ file")

    # DepthAnything supervision
    da_group = parser.add_argument_group("DepthAnything supervision")
    da_group.add_argument("--use-depth-anything", action="store_true",
                         help="Use monocular DepthAnything depth as a per-view "
                              "supervision signal for the voxel volume")
    da_group.add_argument("--lambda-depth", type=float, default=0.5,
                         help="Weight on the L1 depth loss (default: 0.5)")
    da_group.add_argument("--lambda-freespace", type=float, default=0.5,
                         help="Weight on the global free-space penalty "
                              "(density at voxels in front of the surface "
                              "implied by the depth map). Default: 0.5")
    da_group.add_argument("--depth-anything-model", type=str,
                         default="depth-anything/Depth-Anything-V2-Small-hf",
                         help="HuggingFace DepthAnything checkpoint id")
    da_group.add_argument("--reference-height", type=float, default=None,
                         help="Known world-height of the subject (e.g. 1.5 "
                              "for a humanoid). When set, the orbit camera "
                              "distance is recovered per-frame via "
                              "depth = focal * height / bbox_height_px, "
                              "replacing the hardcoded cam_radius=2.5 fudge.")
    
    # Output
    parser.add_argument("--out-dir", type=str, default="video_voxel_out",
                       help="Output directory (default: video_voxel_out)")
    
    args = parser.parse_args()
    
    # Auto-estimate orbit period and direction
    print(f"Analyzing video: {args.video_path}")
    orbit_period_frames, fps = estimate_orbit_period(
        args.video_path, start_frame=args.start_frame
    )
    direction = estimate_orbit_direction(
        args.video_path, start_frame=args.start_frame
    )
    
    # Print mode info
    if args.neighbor_growth:
        print("=" * 60)
        print("USING NEIGHBOR GROWTH MODE (organic growth from center seed)")
        print("=" * 60)
    else:
        print("=" * 60)
        print("USING HARD CORE MODE (expanding radial constraint)")
        print("=" * 60)
    
    if args.viewer:
        print("=" * 60)
        print("LIVE VIEWER ENABLED")
        print("  C/V=cubes/volume, arrows=orbit, +/-=zoom, SPACE=pause rotation")
        print("=" * 60)
    
    # Determine grid size
    if args.grid_size is not None:
        grid_size = args.grid_size
        print(f"Using custom grid size: {grid_size}³")
    elif args.sharded:
        grid_size = 512  # Conservative default: 256³ per shard across 8 GPUs
        print("=" * 60)
        print(f"SHARDED MODE: Using {grid_size}³ resolution across 8 GPUs")
        shard_size = grid_size // 2  # 2x2x2 = 8 shards
        print(f"  Each shard: {shard_size}³ (~{shard_size**3 * 4 * 4 / 1e9:.2f} GB per shard)")
        print("=" * 60)
    else:
        grid_size = 128   # Standard resolution for single GPU
    
    # Print configuration summary
    print()
    print("Configuration:")
    print(f"  Grid size: {grid_size}³")
    print(f"  Frame resolution: {args.img_res[0]}×{args.img_res[1]}")
    print(f"  Samples per ray: {args.n_samples}")
    print(f"  Iterations: {args.n_iters}")
    frame_suffix = 'st' if args.frame_step == 1 else 'nd' if args.frame_step == 2 else 'rd' if args.frame_step == 3 else 'th'
    print(f"  Frame step: {args.frame_step} (using every {args.frame_step}{frame_suffix} frame)")
    print(f"  Regularization: L1={args.lambda_l1:.4f}, TV_σ={args.lambda_tv_sigma:.4f}, TV_RGB={args.lambda_tv_rgb:.4f}")
    print(f"  Background removal: {'OpenAI DALL-E' if args.openai_bg_removal else 'Traditional (background subtraction)'}")
    print()

    train_from_video(
        video_path=args.video_path,
        orbit_period_frames=orbit_period_frames,
        direction=direction,
        start_frame=args.start_frame,
        frame_step=args.frame_step,
        grid_size=grid_size,
        img_res=tuple(args.img_res),
        n_samples=args.n_samples,
        n_iters=args.n_iters,
        scene_radius=args.scene_radius,
        fov_y_deg=args.fov,
        out_dir=args.out_dir,
        use_neighbor_growth=args.neighbor_growth,
        enable_viewer=args.viewer,
        use_sharded=args.sharded,
        lambda_l1=args.lambda_l1,
        lambda_tv_sigma=args.lambda_tv_sigma,
        lambda_tv_rgb=args.lambda_tv_rgb,
        use_openai_bg_removal=args.openai_bg_removal,
        openai_api_key=args.openai_api_key,
        checkpoint_npz=args.checkpoint,
        use_depth_anything=args.use_depth_anything,
        lambda_depth=args.lambda_depth,
        lambda_freespace=args.lambda_freespace,
        depth_anything_model=args.depth_anything_model,
        reference_height=args.reference_height,
    )
