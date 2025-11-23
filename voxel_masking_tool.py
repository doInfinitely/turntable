# voxel_masking_tool.py
# Interactive tool to mask/unmask voxels by casting rays
#
# Controls:
#   Arrow keys     - orbit (left/right yaw, up/down pitch)
#   +/-            - zoom in/out
#   V              - volumetric mode
#   C              - cube mode
#   Left Click     - MASK voxels along ray (remove them)
#   Right Click    - UNMASK voxels along ray (restore them)
#   U              - Undo last operation
#   Ctrl+S         - Save modified volume
#   ESC / close    - quit

import sys
import math
import numpy as np
import pygame
import torch
import torch.nn.functional as F
from pathlib import Path

# ---------- Volume & camera utilities (Torch) ----------

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


def build_pose_from_eye(eye_np, target_np, device="cpu"):
    """
    Build world->camera rotation R, translation t that look from eye -> target.
    Convention: +Z is forward.
    """
    eye = np.asarray(eye_np, dtype=np.float32)
    target = np.asarray(target_np, dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8

    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8

    true_up = np.cross(right, forward)

    # world->cam
    R = np.stack([right, true_up, forward], axis=0)
    t = -R @ eye

    R_t = torch.from_numpy(R).to(device)
    t_t = torch.from_numpy(t).to(device).view(3, 1)
    return R_t, t_t


def world_to_grid(pts_world, scene_radius=1.5):
    """
    Map world coordinates to [-1,1]^3, same convention as training.
    pts_world: [1,S,H,W,3] or [N,3]
    """
    return pts_world / scene_radius


def cast_ray_through_pixel(px, py, h, w, K, R, t, near=0.1, far=5.0, n_samples=256, device="cpu"):
    """
    Cast a single ray through pixel (px, py) and return sample points.
    
    Args:
        px, py: pixel coordinates (floats)
        h, w: image height, width
        K, R, t: camera intrinsics and extrinsics
        near, far: ray bounds
        n_samples: number of samples along the ray
    
    Returns:
        pts_world: [n_samples, 3] sample points in world coordinates
    """
    # Create pixel coordinate
    pix = torch.tensor([[px, py, 1.0]], dtype=torch.float32, device=device)  # 1x3
    
    # Unproject to camera space
    K_inv = torch.inverse(K)
    dir_cam = (K_inv @ pix.T).T  # 1x3
    dir_cam = dir_cam / torch.norm(dir_cam, dim=-1, keepdim=True)
    
    # Transform to world space
    R = R.to(device)
    t = t.to(device)
    
    dir_world = (R.transpose(0, 1) @ dir_cam.T).T  # 1x3
    C = -(R.transpose(0, 1) @ t).reshape(1, 3)     # 1x3 (camera center in world)
    
    # Sample along ray
    ts = torch.linspace(near, far, n_samples, device=device).view(-1, 1)  # [n_samples, 1]
    pts_world = C + ts * dir_world  # [n_samples, 3]
    
    return pts_world


def sample_volume(sigma, rgb, pts_world, scene_radius=1.5):
    """
    Sample volume at world points (MATCHES VIEWER).
    sigma: [1,1,D,H,W]
    rgb:   [1,3,D,H,W]
    pts_world: [N,3] or [1,S,H,W,3]
    
    Returns:
        sigma_samples: [N] or [1,S,H,W]
        rgb_samples:   [N,3] or [1,S,H,W,3]
    """
    original_shape = pts_world.shape
    is_flat = (len(original_shape) == 2)
    
    if is_flat:
        # [N,3] -> [1,N,1,1,3]
        N = pts_world.shape[0]
        pts_world = pts_world.view(1, N, 1, 1, 3)
    
    pts_grid = world_to_grid(pts_world, scene_radius)
    
    # Match viewer: align_corners=True, padding_mode="zeros"
    sigma_s = F.grid_sample(sigma, pts_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    rgb_s = F.grid_sample(rgb, pts_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    
    if is_flat:
        # [1,1,N,1,1] -> [N]
        sigma_s = sigma_s.view(N)
        # [1,3,N,1,1] -> [N,3]
        rgb_s = rgb_s.view(3, N).T
    else:
        # [1,1,S,H,W] -> [1,S,H,W]
        sigma_s = sigma_s.squeeze(1)
        # [1,3,S,H,W] -> [1,S,H,W,3]
        rgb_s = rgb_s.permute(0, 2, 3, 4, 1)
    
    return sigma_s, rgb_s


def volume_render(sigma_samples, rgb_samples, n_samples):
    """
    NeRF-style compositing along rays (MATCHES VIEWER).
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


def generate_rays_full(h, w, K, R, t, n_samples=64, near=0.1, far=5.0, device="cpu"):
    """
    Generate 3D sample points along rays for all pixels.
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

    pts = C_exp + ts[..., None] * dirs_world  # SxHxWx3 (broadcast)
    pts = pts.unsqueeze(0)                    # [1,S,H,W,3]
    return pts


def mask_voxels_along_ray(sigma, pts_world, scene_radius=1.5, mask_value=0.0):
    """
    Set sigma to mask_value for all voxels along the ray.
    
    Args:
        sigma: [1,1,D,H,W] tensor
        pts_world: [N,3] ray sample points
        scene_radius: scene radius
        mask_value: value to set (0.0 for masking)
    
    Returns:
        modified sigma (in-place modification)
    """
    # Convert world points to grid coordinates [-1, 1]
    pts_grid = world_to_grid(pts_world, scene_radius)  # [N, 3]
    
    # Get volume dimensions
    D, H, W = sigma.shape[2], sigma.shape[3], sigma.shape[4]
    
    # Convert from [-1, 1] to voxel indices
    # grid_sample uses: -1 -> 0, +1 -> size-1
    pts_idx = (pts_grid + 1.0) * 0.5  # [N, 3] in [0, 1]
    pts_idx[:, 0] *= (W - 1)  # x
    pts_idx[:, 1] *= (H - 1)  # y
    pts_idx[:, 2] *= (D - 1)  # z
    
    # Round to nearest voxel
    pts_idx = torch.round(pts_idx).long()
    
    # Clamp to valid range
    pts_idx[:, 0] = torch.clamp(pts_idx[:, 0], 0, W - 1)
    pts_idx[:, 1] = torch.clamp(pts_idx[:, 1], 0, H - 1)
    pts_idx[:, 2] = torch.clamp(pts_idx[:, 2], 0, D - 1)
    
    # Set voxels along ray to mask_value
    for i in range(pts_idx.shape[0]):
        x, y, z = pts_idx[i, 0].item(), pts_idx[i, 1].item(), pts_idx[i, 2].item()
        sigma[0, 0, z, y, x] = mask_value
    
    return sigma


# ---------- Cube rendering (with masking visualization) ----------

def render_view_cubes(sigma_np, rgb_np, R, t, K, img_res=(512, 512),
                     scene_radius=1.5, sigma_thresh=0.5):
    """
    Render voxels as cubes with depth sorting.
    Returns RGB image as numpy array.
    """
    H_img, W_img = img_res
    D, H, W = sigma_np.shape
    
    # Create output image
    img = np.ones((H_img, W_img, 3), dtype=np.uint8) * 30  # Dark background
    depth_buffer = np.full((H_img, W_img), np.inf)
    
    # Get all active voxels
    mask = sigma_np > sigma_thresh
    idxs = np.argwhere(mask)  # [N, 3] - (z, y, x)
    
    if len(idxs) == 0:
        return img
    
    # Voxel centers in world space
    zs = np.linspace(-scene_radius, scene_radius, D)
    ys = np.linspace(-scene_radius, scene_radius, H)
    xs = np.linspace(-scene_radius, scene_radius, W)
    
    # Build list of cubes
    cubes = []
    for z_i, y_i, x_i in idxs:
        pos = np.array([xs[x_i], ys[y_i], zs[z_i]], dtype=np.float32)
        color = (rgb_np[z_i, y_i, x_i] * 255).astype(np.uint8)
        cubes.append((pos, color, sigma_np[z_i, y_i, x_i]))
    
    # Transform to camera space and compute depths
    R_np = R.cpu().numpy()
    t_np = t.cpu().numpy().flatten()
    K_np = K.cpu().numpy()
    
    cubes_with_depth = []
    for pos, color, sigma_val in cubes:
        pos_cam = R_np @ pos + t_np
        depth = pos_cam[2]
        
        if depth <= 0:  # Behind camera
            continue
        
        # Project to image
        pos_img = K_np @ pos_cam
        px = pos_img[0] / pos_img[2]
        py = pos_img[1] / pos_img[2]
        
        if 0 <= px < W_img and 0 <= py < H_img:
            cubes_with_depth.append((px, py, depth, color))
    
    # Sort by depth (far to near)
    cubes_with_depth.sort(key=lambda x: -x[2])
    
    # Render cubes
    cube_size = max(2, min(8, 3000 // len(cubes_with_depth)))  # Adaptive size
    
    for px, py, depth, color in cubes_with_depth:
        px, py = int(px), int(py)
        
        # Draw cube as a small square
        for dy in range(-cube_size, cube_size + 1):
            for dx in range(-cube_size, cube_size + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < W_img and 0 <= ny < H_img:
                    if depth < depth_buffer[ny, nx]:
                        depth_buffer[ny, nx] = depth
                        img[ny, nx] = color
    
    return img


# ---------- Main interactive viewer ----------

def main():
    if len(sys.argv) < 2:
        print("Usage: python voxel_masking_tool.py <recon_volume.npz>")
        sys.exit(1)
    
    npz_path = Path(sys.argv[1])
    if not npz_path.exists():
        print(f"Error: {npz_path} not found")
        sys.exit(1)
    
    # Load volume
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    sigma_np = data["sigma"]  # [D,H,W]
    rgb_np = data["rgb"]      # [D,H,W,3]
    
    print(f"Volume shape: {sigma_np.shape}")
    print(f"Active voxels (σ>0.5): {(sigma_np > 0.5).sum()}")
    print()
    print("Starting in VOLUMETRIC rendering mode (press C for cube mode)")
    print("Controls: Left Click=Mask, Right Click=Unmask, U=Undo, Ctrl+S=Save")
    print()
    
    # Store original for undo
    undo_stack = []
    max_undo = 20
    
    # Convert to torch
    device = "cpu"  # Use CPU for interactive editing
    sigma_t = torch.from_numpy(sigma_np[None, None]).float().to(device)  # [1,1,D,H,W]
    rgb_t = torch.from_numpy(rgb_np).float().permute(3, 0, 1, 2)[None].to(device)  # [1,3,D,H,W]
    
    # Scene parameters
    scene_radius = 1.5
    
    # Initialize pygame
    pygame.init()
    img_res = (800, 800)
    screen = pygame.display.set_mode(img_res)
    pygame.display.set_caption(f"Voxel Masking Tool - {npz_path.name}")
    clock = pygame.time.Clock()
    
    # Camera parameters
    orbit_theta = 0.0  # yaw
    orbit_phi = 0.0    # pitch
    orbit_radius = 3.0
    target = np.array([0.0, 0.0, 0.0])
    
    # Rendering mode
    render_mode = "volume"  # "cubes" or "volume" - default to volumetric for better view
    
    # UI state
    tool_info = "Left Click: MASK | Right Click: UNMASK | U: Undo | Ctrl+S: Save"
    last_operation = ""
    
    # Masking parameters
    ray_samples = 256  # More samples for better coverage
    
    # Volumetric rendering parameters
    volume_n_samples = 128  # Higher quality for viewing
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_v:
                    render_mode = "volume"
                    print("Mode: Volumetric rendering (recommended)")
                elif event.key == pygame.K_c:
                    render_mode = "cubes"
                    print("Mode: Cube rendering (debug view)")
                elif event.key == pygame.K_u:
                    # Undo
                    if undo_stack:
                        sigma_t = undo_stack.pop()
                        last_operation = "Undone"
                        print(f"Undo - stack size: {len(undo_stack)}")
                    else:
                        print("Nothing to undo")
                elif event.key == pygame.K_s and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    # Save
                    save_path = npz_path.parent / f"{npz_path.stem}_masked.npz"
                    sigma_save = sigma_t[0, 0].cpu().numpy()
                    rgb_save = rgb_t[0].cpu().numpy().transpose(1, 2, 3, 0)
                    np.savez(save_path, sigma=sigma_save, rgb=rgb_save)
                    print(f"Saved to {save_path}")
                    last_operation = f"Saved to {save_path.name}"
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get mouse position
                mx, my = pygame.mouse.get_pos()
                
                # Save for undo
                undo_stack.append(sigma_t.clone())
                if len(undo_stack) > max_undo:
                    undo_stack.pop(0)
                
                # Build current camera
                eye = np.array([
                    orbit_radius * math.cos(orbit_phi) * math.cos(orbit_theta),
                    orbit_radius * math.sin(orbit_phi),
                    orbit_radius * math.cos(orbit_phi) * math.sin(orbit_theta)
                ])
                K = make_intrinsics(img_res[0], img_res[1], fov_y_deg=45.0, device=device)
                R, t = build_pose_from_eye(eye, target, device=device)
                
                # Cast ray through clicked pixel
                pts_world = cast_ray_through_pixel(
                    mx, my, img_res[0], img_res[1],
                    K, R, t,
                    near=0.1, far=5.0,
                    n_samples=ray_samples,
                    device=device
                )
                
                if event.button == 1:  # Left click - MASK
                    sigma_t = mask_voxels_along_ray(sigma_t, pts_world, scene_radius, mask_value=0.0)
                    last_operation = f"Masked at ({mx}, {my})"
                    print(f"Masked ray at pixel ({mx}, {my})")
                
                elif event.button == 3:  # Right click - UNMASK
                    # For unmask, we need to restore voxels, but we don't know their original values
                    # For now, set to a default value (e.g., 5.0)
                    sigma_t = mask_voxels_along_ray(sigma_t, pts_world, scene_radius, mask_value=5.0)
                    last_operation = f"Unmasked at ({mx}, {my})"
                    print(f"Unmasked ray at pixel ({mx}, {my})")
        
        # Handle keyboard state for camera control
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            orbit_theta -= 0.05
        if keys[pygame.K_RIGHT]:
            orbit_theta += 0.05
        if keys[pygame.K_UP]:
            orbit_phi = min(orbit_phi + 0.05, math.pi / 2 - 0.1)
        if keys[pygame.K_DOWN]:
            orbit_phi = max(orbit_phi - 0.05, -math.pi / 2 + 0.1)
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            orbit_radius = max(1.0, orbit_radius - 0.1)
        if keys[pygame.K_MINUS]:
            orbit_radius = min(10.0, orbit_radius + 0.1)
        
        # Update camera
        eye = np.array([
            orbit_radius * math.cos(orbit_phi) * math.cos(orbit_theta),
            orbit_radius * math.sin(orbit_phi),
            orbit_radius * math.cos(orbit_phi) * math.sin(orbit_theta)
        ])
        K = make_intrinsics(img_res[0], img_res[1], fov_y_deg=45.0, device=device)
        R, t = build_pose_from_eye(eye, target, device=device)
        
        # Render
        if render_mode == "cubes":
            sigma_render = sigma_t[0, 0].cpu().numpy()
            rgb_render = rgb_t[0].cpu().numpy().transpose(1, 2, 3, 0)
            img_np = render_view_cubes(sigma_render, rgb_render, R, t, K, img_res, scene_radius)
        else:
            # Volumetric rendering (matches viewer exactly)
            pts = generate_rays_full(img_res[0], img_res[1], K, R, t, 
                                    n_samples=volume_n_samples,
                                    near=0.1, far=5.0, device=device)
            
            sigma_s, rgb_s = sample_volume(sigma_t, rgb_t, pts, scene_radius)
            rgb_img = volume_render(sigma_s, rgb_s, volume_n_samples)  # [1,3,H,W]
            
            # Convert to [H,W,3] format
            img_np = (rgb_img[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            
            # Add gradient background for better depth perception
            bg = np.linspace(20, 50, img_res[0])[:, None] * np.ones((img_res[0], img_res[1], 1))
            bg = np.repeat(bg, 3, axis=2).astype(np.uint8)
            
            # Composite volume over background using alpha from rendered image
            alpha = rgb_img[0].sum(dim=0, keepdim=True).clamp(0, 1).permute(1, 2, 0)  # [H,W,1]
            alpha_np = alpha.cpu().numpy()
            img_np = (img_np * alpha_np + bg * (1 - alpha_np)).astype(np.uint8)
        
        # Convert to pygame surface
        img_np = np.ascontiguousarray(img_np)
        surf = pygame.surfarray.make_surface(img_np.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        
        # Draw UI
        font = pygame.font.Font(None, 24)
        
        # Tool info
        text = font.render(tool_info, True, (255, 255, 255))
        screen.blit(text, (10, 10))
        
        # Mode info with color coding
        mode_color = (0, 255, 100) if render_mode == "volume" else (255, 200, 0)
        mode_text = font.render(f"Mode: {render_mode.upper()} | Radius: {orbit_radius:.1f}", True, mode_color)
        screen.blit(mode_text, (10, 35))
        
        # Last operation
        if last_operation:
            op_text = font.render(last_operation, True, (0, 255, 0))
            screen.blit(op_text, (10, 60))
        
        # Voxel count
        active_voxels = (sigma_t[0, 0] > 0.5).sum().item()
        count_text = font.render(f"Active voxels: {active_voxels}", True, (255, 255, 0))
        screen.blit(count_text, (10, img_res[1] - 30))
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()
    print("Viewer closed")


if __name__ == "__main__":
    main()

