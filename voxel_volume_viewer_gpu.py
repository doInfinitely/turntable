# voxel_volume_viewer_gpu.py
# GPU-accelerated voxel viewer using PyTorch3D for cube rendering
#
# Controls:
#   Arrow keys   - orbit (left/right yaw, up/down pitch)
#   +/-          - zoom in/out
#   V            - volumetric mode
#   C            - cube mode
#   ESC / close  - quit

import sys
import math
import numpy as np
import pygame
import torch
import torch.nn.functional as F

# Try to import PyTorch3D for GPU-accelerated mesh rendering
try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        HardPhongShader,
        PointLights,
        TexturesVertex,
    )
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("[WARN] PyTorch3D not available. Cube mode will use CPU fallback.")


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
    pts_world: [1,S,H,W,3]
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

    pts = C_exp + ts[..., None] * dirs_world  # SxHxWx3 (broadcast)
    pts = pts.unsqueeze(0)                    # [1,S,H,W,3]
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
    NeRF-style compositing along rays.
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


def render_view_volume(sigma, rgb, eye, target,
                       img_size=(128, 128),
                       fov_y_deg=45.0,
                       scene_radius=1.5,
                       n_samples=64,
                       device="cpu"):
    """
    Volumetric rendering: same as training.
    Returns [3,H,W] torch in [0,1]
    """
    H, W = img_size
    K = make_intrinsics(H, W, fov_y_deg=fov_y_deg, device=device)
    R, t = build_pose_from_eye(eye, target, device=device)

    pts = generate_rays(H, W, K, R, t,
                        n_samples=n_samples,
                        near=0.1,
                        far=5.0,
                        device=device)
    sigma_s, rgb_s = sample_volume(sigma, rgb, pts, scene_radius=scene_radius)
    rgb_img = volume_render(sigma_s, rgb_s, n_samples)  # [1,3,H,W]
    return rgb_img[0]  # [3,H,W]


# ---------- GPU-accelerated cube renderer (PyTorch3D) ----------

def build_voxel_mesh(sigma_np, rgb_np, scene_radius=1.5, thresh_factor=0.3, device="cpu"):
    """
    Build a mesh of cubes from voxel data.
    Returns verts, faces, colors for PyTorch3D.
    """
    D, H, W = sigma_np.shape
    
    # Threshold voxels
    max_sigma = float(sigma_np.max()) if sigma_np.size > 0 else 0.0
    if max_sigma <= 0:
        return None, None, None
    
    sigma_thresh = max_sigma * thresh_factor
    mask = sigma_np > sigma_thresh
    idxs = np.argwhere(mask)
    
    if idxs.shape[0] == 0:
        return None, None, None
    
    # World space grid
    zs = np.linspace(-1, 1, D) * scene_radius
    ys = np.linspace(-1, 1, H) * scene_radius
    xs = np.linspace(-1, 1, W) * scene_radius
    voxel_size = 2.0 * scene_radius / D
    
    # Cube template (8 vertices)
    half = voxel_size * 0.5
    cube_verts = np.array([
        [-half, -half, -half], [half, -half, -half], [half, half, -half], [-half, half, -half],
        [-half, -half, half],  [half, -half, half],  [half, half, half],  [-half, half, half],
    ], dtype=np.float32)
    
    # Cube faces (12 triangles = 6 faces * 2 triangles each)
    cube_faces = np.array([
        [0, 2, 1], [0, 3, 2],  # back
        [4, 5, 6], [4, 6, 7],  # front
        [0, 1, 5], [0, 5, 4],  # bottom
        [3, 6, 2], [3, 7, 6],  # top
        [0, 4, 7], [0, 7, 3],  # left
        [1, 2, 6], [1, 6, 5],  # right
    ], dtype=np.int32)
    
    # Build mesh for all voxels
    all_verts = []
    all_faces = []
    all_colors = []
    
    for idx in idxs:
        z_i, y_i, x_i = idx
        center = np.array([xs[x_i], ys[y_i], zs[z_i]], dtype=np.float32)
        color = rgb_np[z_i, y_i, x_i]
        
        # Transform cube vertices to world position
        verts = cube_verts + center
        faces = cube_faces + len(all_verts)  # Offset face indices
        
        all_verts.append(verts)
        all_faces.append(faces)
        # Vertex colors (8 vertices per cube, all same color)
        all_colors.extend([color] * 8)
    
    verts_np = np.vstack(all_verts)
    faces_np = np.vstack(all_faces)
    colors_np = np.array(all_colors, dtype=np.float32)
    
    # Convert to torch
    verts = torch.from_numpy(verts_np).to(device)
    faces = torch.from_numpy(faces_np).to(device)
    colors = torch.from_numpy(colors_np).to(device)
    
    return verts, faces, colors


def render_view_cubes_gpu(sigma_np, rgb_np, eye, target,
                          img_size=(256, 256),
                          fov_y_deg=45.0,
                          scene_radius=1.5,
                          thresh_factor=0.3,
                          device="cpu"):
    """
    GPU-accelerated cube rendering using PyTorch3D.
    """
    if not PYTORCH3D_AVAILABLE:
        return render_view_cubes_cpu_fallback(sigma_np, rgb_np, eye, target, 
                                               img_size, fov_y_deg, scene_radius, thresh_factor)
    
    H, W = img_size
    
    # Build mesh
    verts, faces, colors = build_voxel_mesh(sigma_np, rgb_np, scene_radius, thresh_factor, device)
    
    if verts is None:
        return np.zeros((H, W, 3), dtype=np.uint8)
    
    # Create mesh
    verts_batch = verts.unsqueeze(0)  # [1, V, 3]
    faces_batch = faces.unsqueeze(0)  # [1, F, 3]
    colors_batch = colors.unsqueeze(0)  # [1, V, 3]
    
    textures = TexturesVertex(verts_features=colors_batch)
    mesh = Meshes(verts=verts_batch, faces=faces_batch, textures=textures)
    
    # Set up camera (PyTorch3D convention)
    R_p3d, T_p3d = look_at_view_transform(
        eye=[eye.tolist()],
        at=[target.tolist()],
        up=[[0, 1, 0]],
        device=device
    )
    
    cameras = FoVPerspectiveCameras(
        device=device,
        R=R_p3d,
        T=T_p3d,
        fov=fov_y_deg,
    )
    
    # Rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    
    # Create renderer
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )
    
    # Render
    with torch.no_grad():
        images = renderer(mesh)  # [1, H, W, 4] (RGBA)
        rgb = images[0, ..., :3]  # [H, W, 3]
        rgb = torch.clamp(rgb, 0, 1)
    
    # Convert to numpy
    img_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
    return img_np


def render_view_cubes_cpu_fallback(sigma_np, rgb_np, eye, target,
                                   img_size=(256, 256),
                                   fov_y_deg=45.0,
                                   scene_radius=1.5,
                                   thresh_factor=0.3):
    """
    CPU fallback for cube rendering (simple point rendering).
    """
    H_img, W_img = img_size
    
    # Camera intrinsics
    fov_y = math.radians(fov_y_deg)
    fy = 0.5 * H_img / math.tan(0.5 * fov_y)
    fx = fy
    cx = W_img / 2.0
    cy = H_img / 2.0
    
    # Build R,t
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8
    true_up = np.cross(right, forward)
    
    R = np.stack([right, true_up, forward], axis=0)
    t = -R @ eye
    
    D, H, W = sigma_np.shape
    
    # Threshold
    max_sigma = float(sigma_np.max()) if sigma_np.size > 0 else 0.0
    if max_sigma <= 0:
        return np.zeros((H_img, W_img, 3), dtype=np.uint8)
    
    sigma_thresh = max_sigma * thresh_factor
    mask = sigma_np > sigma_thresh
    idxs = np.argwhere(mask)
    
    if idxs.shape[0] == 0:
        return np.zeros((H_img, W_img, 3), dtype=np.uint8)
    
    zs = np.linspace(-1, 1, D) * scene_radius
    ys = np.linspace(-1, 1, H) * scene_radius
    xs = np.linspace(-1, 1, W) * scene_radius
    
    z_idx, y_idx, x_idx = idxs[:, 0], idxs[:, 1], idxs[:, 2]
    Xw = np.stack([xs[x_idx], ys[y_idx], zs[z_idx]], axis=1)
    colors = (rgb_np[z_idx, y_idx, x_idx] * 255.0).astype(np.uint8)
    
    X_cam = (R @ Xw.T).T + t
    z_cam = X_cam[:, 2]
    
    front_mask = z_cam > 0.01
    if not np.any(front_mask):
        return np.zeros((H_img, W_img, 3), dtype=np.uint8)
    
    X_cam = X_cam[front_mask]
    colors = colors[front_mask]
    z_cam = z_cam[front_mask]
    
    u = fx * (X_cam[:, 0] / z_cam) + cx
    v = fy * (X_cam[:, 1] / z_cam) + cy
    
    u_round = np.round(u).astype(int)
    v_round = np.round(v).astype(int)
    
    in_bounds = ((u_round >= 0) & (u_round < W_img) & (v_round >= 0) & (v_round < H_img))
    if not np.any(in_bounds):
        return np.zeros((H_img, W_img, 3), dtype=np.uint8)
    
    u_round = u_round[in_bounds]
    v_round = v_round[in_bounds]
    colors = colors[in_bounds]
    z_cam = z_cam[in_bounds]
    
    order = np.argsort(z_cam)
    u_round = u_round[order]
    v_round = v_round[order]
    colors = colors[order]
    
    img = np.zeros((H_img, W_img, 3), dtype=np.uint8)
    img[v_round, u_round] = colors
    
    return img


# ---------- Viewer main loop ----------

def main(npz_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    if PYTORCH3D_AVAILABLE:
        print("PyTorch3D available - using GPU-accelerated cube rendering")
    else:
        print("PyTorch3D not available - using CPU fallback")

    data = np.load(npz_path)
    sigma_np = data["sigma"].astype(np.float32)       # [D,H,W]
    rgb_np   = data["rgb"].astype(np.float32)         # [D,H,W,3] in [0,1]

    D, H, W = sigma_np.shape
    print(f"Loaded volume: D={D}, H={H}, W={W}")

    # Torch tensors for volume mode
    sigma_t = torch.from_numpy(sigma_np)[None, None].to(device)   # [1,1,D,H,W]
    rgb_t   = torch.from_numpy(rgb_np).permute(3, 0, 1, 2)[None].to(device)  # [1,3,D,H,W]

    # Pygame window
    pygame.init()
    win_size = 512
    screen = pygame.display.set_mode((win_size, win_size))
    pygame.display.set_caption("Voxel Volume Viewer GPU (V=volume, C=cubes)")

    clock = pygame.time.Clock()

    # Render resolution (internal)
    render_H, render_W = 256, 256

    # Orbit params
    radius = 2.5
    angle_y = 0.0   # yaw
    angle_x = 0.0   # pitch
    orbit_speed = 0.03
    zoom_speed = 0.05
    fov_y_deg = 45.0
    scene_radius = 1.5
    n_samples = 64

    mode = "volume"  # or "cubes"

    running = True
    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_v:
                    mode = "volume"
                    pygame.display.set_caption(f"Voxel Volume Viewer GPU (mode: volume) [{device}]")
                elif event.key == pygame.K_c:
                    mode = "cubes"
                    pygame.display.set_caption(f"Voxel Volume Viewer GPU (mode: cubes) [{device}]")

        keys = pygame.key.get_pressed()

        # Orbit controls
        if keys[pygame.K_LEFT]:
            angle_y -= orbit_speed
        if keys[pygame.K_RIGHT]:
            angle_y += orbit_speed
        if keys[pygame.K_UP]:
            angle_x = max(angle_x - orbit_speed, -math.pi / 2 + 0.1)
        if keys[pygame.K_DOWN]:
            angle_x = min(angle_x + orbit_speed,  math.pi / 2 - 0.1)

        # Zoom
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            radius = max(0.5, radius - zoom_speed)
        if keys[pygame.K_MINUS]:
            radius += zoom_speed

        # Camera position (spherical)
        cx = radius * math.cos(angle_x) * math.cos(angle_y)
        cy = radius * math.sin(angle_x)
        cz = radius * math.cos(angle_x) * math.sin(angle_y)
        eye = np.array([cx, cy, cz], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if mode == "volume":
            with torch.no_grad():
                img_t = render_view_volume(
                    sigma_t, rgb_t, eye, target,
                    img_size=(render_H, render_W),
                    fov_y_deg=fov_y_deg,
                    scene_radius=scene_radius,
                    n_samples=n_samples,
                    device=device,
                )  # [3,H,W]
            img_np = img_t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = render_view_cubes_gpu(
                sigma_np, rgb_np, eye, target,
                img_size=(render_H, render_W),
                fov_y_deg=fov_y_deg,
                scene_radius=scene_radius,
                thresh_factor=0.3,
                device=device,
            )

        # Convert to Surface and scale to window
        surf = pygame.surfarray.make_surface(np.rot90(img_np, k=1))
        surf = pygame.transform.smoothscale(surf, (win_size, win_size))

        screen.blit(surf, (0, 0))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python voxel_volume_viewer_gpu.py <path_to_recon_volume.npz>")
        sys.exit(1)

    main(sys.argv[1])

