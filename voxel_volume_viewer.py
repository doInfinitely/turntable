# voxel_volume_viewer.py
# View a reconstructed voxel volume with two modes:
#   1) NeRF-style volumetric rendering (same as training)
#   2) Voxel "cubes" (one pixel per voxel center, depth-sorted)
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


# ---------- Cube renderer (Minecraft-style with actual cube faces) ----------

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


def render_view_cubes(sigma_np, rgb_np, eye, target,
                      img_size=(128, 128),
                      fov_y_deg=45.0,
                      scene_radius=1.5,
                      thresh_factor=0.3):
    """
    Minecraft-style cube voxel renderer with actual cube faces.
    
    sigma_np: [D,H,W]
    rgb_np:   [D,H,W,3] in [0,1]
    eye, target: np arrays (3,)
    Returns: uint8 image [H,W,3]
    """
    H_img, W_img = img_size

    # Camera intrinsics
    fov_y = math.radians(fov_y_deg)
    fy = 0.5 * H_img / math.tan(0.5 * fov_y)
    fx = fy
    cx_i = W_img / 2.0
    cy_i = H_img / 2.0

    # Build camera R,t
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
    voxel_size = 2.0 * scene_radius / D

    # Cube vertices (8 corners)
    half_size = voxel_size * 0.5
    cube_verts = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # back
        [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1],   # front
    ], dtype=np.float32) * half_size

    # Cube faces (6 faces, 4 vertices each)
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

            # Backface culling
            face_normal_world = face_normals[face_idx]
            face_normal_cam = R @ face_normal_world
            if face_normal_cam[2] >= 0:  # Face pointing away
                continue

            # Check if vertices are in front
            if v0[2] <= 0.01 or v1[2] <= 0.01 or v2[2] <= 0.01 or v3[2] <= 0.01:
                continue

            # Project vertices
            def project(v):
                u = fx * (v[0] / v[2]) + cx_i
                v_p = fy * (v[1] / v[2]) + cy_i
                return np.array([u, v_p]), v[2]

            p0, z0 = project(v0)
            p1, z1 = project(v1)
            p2, z2 = project(v2)
            p3, z3 = project(v3)

            # Simple lighting
            light_dir = -forward
            face_brightness = max(0.3, abs(np.dot(face_normal_world, light_dir)))
            lit_color = (color * face_brightness).astype(np.uint8)

            # Rasterize two triangles
            for tri_verts, tri_depths in [([p0, p1, p2], [z0, z1, z2]), 
                                           ([p0, p2, p3], [z0, z2, z3])]:
                rasterize_triangle(img, z_buffer, tri_verts, tri_depths, lit_color, H_img, W_img)

    return img


# ---------- Viewer main loop ----------

def main(npz_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

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
    pygame.display.setcaption = pygame.display.set_caption  # alias
    pygame.display.setcaption("Voxel Volume Viewer (V=volume, C=cubes)")

    clock = pygame.time.Clock()

    # Render resolution (internal)
    render_H, render_W = 256, 256

    # Orbit params
    radius = 2.5  # Match training camera distance
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
                    pygame.display.setcaption("Voxel Volume Viewer (mode: volume)")
                elif event.key == pygame.K_c:
                    mode = "cubes"
                    pygame.display.setcaption("Voxel Volume Viewer (mode: cubes)")

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
            img_np = render_view_cubes(
                sigma_np, rgb_np, eye, target,
                img_size=(render_H, render_W),
                fov_y_deg=fov_y_deg,
                scene_radius=scene_radius,
                thresh_factor=0.3,
            )

        # Convert to Surface and scale to window
        surf = pygame.surfarray.make_surface(np.rot90(img_np, k=1))
        surf = pygame.transform.smoothscale(surf, (win_size, win_size))

        screen.blit(surf, (0, 0))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) < 1 + 1:
        print("Usage: python voxel_volume_viewer.py <path_to_recon_volume.npz>")
        sys.exit(1)

    main(sys.argv[1])

