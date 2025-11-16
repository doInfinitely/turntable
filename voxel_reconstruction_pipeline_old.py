import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import math
from pathlib import Path


# ===============================================================
# Ground truth voxel grid (RGBA)
# ===============================================================

def make_gt_voxels(N=32):
    """
    Creates an RGBA volume with a colored sphere.
    Output shape: [4, D, H, W]
    """
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    z = np.linspace(-1, 1, N)

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    r = np.sqrt(X*X + Y*Y + Z*Z)
    inside = r < 0.6

    rgba = np.zeros((N, N, N, 4), dtype=np.float32)

    # Alpha
    rgba[..., 3] = np.clip((0.6 - r) * 3, 0, 1)

    # Colors vary with coordinates
    rgba[..., 0] = (X + 1) / 2
    rgba[..., 1] = (Y + 1) / 2
    rgba[..., 2] = (Z + 1) / 2

    rgba[~inside] = 0

    return torch.tensor(rgba).permute(3, 0, 1, 2)  # [4,D,H,W]


# ===============================================================
# Orbit cameras
# ===============================================================

def orbit_camera(num_views, radius=4.0):
    """
    Yields camera positions on a circle
    """
    for i in range(num_views):
        theta = 2 * math.pi * i / num_views
        yield np.array([
            radius * math.cos(theta),
            radius * 0.4,
            radius * math.sin(theta)
        ], dtype=np.float32)


# ===============================================================
# Differentiable voxel renderer
# ===============================================================

def render_voxels(vox, cam_pos, img_res=128, steps=64):
    """
    Very simple ray marcher.
    vox: [4,D,H,W]
    cam_pos: (3,)
    returns [3,H,W]
    """
    device = vox.device
    _, D, H, W = vox.shape
    half = D / 2

    cam = torch.tensor(cam_pos, dtype=torch.float32, device=device)

    # Camera basis
    forward = -cam
    forward = forward / (torch.norm(forward) + 1e-8)

    world_up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
    right = torch.cross(forward, world_up, dim=0)
    right = right / (torch.norm(right) + 1e-8)
    up = torch.cross(right, forward, dim=0)

    # Pixel grid in camera plane
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, img_res, device=device),
        torch.linspace(-1, 1, img_res, device=device),
        indexing="ij"
    )
    xs = xs.unsqueeze(-1)
    ys = ys.unsqueeze(-1)

    cam_plane = cam + xs * right + ys * up   # [H,W,3]

    # Ray marching param t
    ts = torch.linspace(0, 6, steps, device=device)[None, None, :, None]  # [1,1,S,1]

    ray_dir = forward.view(1, 1, 1, 3)  # [1,1,1,3]

    pts = cam_plane.unsqueeze(2) + ts * ray_dir  # [H,W,S,3]

    # Convert to voxel coords
    p = pts + half
    p = torch.clamp(p, 0, D - 1)
    pi = p.long()   # indices [H,W,S,3]

    # Sample voxels
    rgba = vox[:, pi[..., 0], pi[..., 1], pi[..., 2]]  # [4,H,W,S]
    rgba = rgba.permute(1, 2, 3, 0)  # [H,W,S,4]

    rgb = rgba[..., :3]   # [H,W,S,3]
    a   = rgba[..., 3]    # [H,W,S]

    # Alpha compositing front-to-back
    rgb_acc = torch.zeros((img_res, img_res, 3), device=device)
    alpha_acc = torch.zeros((img_res, img_res), device=device)

    for s in range(steps):
        col = rgb[:, :, s, :]   # [H,W,3]
        al  = a[:, :, s]        # [H,W]

        rgb_acc += (1 - alpha_acc).unsqueeze(-1) * col * al.unsqueeze(-1)
        alpha_acc += (1 - alpha_acc) * al

    return rgb_acc.permute(2, 0, 1)  # [3,H,W]


# ===============================================================
# Training
# ===============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    N = 32
    img_res = 128
    num_views = 20
    steps = 64

    out_dir = Path("voxel_recon_out")
    out_dir.mkdir(exist_ok=True)

    # Ground truth voxels
    gt_vox = make_gt_voxels(N).to(device)

    # Render ground truth images
    cams = list(orbit_camera(num_views))
    gt_imgs = []
    for i, cam in enumerate(cams):
        img = render_voxels(gt_vox, cam, img_res=img_res, steps=steps)
        gt_imgs.append(img.detach())
        img_np = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"gt_{i:03d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    gt_stack = torch.stack(gt_imgs)  # [V,3,H,W]

    # Learnable voxel volume
    recon = nn.Parameter(torch.rand(4, N, N, N, device=device))
    optimizer = optim.Adam([recon], lr=5e-2)

    # Training loop
    for it in range(501):
        pred_imgs = []
        for cam in cams:
            img = render_voxels(recon.sigmoid(), cam, img_res=img_res, steps=steps)
            pred_imgs.append(img)
        pred_stack = torch.stack(pred_imgs)

        loss = F.mse_loss(pred_stack, gt_stack)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 20 == 0:
            print(f"[{it}/500] loss={loss.item():.6e}")

    # Save recon images
    recon_vox = recon.sigmoid().detach()
    for i, cam in enumerate(cams):
        img = render_voxels(recon_vox, cam, img_res=img_res, steps=steps)
        img_np = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"recon_{i:03d}.png"),
                    cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    print("Done. Check voxel_recon_out/")


if __name__ == "__main__":
    main()

