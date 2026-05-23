#!/usr/bin/env python3
"""Filter GaussianShell floaters using Depth Anything.

For each view, DA gives a depth surface. Any Gaussian in front of that
surface (between camera and object) is a floater. Sweep the DA-to-camera
scale factor to find the right calibration.
"""

import argparse, math, os, subprocess, tempfile
import cv2
import numpy as np
import torch

from gaussian_shell import GaussianShell, render_shell
from video_orbit_voxel_recon import (
    build_orbit_poses_from_period,
    estimate_orbit_period,
    estimate_orbit_direction,
    make_intrinsics,
)


def get_da_inv_depths(video_path, frame_indices, img_res, device="cuda:0"):
    """Run Depth Anything, return [V, H, W] inverse-depth (larger=closer)."""
    from depth_anything import DepthAnythingEstimator
    da = DepthAnythingEstimator(device=device)
    cap = cv2.VideoCapture(video_path)
    Ht, Wt = img_res
    maps = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {fi}")
        frame_bgr = cv2.resize(frame_bgr, (Wt, Ht), interpolation=cv2.INTER_AREA)
        from PIL import Image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = da.pipe(Image.fromarray(rgb))
        inv_d = result["predicted_depth"]
        if isinstance(inv_d, torch.Tensor):
            inv_d = inv_d.detach().cpu().numpy()
        if inv_d.ndim == 3:
            inv_d = inv_d[0]
        inv_d = cv2.resize(inv_d, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        maps.append(inv_d)
    cap.release()
    print(f"  DA done: {len(maps)} frames")
    return np.stack(maps, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shell-npz", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--img-res", type=int, nargs=2, default=[384, 384])
    p.add_argument("--fov-y-deg", type=float, default=45.0)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--min-views-frac", type=float, default=0.3,
                   help="Fraction of visible views a Gaussian must be a floater in to be suppressed")
    p.add_argument("--scale", type=float, default=None,
                   help="If set, apply this single scale factor and save culled npz (no sweep)")
    p.add_argument("--out-npz", default=None,
                   help="Output path for culled npz (default: <input>_culled.npz)")
    args = p.parse_args()

    H, W = args.img_res

    # 1. Orbit params
    print("[1/4] Orbit estimation...")
    orbit_period, _ = estimate_orbit_period(args.video)
    direction = estimate_orbit_direction(args.video)
    frame_indices = list(range(orbit_period))

    # 2. Poses & intrinsics
    K = make_intrinsics(H, W, fov_y_deg=args.fov_y_deg, device="cpu")
    poses = build_orbit_poses_from_period(
        frame_indices, orbit_period, direction,
        radius=2.5, device="cpu", theta0=0.0,
    )
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # 3. DA inverse depth maps
    print(f"[2/4] Running Depth Anything on {len(frame_indices)} frames...")
    da_inv = get_da_inv_depths(args.video, frame_indices, args.img_res, device=args.device)
    # Clamp negatives
    da_inv = np.clip(da_inv, 0.01, None)
    V = len(frame_indices)

    # 4. Project all Gaussians into all views
    print("[3/4] Projecting Gaussians...")
    d = np.load(args.shell_npz)
    dirs = d["directions"]
    radii = d["radii"]
    anchors = d["anchors"]
    n_shells = int(d["n_shells"])
    n_per_shell = int(d["n_per_shell"])

    positions = np.concatenate(
        [anchors[s] + dirs * radii[s, :, None] for s in range(n_shells)]
    ).astype(np.float32)  # [G, 3]
    G = len(positions)

    # For each view, get each Gaussian's cam_z and the DA inv_depth at its pixel
    # Store: cam_z[V, G] and da_at_pixel[V, G]
    cam_z_all = np.zeros((V, G), dtype=np.float32)
    da_at_pixel = np.zeros((V, G), dtype=np.float32)
    visible = np.zeros((V, G), dtype=bool)

    for vi, (R, t) in enumerate(poses):
        R_np = R.cpu().numpy()
        t_np = t.cpu().numpy().flatten()
        p_cam = positions @ R_np.T + t_np[None, :]
        z = p_cam[:, 2]
        safe_z = np.clip(z, 0.05, None)
        u = fx * p_cam[:, 0] / safe_z + cx
        v = fy * p_cam[:, 1] / safe_z + cy

        on_screen = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        cam_z_all[vi] = z
        visible[vi] = on_screen

        u_int = np.clip(u.astype(int), 0, W - 1)
        v_int = np.clip(v.astype(int), 0, H - 1)
        da_at_pixel[vi] = da_inv[vi][v_int, u_int]

    # 5. Cull or sweep
    # Model: surface_z = scale / da_inv_depth
    # A Gaussian is a floater if cam_z < surface_z (it's between camera and surface)
    # i.e. cam_z * da_at_pixel < scale. Floater if product < scale.

    product = cam_z_all * da_at_pixel  # [V, G]

    def apply_cull(scale):
        is_floater = visible & (product < scale)  # [V, G]
        n_vis = visible.sum(axis=0).clip(1)
        floater_frac = is_floater.sum(axis=0) / n_vis
        suppress = floater_frac > args.min_views_frac
        return suppress.reshape(n_shells, n_per_shell)

    if args.scale is not None:
        # Single-scale cull: save filtered npz
        print(f"[4/4] Culling at scale={args.scale}, min_views_frac={args.min_views_frac}...")
        suppress_2d = apply_cull(args.scale)
        kept = (~suppress_2d).sum()
        removed = suppress_2d.sum()
        print(f"  Kept {kept}, removed {removed} Gaussians")

        alphas_f = d["alphas"].copy()
        colors_f = d["colors"].copy()
        alphas_f[suppress_2d] = 0.0
        colors_f[suppress_2d] = 0.0

        out_npz = args.out_npz or args.shell_npz.replace(".npz", "_culled.npz")
        np.savez(out_npz,
            directions=d["directions"], anchors=d["anchors"],
            radii=d["radii"], alphas=alphas_f, colors=colors_f,
            sigma_world=d["sigma_world"], n_shells=d["n_shells"],
            n_per_shell=d["n_per_shell"], share_directions=d["share_directions"])
        print(f"Saved: {out_npz}")
    else:
        # Sweep mode
        print("[4/4] Rendering sweep...")
        device = args.device
        K_render = make_intrinsics(H, W, fov_y_deg=args.fov_y_deg, device=device)
        C = np.array([2.5, 0, 0], dtype=np.float32)
        fwd = -C / np.linalg.norm(C)
        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        true_up = np.cross(right, fwd)
        R_render = torch.from_numpy(np.stack([right, true_up, fwd]).astype(np.float32)).to(device)
        t_render = torch.from_numpy((-R_render.cpu().numpy() @ C[:, None]).astype(np.float32)).to(device)

        tmpdir = tempfile.mkdtemp()
        p50 = np.median(product[visible])
        print(f"  product median (visible): {p50:.2f}")
        scale_values = np.linspace(p50 * 0.2, p50 * 2.0, 50)

        for i, scale in enumerate(scale_values):
            suppress_2d = apply_cull(scale)
            alphas_f = d["alphas"].copy()
            colors_f = d["colors"].copy()
            alphas_f[suppress_2d] = 0.0
            colors_f[suppress_2d] = 0.0
            kept = (~suppress_2d).sum()

            np.savez("/tmp/_tmp_df.npz",
                directions=d["directions"], anchors=d["anchors"],
                radii=d["radii"], alphas=alphas_f, colors=colors_f,
                sigma_world=d["sigma_world"], n_shells=d["n_shells"],
                n_per_shell=d["n_per_shell"], share_directions=d["share_directions"])
            shell = GaussianShell.from_npz("/tmp/_tmp_df.npz").to(device)
            with torch.no_grad():
                img, _ = render_shell(shell, K_render, R_render, t_render, H, W,
                                      chunk_size=1024, use_checkpoint=False)
            np_img = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, f"scale={scale:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(bgr, f"{kept} pts", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imwrite(f"{tmpdir}/{i:05d}.png", bgr)
            print(f"  scale={scale:.2f}  kept={kept:6d}")

        out_path = args.shell_npz.replace(".npz", "_depth_sweep.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", "4", "-i", f"{tmpdir}/%05d.png",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            out_path,
        ], check=True)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
