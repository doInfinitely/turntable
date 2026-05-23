# view_shell.py
# Render a free-camera turntable of a trained GaussianShell.
#
# Loads shell.npz produced by gaussian_shell.py (final or snapshot), renders
# an orbit at user-specified elevation, and writes an mp4 — handy for seeing
# what the shell looks like from views the training orbit never sees (top,
# bottom, oblique).

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from gaussian_shell import GaussianShell, render_shell
from video_orbit_voxel_recon import make_intrinsics


def _orbit_pose(theta, elevation_rad, radius, device):
    """Camera looking at origin from (radius, elevation, azimuth=theta)."""
    C = np.array([
        radius * math.cos(elevation_rad) * math.cos(theta),
        radius * math.sin(elevation_rad),
        radius * math.cos(elevation_rad) * math.sin(theta),
    ], dtype=np.float32)
    look = np.zeros(3, dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    fwd = look - C
    fwd /= np.linalg.norm(fwd) + 1e-8
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-8
    true_up = np.cross(right, fwd)
    R = np.stack([right, true_up, fwd], axis=0).astype(np.float32)
    t = (-R @ C[:, None]).astype(np.float32)
    return torch.from_numpy(R).to(device), torch.from_numpy(t).to(device)


def render_turntable(
    shell,
    n_frames=60,
    img_size=(256, 256),
    fov_y_deg=45.0,
    radius=2.5,
    elevation_deg=20.0,
    chunk_size=1024,
    device="cuda:0",
):
    H, W = img_size
    K = make_intrinsics(H, W, fov_y_deg=fov_y_deg, device=device)
    elev = math.radians(elevation_deg)
    frames = []
    for i in range(n_frames):
        theta = 2 * math.pi * i / n_frames
        R, t = _orbit_pose(theta, elev, radius, device)
        with torch.no_grad():
            img, _ = render_shell(
                shell, K, R, t, H, W,
                chunk_size=chunk_size, use_checkpoint=False,
            )
        np_img = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames.append(np_img)
    return frames


def write_video(frames, out_path, fps=24):
    """Write frames as a video. If out_path ends in .webm, encode via ffmpeg
    (libvpx-vp9 — browser-friendly, no proprietary codecs). Otherwise use
    OpenCV's VideoWriter (mp4v fourcc — needs codec support to play)."""
    out_path = Path(out_path)
    H, W = frames[0].shape[:2]
    if out_path.suffix.lower() == ".webm":
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(".webm output requires ffmpeg in PATH")
        with tempfile.TemporaryDirectory() as td:
            for i, f in enumerate(frames):
                cv2.imwrite(f"{td}/{i:05d}.png", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-framerate", str(fps),
                 "-i", f"{td}/%05d.png",
                 "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32",
                 "-pix_fmt", "yuv420p",
                 str(out_path)],
                check=True,
            )
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()


def main():
    p = argparse.ArgumentParser(description="Render a turntable of a trained shell.")
    p.add_argument("--shell-npz", required=True, help="Path to shell.npz")
    p.add_argument("--out", default=None,
                   help="Output video path. .webm uses ffmpeg/VP9; .mp4 uses "
                        "OpenCV's mp4v (default: <shell-npz>.turntable.webm).")
    p.add_argument("--n-frames", type=int, default=60)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--img-res", type=int, nargs=2, default=[256, 256])
    p.add_argument("--fov-y-deg", type=float, default=45.0)
    p.add_argument("--radius", type=float, default=2.5)
    p.add_argument("--elevation-deg", type=float, default=20.0,
                   help="Elevation above the equator (training orbit was 0°)")
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--device", default=None,
                   help="cpu / cuda:0 / etc. Default: cuda if available else cpu.")
    p.add_argument("--also-png-grid", action="store_true",
                   help="Also write a 4-panel PNG with views at 0/90/180/270° azimuth.")
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out_path = args.out or (Path(args.shell_npz).with_suffix("").as_posix()
                            + ".turntable.webm")

    shell = GaussianShell.from_npz(args.shell_npz).to(device)
    print(f"[VIEWER] loaded shell: {shell.n_shells} shells × {shell.n_per_shell} "
          f"= {shell.n_shells * shell.n_per_shell} Gaussians, sigma_world="
          f"{shell.sigma_world:.4f}, device={device}")

    frames = render_turntable(
        shell,
        n_frames=args.n_frames,
        img_size=tuple(args.img_res),
        fov_y_deg=args.fov_y_deg,
        radius=args.radius,
        elevation_deg=args.elevation_deg,
        chunk_size=args.chunk_size,
        device=device,
    )
    write_video(frames, out_path, fps=args.fps)
    print(f"[VIEWER] wrote {len(frames)} frames at {args.img_res[0]}×"
          f"{args.img_res[1]} to {out_path}")

    if args.also_png_grid:
        grid_path = Path(out_path).with_suffix(".grid.png")
        # Pick frames closest to 0/90/180/270°
        idxs = [int(round(k * args.n_frames / 4)) % args.n_frames for k in range(4)]
        H, W = frames[0].shape[:2]
        row = np.concatenate([frames[i] for i in idxs], axis=1)
        cv2.imwrite(str(grid_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        print(f"[VIEWER] wrote 4-panel grid to {grid_path}")


if __name__ == "__main__":
    main()
