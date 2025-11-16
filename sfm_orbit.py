#!/usr/bin/env python3
"""
Minimal, explicit SfM-style reconstruction from an orbit video.

Usage:
    1. Extract frames with ffmpeg, e.g.:
       ffmpeg -y -i orbit.mp4 -vf fps=4 frames/%06d.jpg

    2. Run:
       python sfm_orbit.py --frames_dir frames --out points3d_sfm.ply

Requirements:
    pip install opencv-python numpy
"""

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np


def load_images(frames_dir):
    frames = sorted(
        glob.glob(os.path.join(frames_dir, "*.jpg"))
        + glob.glob(os.path.join(frames_dir, "*.png"))
        + glob.glob(os.path.join(frames_dir, "*.jpeg"))
    )
    if not frames:
        raise RuntimeError(f"No images found in {frames_dir}")
    imgs = [cv2.imread(f, cv2.IMREAD_COLOR) for f in frames]
    return frames, imgs


def build_intrinsics(img_shape, focal_scale=1.2):
    """Build a simple pinhole K based on image size."""
    h, w = img_shape[:2]
    f = focal_scale * max(w, h)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)
    return K


def detect_and_describe(img):
    """Detect ORB keypoints and descriptors."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2000)
    kps, des = orb.detectAndCompute(gray, None)
    return kps, des


def match_features(des1, des2, max_matches=800):
    """Brute-force Hamming matcher with cross-check."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    return matches[:max_matches]


def extract_matched_points(kps1, kps2, matches):
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def estimate_relative_pose(pts1, pts2, K):
    """
    Estimate essential matrix and relative pose between two views.

    pts1, pts2: Nx2 pixel coords
    K: 3x3 intrinsics
    """
    E, inliers = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0  # pixel threshold
    )
    if E is None:
        return None, None, None

    inliers = inliers.ravel().astype(bool)
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]

    _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)
    return R, t, inliers


def triangulate_points(P1, P2, pts1, pts2):
    """
    Triangulate corresponding points between two views.
    P1, P2: 3x4 projection matrices
    pts1, pts2: Nx2 pixel coords
    """
    pts1_h = pts1.T  # 2xN
    pts2_h = pts2.T

    pts4d_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)  # 4xN
    pts3d = (pts4d_h[:3] / pts4d_h[3]).T  # Nx3
    return pts3d


def make_projection_matrix(K, R, t):
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def write_ply(points, colors, out_path):
    """
    Write an ASCII PLY with XYZ + RGB.
    points: Nx3
    colors: Nx3 uint8 (BGR or RGB; we’ll output as RGB)
    """
    # Convert BGR (OpenCV) to RGB
    colors_rgb = colors[:, ::-1] if colors is not None else None

    with open(out_path, "w") as f:
        n = len(points)
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
        for i in range(n):
            x, y, z = points[i]
            if colors_rgb is not None:
                r, g, b = colors_rgb[i]
            else:
                r = g = b = 255
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def reconstruct_sequence(imgs, K, sample_step=1):
    """
    Incremental 2-view SfM along a sequence.
    Returns concatenated 3D points and colors.
    """
    # First camera at origin
    R_cum = np.eye(3)
    t_cum = np.zeros((3, 1))

    all_points = []
    all_colors = []

    # Use first image for color sampling baseline
    for i in range(0, len(imgs) - sample_step, sample_step):
        img1 = imgs[i]
        img2 = imgs[i + sample_step]

        print(f"[Pair] {i} -> {i + sample_step}")

        kps1, des1 = detect_and_describe(img1)
        kps2, des2 = detect_and_describe(img2)

        if des1 is None or des2 is None:
            print("  No descriptors, skipping")
            continue

        matches = match_features(des1, des2)
        if len(matches) < 20:
            print(f"  Too few matches ({len(matches)}), skipping")
            continue

        pts1, pts2 = extract_matched_points(kps1, kps2, matches)

        R_rel, t_rel, inliers = estimate_relative_pose(pts1, pts2, K)
        if R_rel is None:
            print("  Pose estimation failed, skipping")
            continue

        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]

        # Update global pose: new camera pose in world frame
        R_cum = R_rel @ R_cum
        t_cum = R_rel @ t_cum + t_rel

        # Projection matrices
        P1 = make_projection_matrix(K, np.eye(3), np.zeros((3, 1)))      # world = cam0
        P2 = make_projection_matrix(K, R_cum, t_cum)                     # current cam

        pts3d = triangulate_points(P1, P2, pts1_in, pts2_in)

        # Filter out points with crazy depth (e.g. behind camera or too far)
        z = pts3d[:, 2]
        good = (z > 0.1) & (z < 100.0)
        pts3d = pts3d[good]
        pts1_valid = pts1_in[good]

        # Sample colors from first image of the pair
        pts1_pix = np.round(pts1_valid).astype(int)
        h, w = img1.shape[:2]
        mask = (
            (pts1_pix[:, 0] >= 0) & (pts1_pix[:, 0] < w) &
            (pts1_pix[:, 1] >= 0) & (pts1_pix[:, 1] < h)
        )
        pts3d = pts3d[mask]
        pts1_pix = pts1_pix[mask]

        colors = img1[pts1_pix[:, 1], pts1_pix[:, 0], :]

        print(f"  Triangulated {len(pts3d)} points")

        all_points.append(pts3d)
        all_colors.append(colors)

    if not all_points:
        raise RuntimeError("No points reconstructed; check matches / frames / K guess.")

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    return all_points, all_colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True, help="Directory of extracted frames")
    parser.add_argument("--out", required=True, help="Output PLY path")
    parser.add_argument("--step", type=int, default=1,
                        help="Frame step between pairs (1=adjacent, 2=every other, etc.)")
    args = parser.parse_args()

    frame_paths, imgs = load_images(args.frames_dir)
    print(f"Loaded {len(imgs)} frames from {args.frames_dir}")

    K = build_intrinsics(imgs[0].shape, focal_scale=1.2)
    print("K =\n", K)

    pts, cols = reconstruct_sequence(imgs, K, sample_step=args.step)
    print(f"Total reconstructed points: {len(pts)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_ply(pts, cols, out_path)
    print("Wrote PLY:", out_path)


if __name__ == "__main__":
    main()

