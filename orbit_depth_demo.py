# orbit_depth_demo.py
import cv2
import numpy as np

from affine_vector_field import dense_motion_field
from orbit_depth_utils import (
    load_grayscale_frames,
    estimate_orbit_period,
    infer_rotation_direction,
    depth_from_flow,
)


def main(video_path, pair_start_frame=0):
    # 1) Load a manageable number of frames
    frames = load_grayscale_frames(video_path, max_frames=600)
    if len(frames) < pair_start_frame + 2:
        raise ValueError("Not enough frames for the requested pair.")

    H, W = frames[0].shape[:2]

    # Read FPS (can be 0 on some codecs)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        fps = 30.0  # fallback

    # 2) Estimate orbit period (in frames) from image self-similarity
    period_frames, diffs = estimate_orbit_period(frames)
    print(f"Estimated orbit period: {period_frames} frames, fps={fps:.2f}")

    # 3) Pick the frame pair
    img1 = frames[pair_start_frame]
    img2 = frames[pair_start_frame + 1]

    # 4) Compute affine-based motion field
    flow, errors, affmats = dense_motion_field(
        img1,
        img2,
        window=32,
        stride=16,
        search=16,
        scales=[1.0],
        angles=[0.0],
        shears=[0.0],
        reg_lambda=1e-4,
    )
    print("flow shape:", flow.shape)

    # 5) Infer rotation direction
    rot_dir_sign = infer_rotation_direction(flow)
    print("Rotation direction sign:", rot_dir_sign)

    # 6) Convert flow to depth
    depth = depth_from_flow(
        flow,
        frame_width=W,
        period_frames=period_frames,
        fps=fps,
        rot_dir_sign=rot_dir_sign,
    )

    # 7) Visualize depth (upsample to full res)
    depth_norm = depth - depth.min()
    if depth_norm.max() > 0:
        depth_norm /= depth_norm.max()

    depth_resized = cv2.resize(depth_norm, (W, H), interpolation=cv2.INTER_CUBIC)
    depth_vis = (depth_resized * 255).astype(np.uint8)
    depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    cv2.imwrite("depth_debug.png", depth_vis_color)
    print("Saved depth_debug.png")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python orbit_depth_demo.py path/to/video.mp4 [start_frame]")
        raise SystemExit

    video_path = sys.argv[1]
    if len(sys.argv) > 2:
        start = int(sys.argv[2])
    else:
        start = 0

    main(video_path, pair_start_frame=start)

