# orbit_depth_utils.py
import cv2
import numpy as np
from affine_vector_field import dense_motion_field


def load_grayscale_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def estimate_orbit_period(frames, min_separation=10):
    """
    Estimate the orbit period in *frames* by comparing each frame to frame 0.

    Returns:
      period_frames  (int)
      diffs          (np.ndarray) length N-1 of MSEs to frame 0
    """
    ref = frames[0].astype(np.float32) / 255.0
    diffs = []
    for i in range(1, len(frames)):
        cur = frames[i].astype(np.float32) / 255.0
        d = np.mean((cur - ref) ** 2)
        diffs.append(d)
    diffs = np.array(diffs)

    # Ignore early frames (trivially similar); find first strong minimum
    start = min_separation
    if start >= len(diffs):
        # fallback: just use the min overall
        idx = int(np.argmin(diffs))
    else:
        idx = int(np.argmin(diffs[start:]) + start)

    period_frames = idx + 1  # because diffs[0] is frame 1 vs 0
    return period_frames, diffs

def infer_rotation_direction(flow):
    """
    flow: (Hc, Wc, 2)

    Returns:
      dir_sign: +1 for "camera moving right", -1 for left.
    """
    mean_dx = np.mean(flow[..., 0])
    if mean_dx >= 0:
        return +1.0
    else:
        return -1.0

def depth_from_flow(
    flow,
    frame_width,
    period_frames,
    fps,
    rot_dir_sign,
    eps=1e-6,
):
    """
    flow: (Hc, Wc, 2) affine motion (dx, dy) in pixels from frame t -> t+1
    frame_width: original video width in pixels
    period_frames: orbit period in frames
    fps: frames per second (float)
    rot_dir_sign: +1 or -1 (from infer_rotation_direction)

    Returns:
      depth: (Hc, Wc) float map
    """
    # Angular speed (rad/s); if fps unknown, we still get a consistent scale
    if fps <= 0:
        fps = 1.0
    T_sec = period_frames / fps
    omega = 2.0 * np.pi / max(T_sec, eps)  # rad/s

    # Project flow onto rotation direction (assume horizontal orbit)
    dx = flow[..., 0]  # (Hc, Wc)
    aligned = rot_dir_sign * dx  # positive = with rotation

    # Normalize by max possible motion (≈ frame width)
    max_pix = max(frame_width, 1)
    m_norm = aligned / max_pix  # [-?, 1] ideally

    # Depth rule:
    #  - aligned == 0  -> depth ~ 1
    #  - aligned == max_pix -> depth ~ 0
    #  - opposite motion => depth > 1
    #  - scale falls with larger |omega|
    depth = 1.0 - (m_norm / max(abs(omega), eps))

    return depth

