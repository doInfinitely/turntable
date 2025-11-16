import cv2
import numpy as np
from pinhole_depth import estimate_depth_from_flow
from orbit_depth_utils import estimate_orbit_period, load_grayscale_frames

def run(video_path, frame_idx, f=1200, R=1.0):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, f1 = cap.read()
    ret, f2 = cap.read()

    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    # Far superior optical flow (much smoother than affine sampling)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=25,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )

    H, W = g1.shape
    u_coords, v_coords = np.meshgrid(
        np.arange(W) - W/2,
        np.arange(H) - H/2
    )

    # estimate orbit period (your previous function)
    frames = load_grayscale_frames(video_path, max_frames=600)
    orbit_period_frames = estimate_orbit_period(frames)[0]
    orbit_period_seconds = orbit_period_frames / fps
    omega = 2*np.pi / orbit_period_seconds

    Z = estimate_depth_from_flow(
        flow,
        u_coords,
        v_coords,
        f=f, R=R,
        dt=1/fps,
        t=(frame_idx/ orbit_period_frames) * 2*np.pi,
    )

    Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-6)
    depth_vis = cv2.applyColorMap((Z_norm*255).astype(np.uint8), cv2.COLORMAP_MAGMA)

    cv2.imwrite("depth_pinhole.png", depth_vis)
    print("Saved depth_pinhole.png")

if __name__ == "__main__":
    run("/Users/remy/Code/turntable/20251115_111317_minimax_hailuo-2.3_Camera_movement_the_camera_or.mp4", 120)
