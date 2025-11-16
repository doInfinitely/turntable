# example_usage.py
import cv2
import sys
import numpy as np

from affine_vector_field import dense_motion_field


def extract_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    return frame


def visualize_flow(flow, base_img, stride=16):
    vis = base_img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    h_cells, w_cells, _ = flow.shape

    for i in range(h_cells):
        for j in range(w_cells):
            dx, dy = flow[i, j]
            y = i * stride + stride // 2
            x = j * stride + stride // 2

            cv2.arrowedLine(
                vis,
                (x, y),
                (int(x + dx), int(y + dy)),
                (0, 255, 0),
                1,
                tipLength=0.3
            )

    return vis


def run_vector_field(video_path, start_frame_idx):
    print(f"[INFO] Loading frames {start_frame_idx} and {start_frame_idx + 1}...")

    frame1 = extract_frame(video_path, start_frame_idx)
    frame2 = extract_frame(video_path, start_frame_idx + 1)

    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    print("[INFO] Computing affine vector field...")
    flow, errors, affine_mats = dense_motion_field(
        g1,
        g2,
        window=32,
        stride=16,
        search=16,
        scales=[1.0],
        angles=[-5, 0, 5],
        shears=[0],
    )

    print("[INFO] Flow field shape:", flow.shape)

    vis = visualize_flow(flow, g1, stride=16)

    out_path = "flow_visualization.png"
    cv2.imwrite(out_path, vis)
    print(f"[INFO] Saved vector field visualization → {out_path}")

    cv2.imshow("Vector Field", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return flow, errors, affine_mats


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python example_usage.py <video_path> <start_frame_idx>")
        sys.exit(1)

    video_path = sys.argv[1]
    start_frame_idx = int(sys.argv[2])

    run_vector_field(video_path, start_frame_idx)

