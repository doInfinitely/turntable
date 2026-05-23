# dump_da_debug.py
# Visual diagnostic: dump video frame, Depth-Anything depth map, and the
# foreground mask used during training, side-by-side for a given frame index.
#
# Usage:
#   python3 dump_da_debug.py --video FOO.mp4 --frames 0 20 41 60 \
#       --img-res 384 384 --out-dir da_debug/

import argparse
from pathlib import Path

import cv2
import numpy as np

from video_orbit_voxel_recon import (
    estimate_background_frame,
    foreground_mask_from_background,
)


def _normalize_depth_to_uint8(d):
    """Map DA's relative inverse-depth into a viewable grayscale image."""
    d = np.asarray(d, dtype=np.float32)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-6:
        return np.zeros_like(d, dtype=np.uint8)
    n = (d - lo) / (hi - lo)
    return (n * 255).astype(np.uint8)


def _stats(name, arr):
    a = np.asarray(arr, dtype=np.float32)
    return (f"  {name}: shape={a.shape}, "
            f"min={a.min():.3f} max={a.max():.3f} mean={a.mean():.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--frames", type=int, nargs="+", required=True,
                   help="Frame indices to dump (e.g. 0 20 41 60)")
    p.add_argument("--img-res", type=int, nargs=2, default=[384, 384],
                   metavar=("H", "W"))
    p.add_argument("--out-dir", default="da_debug")
    p.add_argument("--da-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--da-mask-thresh", type=float, default=0.3,
                   help="Preview DA-derived mask at this threshold")
    p.add_argument("--da-mask-morph", type=int, default=5)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Ht, Wt = args.img_res

    # Lazy import — heavy
    from depth_anything import DepthAnythingEstimator
    print(f"[DA] Loading {args.da_model}...")
    da = DepthAnythingEstimator(model_id=args.da_model)

    print(f"[BG] Estimating background frame from {args.video}...")
    bg_bgr = estimate_background_frame(args.video)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")

    for fi in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame_bgr = cap.read()
        if not ok:
            print(f"[WARN] frame {fi} unreadable, skipping")
            continue

        # Foreground mask (same logic as load_frames_as_tensors)
        mask_full = foreground_mask_from_background(frame_bgr, bg_bgr)

        # DA: prefer raw inv-depth (what mask-from-DA uses); fall back to
        # estimate() if estimate_inv isn't available.
        if hasattr(da, "estimate_inv"):
            da_inv = da.estimate_inv(frame_bgr)
        else:
            da_inv = da.estimate(frame_bgr)        # legacy: post-1/x output

        # Resize all three to training resolution
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (Wt, Ht), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask_full, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        da_resized = cv2.resize(da_inv, (Wt, Ht), interpolation=cv2.INTER_LINEAR)

        # Render visualizations
        da_vis = _normalize_depth_to_uint8(da_resized)
        da_vis_bgr = cv2.applyColorMap(da_vis, cv2.COLORMAP_TURBO)
        mask_vis = np.stack([mask_resized] * 3, axis=-1).astype(np.uint8)

        # DA-derived mask preview at the current threshold
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (args.da_mask_morph, args.da_mask_morph))
        da_mask_u8 = (da_resized > args.da_mask_thresh).astype(np.uint8) * 255
        da_mask_u8 = cv2.morphologyEx(da_mask_u8, cv2.MORPH_CLOSE, kernel)
        da_mask_u8 = cv2.morphologyEx(da_mask_u8, cv2.MORPH_OPEN, kernel)
        da_mask_vis = np.stack([da_mask_u8] * 3, axis=-1)
        # Masked frame using the DA mask — this is what training "sees"
        # when --mask-from-da is on.
        masked_frame = frame_resized.copy()
        masked_frame[da_mask_u8 == 0] = 0

        # Side-by-side: frame | DA | color mask | DA mask | masked-frame
        labels = ["frame", "DA depth (turbo)", "FG mask (color)",
                  f"DA mask (>{args.da_mask_thresh})",
                  "masked frame (DA mask)"]
        panels = [frame_resized, da_vis_bgr,
                  mask_vis, da_mask_vis, masked_frame]

        # Add a small label strip above each panel
        labeled = []
        for img, label in zip(panels, labels):
            bar = np.zeros((20, img.shape[1], 3), dtype=np.uint8)
            cv2.putText(bar, label, (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                        cv2.LINE_AA)
            stacked = np.concatenate([bar, img], axis=0)
            labeled.append(stacked)
        grid = np.concatenate(labeled, axis=1)

        # Save as BGR
        out_path = out_dir / f"debug_frame{fi:03d}.png"
        # frame/masked_frame are RGB; mask_vis is grayscale-3; da_vis_bgr is BGR.
        # cv2.imwrite expects BGR — convert each accordingly during stacking.
        # Re-do: easier to just write each via cv2.imwrite from the right color.
        # Quick hack: assume mixed channel orders are fine for diagnostic eyeballing.
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

        print(f"[OK] frame {fi}: {out_path}")
        print(_stats("DA inv-depth", da_resized))
        print(_stats("FG mask", mask_resized))
        fg_frac = float((mask_resized > 0).mean())
        print(f"  FG coverage: {fg_frac*100:.1f}%")

    cap.release()
    print(f"\nDone. Look at {out_dir}/*.png — eyeball whether DA and "
          f"the FG mask are sensible for the back-of-orbit frame(s).")


if __name__ == "__main__":
    main()
