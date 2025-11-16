# turntable_multi_edit.py
import os, time, base64, io
from pathlib import Path
from typing import List, Optional
from openai import OpenAI
from PIL import Image, ImageChops, ImageStat
from contextlib import nullcontext
from pathlib import Path
import io

# -----------------------
# CONFIG
# -----------------------
MODEL              = "gpt-image-1"          # images.edit-capable model
REF_IMAGE_PATH     = "reference.png"        # transparent PNG (boy+cat+cubes)
OUT_DIR            = Path("turntable_out")  # output folder
SIZE               = "1024x1024"
N_FRAMES           = 35                     # total including the 0° reference
DEG_PER_STEP       = 10                     # 10° increments → 35 frames = 360°
USE_TEMPORAL_HINT  = True                   # also pass previous frame as 2nd image
BACKGROUND_ARG     = "transparent"          # if SDK rejects, set to None
RETRIES            = 4

# Optional: also emit silhouettes for voxel carving (solid black on transparent)
EMIT_SILHOUETTES   = False
SILH_DIR           = Path("turntable_silhouettes")

# -----------------------
# PROMPTS
# -----------------------
BASE_INSTRUCTIONS = """
You are producing a single frame of a 360-degree 3D camera turntable around a fixed scene.
Keep the character, purple cat, and glowing cubes exactly matching the reference:
- same identity, proportions, hairstyle, clothing, facial expression, pose
- same relative positions of cubes and cat to the character
The camera orbits horizontally around the vertical axis through the scene center.
This is not a 2D warp: render a plausible 3D view of the same scene.

Constraints:
- Output must be a single PNG with a fully transparent background (RGBA with alpha).
- No text or extra elements. No glow fringes; edges should be crisp.
- Keep camera distance and framing consistent across all frames (medium shot).
- Lighting is fixed in world coordinates (key light from +X, fill from +Z).
- Do not alter expression, pose, clothing, or proportions between frames.
"""

FRAME_TEMPLATE = """
Turntable frame {idx}/{total} at camera azimuth = {azimuth} degrees (relative to the reference "front" view).
Rotate the camera by +{azimuth}° around the vertical axis, always looking at the same scene center.
The scene remains fixed; only the camera position changes.
"""

SILH_PROMPT_EXTRA = """
Override output appearance: render the subject as a solid black silhouette with fully opaque fill on a transparent background.
Keep edges tight, with no feathering or glow.
"""

# -----------------------
# UTILITIES
# -----------------------
client = OpenAI()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_png_b64(path: Path, b64: str):
    path.write_bytes(base64.b64decode(b64))

def force_transparency_if_white_bg(img: Image.Image) -> Image.Image:
    """Turn near-white background fully transparent, keep subject opaque."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    # Compare against pure white to find non-white foreground
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    diff = ImageChops.difference(img, bg).convert("L")
    # Threshold: lower -> more aggressive knockout of white; tweak 4..15 as needed
    alpha = diff.point(lambda p: 0 if p < 6 else 255)
    img.putalpha(alpha)
    return img

# -----------------------
# GENERATION
# -----------------------
def generate_turntable():
    ensure_dir(OUT_DIR)
    if EMIT_SILHOUETTES:
        ensure_dir(SILH_DIR)

    prev_frame_path = None  # track path of previous frame on disk

    for i in range(1, N_FRAMES):
        angle = i * DEG_PER_STEP
        out_path = OUT_DIR / f"turntable_{angle:03d}.png"
        if out_path.exists():
            print(f"[skip] {out_path}")
            if USE_TEMPORAL_HINT:
                prev_frame_path = out_path
            continue

        prompt = (BASE_INSTRUCTIONS + FRAME_TEMPLATE.format(
            idx=i+1, total=N_FRAMES, azimuth=angle
        )).strip()

        # Use real files so the SDK can set the correct mimetype (image/png)
        with open(REF_IMAGE_PATH, "rb") as f0, \
             (open(prev_frame_path, "rb") if (USE_TEMPORAL_HINT and prev_frame_path) else nullcontext(None)) as f1:

            images_arg = [f0] + ([f1] if f1 else [])

            for attempt in range(1, RETRIES+1):
                try:
                    # Try modern edits.create; fall back to edit if needed
                    kwargs = dict(
                        model=MODEL,
                        prompt=prompt,
                        size=SIZE,
                        n=1,
                        image=images_arg,                 # <-- FILE HANDLES, not bytes
                    )
                    if BACKGROUND_ARG:
                        kwargs["background"] = BACKGROUND_ARG

                    try:
                        resp = client.images.edits.create(**kwargs)  # new-style
                    except AttributeError:
                        resp = client.images.edit(**kwargs)          # legacy-style

                    b64 = resp.data[0].b64_json
                    save_png_b64(out_path, b64)
                    print(f"[ok] {out_path}")

                    # Optional: enforce transparency if SDK ignored background
                    im = Image.open(out_path).convert("RGBA")
                    has_alpha = ("A" in im.getbands())
                    mean_rgb = ImageStat.Stat(im.convert("RGB")).mean
                    if (not has_alpha) or all(ch > 250 for ch in mean_rgb):
                        im = force_transparency_if_white_bg(im)
                        im.save(out_path)

                    if USE_TEMPORAL_HINT:
                        prev_frame_path = out_path  # next step's temporal hint
                    break

                except Exception as e:
                    if attempt == RETRIES:
                        raise
                    wait = 2 ** attempt
                    print(f"[retry {attempt}] {e} -> sleeping {wait}s")
                    time.sleep(wait)

if __name__ == "__main__":
    assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your environment."
    assert Path(REF_IMAGE_PATH).exists(), f"Missing {REF_IMAGE_PATH}"
    generate_turntable()
