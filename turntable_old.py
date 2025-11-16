# pip install --upgrade openai pillow imageio
import os, time, base64, imageio.v3 as iio
from pathlib import Path
from openai import OpenAI

# -----------------------
# CONFIG
# -----------------------
MODEL             = "gpt-image-1"         # image generation model
REF_IMAGE_PATH    = "reference.png"       # your extracted PNG (transparent background)
OUT_DIR           = Path("turntable_out") # output folder
SIZE              = "1024x1024"
N_FRAMES          = 35                    # 0..34 -> 0..340 deg
DEG_PER_STEP      = 10
RETRIES           = 4                     # retry on transient errors / rate limits
BACKGROUND        = "transparent"         # ensure alpha

# -----------------------
# PROMPT ENGINEERING
# -----------------------
# Core instructions written to push for 3D consistency and avoid “new content” drift.
BASE_INSTRUCTIONS = """
You are producing a single frame of a 360-degree 3D turntable.
Keep the boy, purple cat, and glowing cubes exactly as in the reference image:
- same character identity, proportions, pose, clothing, hairstyle, and facial expression
- same cube arrangement relative to the characters
The camera orbits the scene horizontally (around the vertical axis through the scene center).
This is not a 2D warp: render a plausible 3D view of the same scene.

Constraints:
- Output a single PNG with a fully transparent background (RGBA with alpha).
- No text, no extra props, no glow/fringe beyond silhouettes (crisp edges).
- Keep camera distance and framing consistent across frames (medium shot).
- Lighting is fixed in world coordinates (key light from +X, fill from +Z).
- Do not change expression or pose between frames.
"""

FRAME_TEMPLATE = """
TURNtable frame {idx}/{total} at camera azimuth = {azimuth} degrees (relative to the reference "front" view).
Rotate the camera by +{azimuth}° around the vertical axis, looking at the same scene center.
The scene itself remains fixed in world space; only the camera position changes.
"""

# -----------------------
# API CLIENT
# -----------------------
client = OpenAI()

# -----------------------
# IO / HELPERS
# -----------------------
def ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_ref_image_bytes():
    with open(REF_IMAGE_PATH, "rb") as f:
        return f.read()

def save_png_b64(path: Path, b64: str):
    binary = base64.b64decode(b64)
    path.write_bytes(binary)

# -----------------------
# GENERATION LOOP
# -----------------------
def generate_turntable():
    ensure_outdir()
    ref_bytes = load_ref_image_bytes()

    for i in range(N_FRAMES):
        angle = i * DEG_PER_STEP
        prompt = BASE_INSTRUCTIONS + FRAME_TEMPLATE.format(
            idx=i+1, total=N_FRAMES, azimuth=angle
        )

        out_path = OUT_DIR / f"turntable_{angle:03d}.png"
        if out_path.exists():
            print(f"[skip] {out_path}")
            continue

        for attempt in range(1, RETRIES+1):
            try:
                resp = client.images.generate(
                    model=MODEL,
                    prompt=prompt.strip(),
                    size=SIZE,
                    background=BACKGROUND,
                    # Provide the reference PNG so the model keeps identity & pose
                    image=[
                        {"name": Path(REF_IMAGE_PATH).name, "bytes": ref_bytes}
                    ],
                    # quality="high",  # uncomment if supported in your SDK version
                    n=1,
                )
                b64 = resp.data[0].b64_json
                save_png_b64(out_path, b64)
                print(f"[ok] {out_path}")
                break

            except Exception as e:
                if attempt == RETRIES:
                    raise
                wait = 2 ** attempt
                print(f"[retry {attempt}] {e} -> sleeping {wait}s")
                time.sleep(wait)

# -----------------------
# OPTIONAL: MAKE A GIF
# -----------------------
def make_gif(fps=12, gif_name="turntable.gif"):
    frames = []
    for i in range(N_FRAMES):
        angle = i * DEG_PER_STEP
        p = OUT_DIR / f"turntable_{angle:03d}.png"
        if p.exists():
            frames.append(iio.imread(p))
    if frames:
        iio.imwrite(OUT_DIR / gif_name, frames, loop=0, fps=fps)
        print(f"[gif] {OUT_DIR / gif_name}")

if __name__ == "__main__":
    assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your environment."
    assert Path(REF_IMAGE_PATH).exists(), f"Missing {REF_IMAGE_PATH}"
    generate_turntable()
    # make_gif()  # optional

