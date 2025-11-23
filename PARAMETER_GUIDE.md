# Parameter Guide

Complete reference for all command-line parameters in `video_orbit_voxel_recon.py`.

## Quick Start

```bash
# Basic usage (auto-detect orbit, sensible defaults)
python video_orbit_voxel_recon.py video.mp4 0

# Get help
python video_orbit_voxel_recon.py --help
```

## Positional Arguments

### `video_path`
Path to input video file.

**Example:** `video.mp4`

### `start_frame`
Starting frame index (0-based). The script will auto-detect orbit period from this point.

**Example:** `0` (start from beginning)

---

## Volume Settings

### `--grid-size N`
Voxel grid resolution (N×N×N).

**Default:**
- `128` for single GPU (standard mode)
- `512` for multi-GPU (sharded mode)

**Typical values:**
- `64` - fast prototyping, low quality
- `128` - standard quality (default single GPU)
- `256` - high quality (needs ~2 GB memory)
- `512` - very high quality (sharded: 256³ per GPU)
- `1024` - ultra quality (sharded: 512³ per GPU, needs 80GB+ per GPU)

**Examples:**
```bash
# Small grid for quick testing
python video_orbit_voxel_recon.py video.mp4 0 --grid-size 64

# High-res sharded across 8 GPUs
python video_orbit_voxel_recon.py video.mp4 0 --sharded --grid-size 768
```

### `--scene-radius RADIUS`
Scene radius in world units. The voxel grid extends from -RADIUS to +RADIUS in each dimension.

**Default:** `1.5`

**When to adjust:**
- Object too large for grid: increase radius
- Object too small (wasting voxels): decrease radius
- Camera distance affects optimal radius

**Example:**
```bash
python video_orbit_voxel_recon.py video.mp4 0 --scene-radius 2.0
```

---

## Frame Settings

### `--img-res WIDTH HEIGHT`
Frame resolution for training. Higher resolution = better quality but slower.

**Default:** `256 256`

**Typical values:**
- `64 64` - very fast, low quality (debugging)
- `128 128` - fast, decent quality
- `256 256` - balanced (default)
- `512 512` - high quality, slower
- `1024 1024` - ultra quality, very slow

**Memory impact:** Minimal (~1-100 MB total)
**Speed impact:** Quadratic (4× pixels = 4× slower)

**Examples:**
```bash
# Low-res for quick iteration
python video_orbit_voxel_recon.py video.mp4 0 --img-res 128 128

# High-res for final quality
python video_orbit_voxel_recon.py video.mp4 0 --img-res 512 512
```

### `--frame-step N`
Use every Nth frame. Reduces training data for faster iterations.

**Default:** `1` (use all frames)

**When to use:**
- `1` - all frames, best quality (default)
- `2` - every 2nd frame, 2× faster, slight quality loss
- `4` - every 4th frame, 4× faster, noticeable quality loss

**Example:**
```bash
# Fast iteration during development
python video_orbit_voxel_recon.py video.mp4 0 --frame-step 2
```

### `--n-samples N`
Number of samples per ray for volumetric rendering.

**Default:** `64`

**Typical values:**
- `32` - fast, may miss thin features
- `64` - balanced (default)
- `128` - slow, better for thin/detailed objects

**Trade-off:** More samples = slower but better geometry capture

**Example:**
```bash
# More samples for detailed object
python video_orbit_voxel_recon.py video.mp4 0 --n-samples 128
```

### `--fov DEGREES`
Camera field of view in degrees.

**Default:** `45.0`

**When to adjust:**
- Match your actual camera's FOV if known
- Wider FOV (60°-90°) for wide-angle cameras
- Narrower FOV (30°-45°) for telephoto

**Example:**
```bash
python video_orbit_voxel_recon.py video.mp4 0 --fov 60
```

---

## Training Settings

### `--n-iters N`
Number of training iterations.

**Default:** `8000`

**Guidelines:**
- `2000-4000` - quick test
- `8000` - standard quality (default)
- `16000` - high quality, diminishing returns
- `32000+` - research/perfectionism

**Training time:** ~0.5-2 seconds per iteration depending on resolution

**Examples:**
```bash
# Quick test
python video_orbit_voxel_recon.py video.mp4 0 --n-iters 2000

# High quality
python video_orbit_voxel_recon.py video.mp4 0 --n-iters 16000
```

### `--lambda-l1 VALUE`
L1 sparsity regularization weight. Penalizes total density to prevent "clouds."

**Default:** `0.03`

**Effect:**
- Higher (0.05-0.1) = more aggressive sparsity, compact object
- Lower (0.01-0.02) = allow more density, may get noisy clouds
- Zero (0.0) = no sparsity, will get lots of spurious voxels

**Symptoms:**
- Too high: object disappears or has holes
- Too low: clouds of white voxels, hollow interiors

**Examples:**
```bash
# Aggressive sparsity (clean but may lose detail)
python video_orbit_voxel_recon.py video.mp4 0 --lambda-l1 0.05

# Gentle sparsity (may need filtering later)
python video_orbit_voxel_recon.py video.mp4 0 --lambda-l1 0.01
```

### `--lambda-tv-sigma VALUE`
Total Variation smoothness weight for density (sigma).

**Default:** `0.002`

**Effect:**
- Higher (0.005-0.01) = smoother surfaces, less noise
- Lower (0.0005-0.001) = allow more geometric detail
- Zero (0.0) = no smoothness, noisy surfaces

**Trade-off:** Smoothness vs fine detail

**Example:**
```bash
# Very smooth surfaces
python video_orbit_voxel_recon.py video.mp4 0 --lambda-tv-sigma 0.005
```

### `--lambda-tv-rgb VALUE`
Total Variation smoothness weight for color (RGB).

**Default:** `0.001`

**Effect:**
- Higher (0.002-0.005) = smoother colors, less noise
- Lower (0.0001-0.0005) = more color variation
- Zero (0.0) = no color smoothness

**Example:**
```bash
# Smooth colors
python video_orbit_voxel_recon.py video.mp4 0 --lambda-tv-rgb 0.002
```

---

## Reconstruction Modes

### `--neighbor-growth`
Use neighbor-based growth mode (organic expansion from center seed).

**Default:** Hard core mode (expanding radial constraint)

**When to use:**
- Neighbor growth: experimental, may grow unevenly
- Hard core (default): reliable, expands from center uniformly

**Example:**
```bash
python video_orbit_voxel_recon.py video.mp4 0 --neighbor-growth
```

### `--sharded`
Use sharded volume across 8 GPUs for massive resolution.

**Default:** Single GPU mode

**Requirements:**
- Exactly 8 GPUs
- Each GPU needs enough memory for (grid_size/2)³ shard

**Benefits:**
- Enables 512³ - 2048³ grids
- Distributes memory across GPUs

**Example:**
```bash
# 512³ sharded (256³ per GPU)
python video_orbit_voxel_recon.py video.mp4 0 --sharded

# 1024³ sharded (512³ per GPU, needs A100 80GB or B200)
python video_orbit_voxel_recon.py video.mp4 0 --sharded --grid-size 1024
```

---

## Visualization

### `--viewer`
Enable live pygame viewer during training.

**Default:** Disabled

**Note:** Only works on systems with display. Disabled on remote servers.

**Controls:**
- Arrow keys: rotate view
- +/- : zoom
- C/V: switch between cube and volume rendering
- SPACE: pause rotation

**Example:**
```bash
python video_orbit_voxel_recon.py video.mp4 0 --viewer
```

---

## Output

### `--out-dir PATH`
Output directory for results.

**Default:** `video_voxel_out`

**Contents:**
- `recon_volume.npz` - Full voxel grid (sigma + RGB)
- `recon_voxels.ply` - Point cloud PLY file
- `recon_000.png` - Reconstructed views
- `debug_*.png` - Debug visualizations

**Example:**
```bash
python video_orbit_voxel_recon.py video.mp4 0 --out-dir my_reconstruction
```

---

## Common Workflows

### Quick Test (Fast)
```bash
python video_orbit_voxel_recon.py video.mp4 0 \
  --grid-size 64 \
  --img-res 128 128 \
  --n-iters 2000 \
  --frame-step 2
```
**Time:** ~5 minutes, low quality

### Standard Quality (Default)
```bash
python video_orbit_voxel_recon.py video.mp4 0
```
**Time:** ~1-2 hours single GPU, good quality

### High Quality (Single GPU)
```bash
python video_orbit_voxel_recon.py video.mp4 0 \
  --grid-size 256 \
  --img-res 512 512 \
  --n-iters 16000 \
  --lambda-l1 0.04
```
**Time:** ~6-12 hours, high quality

### Ultra Quality (8 GPUs Sharded)
```bash
python video_orbit_voxel_recon.py video.mp4 0 \
  --sharded \
  --grid-size 768 \
  --img-res 512 512 \
  --n-iters 16000 \
  --lambda-l1 0.04
```
**Time:** ~2-4 hours, ultra quality

### Debugging Bad Results

**Problem: Clouds of white voxels**
```bash
# Increase sparsity
python video_orbit_voxel_recon.py video.mp4 0 --lambda-l1 0.05
```

**Problem: Object has holes**
```bash
# Decrease sparsity
python video_orbit_voxel_recon.py video.mp4 0 --lambda-l1 0.01
```

**Problem: Noisy surfaces**
```bash
# Increase smoothness
python video_orbit_voxel_recon.py video.mp4 0 \
  --lambda-tv-sigma 0.005 \
  --lambda-tv-rgb 0.002
```

**Problem: Too smooth, missing details**
```bash
# Decrease smoothness
python video_orbit_voxel_recon.py video.mp4 0 \
  --lambda-tv-sigma 0.001 \
  --lambda-tv-rgb 0.0005
```

---

## Performance Tips

### Speed vs Quality Trade-offs

**Fastest to Slowest:**
1. Grid size: 64³ < 128³ < 256³ < 512³
2. Frame resolution: 64² < 128² < 256² < 512²
3. Frame step: 4 < 2 < 1
4. Samples per ray: 32 < 64 < 128
5. Iterations: 2000 < 8000 < 16000

**Biggest speedups:**
- Reduce frame resolution (quadratic impact)
- Increase frame step (linear speedup)
- Reduce iterations (linear speedup)

**Biggest quality improvements:**
- Increase grid size (more detail)
- Increase frame resolution (better supervision)
- Tune regularization (cleaner results)

