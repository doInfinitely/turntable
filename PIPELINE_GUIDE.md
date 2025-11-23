# Voxel Processing Pipeline Guide

The voxel pipeline orchestrator (`voxel_pipeline.py`) allows you to chain together multiple processing steps into a single automated workflow.

## Overview

The pipeline supports three types of operations:
1. **reconstruct** - 3D reconstruction from orbital video
2. **subdivide** - Increase voxel grid resolution by subdividing voxels
3. **harden** - Sharpen density distribution using differentiable flow-based hardening

## Quick Start

### 1. Generate Example Config
```bash
python voxel_pipeline.py --create-example-config my_pipeline.json
```

### 2. Edit Config for Your Video
```json
{
  "input": {
    "type": "video",
    "path": "my_video.mp4"
  },
  "output": "final_output.npz",
  "steps": [
    {"type": "reconstruct", "params": {"grid_size": 64, "n_iters": 8000}},
    {"type": "subdivide", "params": {"subdivision": 2}},
    {"type": "harden", "params": {"n_iters": 1000}}
  ]
}
```

### 3. Run Pipeline
```bash
python voxel_pipeline.py --config my_pipeline.json
```

## Configuration Reference

### Input Types

#### Video Input
Start from an orbital video (requires reconstruction as first step):
```json
{
  "input": {
    "type": "video",
    "path": "video.mp4"
  }
}
```

#### NPZ Input
Start from an existing voxel volume:
```json
{
  "input": {
    "type": "npz",
    "path": "existing_volume.npz"
  }
}
```

#### Video + Checkpoint Input
Resume reconstruction from a checkpoint (continue training on same video):
```json
{
  "input": {
    "type": "video_with_checkpoint",
    "video_path": "video.mp4",
    "checkpoint_path": "previous_recon_volume.npz"
  }
}
```
The first step must be `reconstruct`. The checkpoint's voxel grid will be loaded as the initial state, and reconstruction will continue from there. This is useful for:
- Continuing interrupted training
- Fine-tuning with different regularization weights
- Training for additional iterations

### Step Types

#### Reconstruct Step

Performs 3D reconstruction from orbital video.

**Common Parameters:**
```json
{
  "type": "reconstruct",
  "params": {
    "video_path": "video.mp4",      // Optional if input is video
    "checkpoint_path": "chk.npz",   // Optional: resume from checkpoint
    "start_frame": 0,                // Starting frame index
    "frame_step": 1,                 // Use every Nth frame
    "grid_size": 128,                // Voxel grid resolution
    "n_iters": 8000,                 // Training iterations
    "img_res": [256, 256],           // Frame resolution
    "n_samples": 64,                 // Samples per ray
    "scene_radius": 1.5,             // Scene radius in world units
    "fov": 45.0,                     // Field of view (degrees)
    
    // Regularization weights
    "lambda_l1": 0.03,               // L1 sparsity
    "lambda_tv_sigma": 0.002,        // TV smoothness (density)
    "lambda_tv_rgb": 0.001,          // TV smoothness (color)
    
    // Mode flags
    "neighbor_growth": false,        // Use neighbor growth mode
    "sharded": false,                // Use multi-GPU sharded mode
    "viewer": false,                 // Enable live viewer
    
    // Background removal
    "openai_bg_removal": false,      // Use OpenAI for bg removal
    "openai_api_key": null           // API key (or use env var)
  }
}
```

**Example - Basic reconstruction:**
```json
{
  "type": "reconstruct",
  "params": {
    "grid_size": 64,
    "n_iters": 8000
  }
}
```

**Example - High quality:**
```json
{
  "type": "reconstruct",
  "params": {
    "grid_size": 128,
    "n_iters": 12000,
    "img_res": [512, 512],
    "n_samples": 96
  }
}
```

#### Subdivide Step

Increases voxel resolution by splitting each voxel into N³ smaller voxels.

**Parameters:**
```json
{
  "type": "subdivide",
  "params": {
    "subdivision": 2  // Each voxel becomes 2³=8 voxels
  }
}
```

**Examples:**
- `subdivision: 2` → 64³ becomes 128³
- `subdivision: 3` → 64³ becomes 192³
- `subdivision: 4` → 64³ becomes 256³

#### Harden Step

Sharpens the density distribution using flow-based optimization.

**Parameters:**
```json
{
  "type": "harden",
  "params": {
    "n_iters": 1000,                 // Optimization iterations
    "flow_lr": 0.01,                 // Learning rate for flows
    "variance_penalty": 0.1,         // Variance reward weight
    "flow_reg": 0.001,               // Flow regularization
    "n_views": 8,                    // Views for recon loss
    "img_res": [64, 64],             // Render resolution
    "n_samples": 64,                 // Samples per ray
    "scene_radius": 1.5,             // Scene radius
    "fov": 45.0,                     // Field of view
    "use_local_variance": true,      // Local variance penalty
    "use_global_variance": true      // Global variance penalty
  }
}
```

**Example - Aggressive hardening:**
```json
{
  "type": "harden",
  "params": {
    "n_iters": 2000,
    "variance_penalty": 0.3,
    "flow_lr": 0.005
  }
}
```

## Example Pipelines

### Basic Pipeline
Reconstruct → Subdivide → Harden
```json
{
  "input": {"type": "video", "path": "video.mp4"},
  "output": "result.npz",
  "steps": [
    {"type": "reconstruct", "params": {"grid_size": 64, "n_iters": 8000}},
    {"type": "subdivide", "params": {"subdivision": 2}},
    {"type": "harden", "params": {"n_iters": 1000}}
  ]
}
```

### Resume from Checkpoint
Continue training from a previous reconstruction:
```json
{
  "input": {
    "type": "video_with_checkpoint",
    "video_path": "video.mp4",
    "checkpoint_path": "video_voxel_out/recon_volume.npz"
  },
  "output": "continued.npz",
  "steps": [
    {
      "type": "reconstruct",
      "params": {
        "n_iters": 4000,
        "lambda_l1": 0.02
      }
    }
  ]
}
```

### From Existing NPZ
Process an existing volume:
```json
{
  "input": {"type": "npz", "path": "existing.npz"},
  "output": "processed.npz",
  "steps": [
    {"type": "subdivide", "params": {"subdivision": 2}},
    {"type": "harden", "params": {"n_iters": 1000}}
  ]
}
```

### Iterative Hardening
Harden → Subdivide → Harden → Subdivide → Harden
```json
{
  "input": {"type": "video", "path": "video.mp4"},
  "output": "super_crisp.npz",
  "steps": [
    {"type": "reconstruct", "params": {"grid_size": 64}},
    {"type": "harden", "params": {"n_iters": 500, "variance_penalty": 0.1}},
    {"type": "subdivide", "params": {"subdivision": 2}},
    {"type": "harden", "params": {"n_iters": 1000, "variance_penalty": 0.2}},
    {"type": "subdivide", "params": {"subdivision": 2}},
    {"type": "harden", "params": {"n_iters": 1500, "variance_penalty": 0.3}}
  ]
}
```
This creates a 256³ volume with progressively sharper details.

### High Resolution Pipeline
For maximum quality:
```json
{
  "input": {"type": "video", "path": "video.mp4"},
  "output": "high_res.npz",
  "steps": [
    {
      "type": "reconstruct",
      "params": {
        "grid_size": 128,
        "n_iters": 12000,
        "img_res": [512, 512],
        "n_samples": 96
      }
    },
    {
      "type": "harden",
      "params": {
        "n_iters": 2000,
        "variance_penalty": 0.2
      }
    }
  ]
}
```

### Checkpoint + Refine + Process
Resume from checkpoint, then subdivide and harden:
```json
{
  "input": {
    "type": "video_with_checkpoint",
    "video_path": "video.mp4",
    "checkpoint_path": "video_voxel_out/recon_volume.npz"
  },
  "output": "refined_final.npz",
  "steps": [
    {"type": "reconstruct", "params": {"n_iters": 4000}},
    {"type": "subdivide", "params": {"subdivision": 2}},
    {"type": "harden", "params": {"n_iters": 1500}}
  ]
}
```

## GPU Management

The pipeline automatically manages GPU usage:

### Single GPU
- Standard operation for grids up to 128³
- All processing on GPU 0

### Multi-GPU Rendering
- **Automatically enabled** for high-res frames (≥128×128 pixels)
- Splits rendering across all available GPUs
- Significant speedup for large frame resolutions

### Sharded Mode
- **Automatically enabled** for massive grids (≥256³ on 8 GPUs)
- Distributes volume across multiple GPUs
- Enables reconstruction at 512³ or higher

**Manual override:**
```json
{
  "type": "reconstruct",
  "params": {
    "grid_size": 512,
    "sharded": true  // Force sharded mode
  }
}
```

## Working Directory

The pipeline creates a temporary working directory for intermediate files:
```
/tmp/voxel_pipeline_XXXXXX/
  step_01_reconstruct/
    recon_volume.npz
    recon_000.png
    ...
  step_02_subdivide.npz
  step_03_harden/
    hardened_final.npz
    ...
```

Files are automatically cleaned up after completion. Only the final output is saved to your specified path.

## Monitoring Progress

Each step prints detailed progress:
```
======================================================================
STEP 1: RECONSTRUCTION
======================================================================
Video: video.mp4
Grid size: 64³
Iterations: 8000
...
[0/8000] loss=0.123456 ...
[10/8000] loss=0.098765 ...
...
✓ Reconstruction complete: /tmp/.../step_01_reconstruct/recon_volume.npz

======================================================================
STEP 2: SUBDIVISION
======================================================================
Current size: 64×64×64
New size: 128×128×128
...
✓ Subdivision complete: /tmp/.../step_02_subdivide.npz
```

## Tips & Best Practices

### Resolution Strategy
1. Start with **64³** for fast iteration
2. Subdivide to **128³** for production
3. Subdivide to **256³** for maximum detail (requires good GPU)

### Hardening Strategy
1. **Light hardening** after initial reconstruction (variance_penalty=0.1)
2. **Subdivide** to increase resolution
3. **Aggressive hardening** at higher resolution (variance_penalty=0.2-0.3)

### Iteration Budget
- **Reconstruction**: 8000-12000 iterations (most important)
- **First hardening**: 500-1000 iterations
- **Subsequent hardening**: 1000-2000 iterations (higher res = more iters)

### Memory Management
Grid size memory usage (approximate):
- 64³: ~1 MB (fits on any GPU)
- 128³: ~8 MB (comfortable on most GPUs)
- 256³: ~64 MB (requires good GPU)
- 512³: ~500 MB (requires sharded mode on 8 GPUs)

## Troubleshooting

### Out of Memory
**Problem:** CUDA out of memory during reconstruction
**Solution:**
1. Reduce `grid_size`
2. Reduce `img_res`
3. Use smaller batch of frames (`frame_step: 2`)

### Slow Performance
**Problem:** Training is very slow
**Solution:**
1. Reduce `n_iters`
2. Reduce `img_res`
3. Reduce `n_samples`
4. Ensure GPU is being used (check logs)

### Poor Quality
**Problem:** Reconstruction looks blurry or incomplete
**Solution:**
1. Increase `n_iters` in reconstruction
2. Adjust regularization (`lambda_l1`, `lambda_tv_sigma`)
3. Use more frames (`frame_step: 1`)
4. Increase `img_res` for training

### Colors Turn White
**Problem:** Hardening makes colors washed out
**Solution:**
1. Reduce hardening iterations
2. Reduce `variance_penalty`
3. Check that reconstruction quality is good first

## Example Workflows

### Quick Test (5 minutes)
```json
{
  "steps": [
    {"type": "reconstruct", "params": {"grid_size": 32, "n_iters": 1000}},
    {"type": "harden", "params": {"n_iters": 100}}
  ]
}
```

### Standard Quality (30 minutes)
```json
{
  "steps": [
    {"type": "reconstruct", "params": {"grid_size": 64, "n_iters": 8000}},
    {"type": "subdivide", "params": {"subdivision": 2}},
    {"type": "harden", "params": {"n_iters": 1000}}
  ]
}
```

### High Quality (2+ hours)
```json
{
  "steps": [
    {"type": "reconstruct", "params": {"grid_size": 128, "n_iters": 12000, "img_res": [512, 512]}},
    {"type": "harden", "params": {"n_iters": 2000}}
  ]
}
```

## Common Workflows

### Split Training into Multiple Sessions
Run initial reconstruction:
```bash
python video_orbit_voxel_recon.py video.mp4 0 --n-iters 8000 --out-dir initial_recon
```

Later, continue with more iterations:
```bash
python video_orbit_voxel_recon.py video.mp4 0 \
    --checkpoint initial_recon/recon_volume.npz \
    --n-iters 4000 \
    --out-dir continued_recon
```

Or use pipeline:
```json
{
  "input": {
    "type": "video_with_checkpoint",
    "video_path": "video.mp4",
    "checkpoint_path": "initial_recon/recon_volume.npz"
  },
  "output": "final.npz",
  "steps": [
    {"type": "reconstruct", "params": {"n_iters": 4000}},
    {"type": "subdivide", "params": {"subdivision": 2}},
    {"type": "harden", "params": {"n_iters": 1000}}
  ]
}
```

## See Also

- `PARAMETER_GUIDE.md` - Detailed parameter documentation
- `example_pipeline.json` - Basic example
- `example_iterative_hardening.json` - Multi-step hardening
- `example_high_res_pipeline.json` - High quality settings
- `example_resume_from_checkpoint.json` - Resume training
- `example_checkpoint_then_process.json` - Resume + refine + harden

