# Multi-GPU Lesson Learned

## What We Tried

Implemented multi-GPU parallelization to speed up voxel reconstruction by splitting frame rendering across 8 GPUs.

## What We Learned

**Multi-GPU rendering was SLOWER than single GPU** ❌

### Measured Performance
- **Single A100**: ~0.15s/iteration
- **8× A100**: ~0.18s/iteration (20% slower!)

### Root Cause: Volume Copy Overhead

```
Every iteration:
  1. Copy 128³ volume (~100MB) from GPU 0 → GPU 1-7
  2. Each GPU renders ~10 frames (82 frames / 8 GPUs)
  3. Copy results back to GPU 0
  
Total transfers: ~800MB/iteration
Parallelism benefit: Minimal (only 10 frames/GPU)

Result: Copy time >> compute time saved
```

## Why This Architecture Failed

1. **Too few frames per GPU** (82 / 8 = 10 frames)
   - Not enough work to amortize the 100MB volume copy cost
   - Would need 500-1000+ frames for multi-GPU to help

2. **Backward pass already fast** (0.03s vs 0.15s rendering)
   - Even if rendering was free, we'd only save ~80% of time
   - Would get at best ~5x speedup, not 8x

3. **PCIe bandwidth bottleneck**
   - 100MB × 8 GPUs = 800MB of data movement per iteration
   - NVLink would help but still significant overhead

## When Multi-GPU Would Work

### ✅ Scenario 1: Many Frames (500-1000+)
```
800 frames / 8 GPUs = 100 frames per GPU
→ Copy 100MB once, render 100× → worthwhile
```

### ✅ Scenario 2: Data Parallelism (Multiple Videos)
```
8 different videos, each GPU trains independently
→ No inter-GPU communication
→ True 8x throughput
```

### ✅ Scenario 3: Cached Volume (Complex Implementation)
```
Keep volume replica on each GPU permanently
Only sync gradients (~10KB) not entire volume (~100MB)
→ Requires careful DDP/gradient synchronization
```

## The Fix: Single GPU is Optimal

**For this workload (82 frames, 128³ volume):**
- Use **1× A100**
- **20-30 minutes** per video
- **$0.50-1.00 per video** on Lambda
- Clean, simple, no wasted GPUs

**For 8 videos in parallel:**
- Use batch script or `CUDA_VISIBLE_DEVICES`
- Each GPU runs independent training
- **TRUE 8x throughput**: 8 videos in 25 minutes

## Key Takeaway

> **More GPUs ≠ Faster**
> 
> Multi-GPU parallelism only helps when:
> 1. Work per GPU >> data transfer cost
> 2. Or: No inter-GPU data transfer needed (true data parallelism)

For small-batch ML workloads like this, **data parallelism** (multiple independent jobs) beats **model parallelism** (splitting one job across GPUs).

## Code Changes

1. **Disabled multi-GPU rendering** in training loop
2. **Added timing diagnostics** to identify bottlenecks
3. **Created batch script** for true multi-video parallelism
4. **Updated documentation** to recommend single GPU

The multi-GPU code is still there (in `render_volume_multigpu`) for future use with:
- Very long videos (500+ frames)
- Or if someone implements the cached-volume approach

