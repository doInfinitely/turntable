# Multi-GPU Training Guide

⚠️ **UPDATE**: Multi-GPU rendering is currently **DISABLED** for this workload.

## Why Multi-GPU Doesn't Help (Currently)

### The Problem: Copy Overhead

**Measured performance on 8× A100**:
- Single GPU: ~0.15s/iteration
- 8 GPUs: ~0.18s/iteration (actually **slower**!)

**Root cause:**
- 128³ volume = ~100MB (sigma + RGB tensors)
- **Every iteration**: Must copy volume to all 8 GPUs = **800MB of PCIe transfers**
- With only 82 frames / 8 GPUs = ~10 frames/GPU
- **Copy time >> parallelism benefit**

### When Multi-GPU Would Help

Multi-GPU rendering would be beneficial for:

1. **Many more frames** (500-1000+):
   - More work per GPU amortizes copy cost
   - Example: 800 frames / 8 GPUs = 100 frames/GPU (worthwhile)

2. **Multiple videos in parallel** (data parallelism):
   - Each GPU trains on a different video
   - No inter-GPU communication needed
   - True 8x speedup

3. **Cached volume approach** (complex):
   - Keep volume replica on each GPU permanently
   - Only sync weight updates (much smaller)
   - Requires careful gradient synchronization

### Current Configuration (128³ Resolution)

- **Voxel Grid**: 128×128×128 = 2,097,152 voxels
- **Training Steps**: 8,000 iterations (L1 regularization needs time to eliminate clouds)
- **Regularization**: Aggressive L1 sparsity (3e-2) + TV smoothness

## Running on Lambda (Single GPU Recommended)

### 1. Launch a Single-GPU Instance

For this workload, **a single A100 is optimal**:

```bash
# From your local machine
lambda instances create \
  --instance-type gpu_1x_a100 \
  --region us-west-1 \
  --name turntable-gpu
```

Or if you already have an 8-GPU instance, it will just use 1 GPU (no waste, others idle).

### 2. Connect and Setup

```bash
# SSH into the instance
ssh ubuntu@<instance-ip>

# Clone/setup your repo
cd ~
git clone <your-repo> turntable  # or rsync your code
cd turntable

# Install dependencies
pip install torch torchvision opencv-python numpy
```

### 3. Verify GPU Detection

```bash
# Check that all 8 GPUs are visible
nvidia-smi

# Quick PyTorch test
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

Expected output: `GPUs available: 8`

### 4. Run Training

```bash
python video_orbit_voxel_recon.py <video_path> 0
```

You should see output like:
```
============================================================
Primary device: cuda:0
Available GPUs: 8
  GPU 0: NVIDIA A100-SXM4-80GB
    Memory: 80.0 GB
  GPU 1: NVIDIA A100-SXM4-80GB
    Memory: 80.0 GB
  ...
CUDA version: 11.8
MULTI-GPU MODE: Will parallelize rendering across 8 GPUs
  Main GPU (cuda:0) will hold the model and orchestrate training
  Worker GPUs will handle subsets of frame rendering
============================================================
```

### 5. Monitor GPU Utilization

```bash
# In a separate terminal, watch GPU usage
watch -n 0.5 nvidia-smi
```

You should see all 8 GPUs active during training.

## Performance Expectations (Single GPU)

### A100 40GB/80GB (Measured)
- **Per iteration**: ~0.15-0.20 seconds (128³, 82 frames)
- **Total training (8000 iters)**: **~20-30 minutes**
- **Breakdown**: 
  - Rendering: ~0.15s (dominant)
  - Backward pass: ~0.03s (fast!)
  - Total: ~0.18s/iter × 8000 = 24 minutes

### Cost
- **1× A100 on Lambda**: ~$1.50-2.00/hour
- **128³ training (8000 iters)**: **~$0.50-1.00 per video** (20-30 min)
- **Much cheaper than 8-GPU instance** ($12-15/hour)

## Memory Requirements

Each GPU needs to hold:
- **Voxel volume**: ~25 MB (128³ × 4 bytes for sigma + 75 MB for RGB)
- **Intermediate tensors**: ~100-500 MB per frame batch
- **Total per GPU**: ~1-2 GB

An **A100 with 80GB** can easily handle this. Even smaller GPUs like **A10 (24GB)** would work fine.

## Scaling to Higher Resolutions

### 256³ Resolution (16M voxels)
- **Memory**: ~8-10 GB per GPU
- **Recommended**: 8× A100 80GB
- **Training time**: ~2-3 hours with 8 GPUs

### 512³ Resolution (134M voxels)
- **Memory**: ~60-80 GB per GPU (tight!)
- **Recommended**: 8× A100 80GB or H100
- **Training time**: ~10-15 hours with 8 GPUs

## Cost Optimization

### Recommendation: Use Single A100
**Best value**: 1× A100 40GB or 80GB

### Lambda GPU Pricing (Current)
- **1× A100 (40GB)**: ~$1.50/hour
- **1× A100 (80GB)**: ~$2.00/hour
- **8× A100**: ~$12-15/hour (overkill, 7 GPUs sit idle)

### Cost per Video (128³ Resolution)
- **Training time**: 20-30 minutes
- **Cost**: **$0.50-1.00 per video** on single A100
- **vs 8× A100**: Same speed, $4-8 per video (8x more expensive!)

### Tips
1. **Use spot instances** if available (50-70% discount → ~$0.25/video)
2. **Terminate immediately** after training
3. **Download results** before terminating:
   ```bash
   scp ubuntu@<ip>:~/turntable/video_voxel_out/recon_volume.npz .
   scp ubuntu@<ip>:~/turntable/video_voxel_out/recon_voxels.ply .
   ```
4. **Batch multiple videos**: If you have 8 videos, use the batch script or manually launch 8 training runs:
   ```bash
   # Option 1: Use batch script (automatic)
   ./batch_train_8gpu.sh video1.mp4 video2.mp4 video3.mp4 video4.mp4 \
                          video5.mp4 video6.mp4 video7.mp4 video8.mp4
   
   # Option 2: Manual (more control)
   # In 8 separate terminals:
   CUDA_VISIBLE_DEVICES=0 python video_orbit_voxel_recon.py video1.mp4 0 &
   CUDA_VISIBLE_DEVICES=1 python video_orbit_voxel_recon.py video2.mp4 0 &
   CUDA_VISIBLE_DEVICES=2 python video_orbit_voxel_recon.py video3.mp4 0 &
   # ... etc for GPUs 3-7
   ```
   This gives **TRUE 8x throughput** (8 videos in ~25 minutes instead of 200 minutes)

## Troubleshooting

### "CUDA out of memory"
- Reduce `grid_size` (e.g., 128 → 96)
- Reduce `n_samples` (e.g., 64 → 48)
- Reduce `img_res` (e.g., (64,64) → (48,48))

### "Only using 1 GPU"
- Check `torch.cuda.device_count()` returns 8
- Ensure CUDA is properly installed: `nvcc --version`
- Verify NCCL is available (for multi-GPU communication)

### "Training is slow with 8 GPUs"
- Check `nvidia-smi` - all GPUs should show activity
- Ensure you're not on a shared node (other users' processes)
- Verify network isn't the bottleneck (shouldn't be for ~80 frames)

## Code Overview

### Key Changes for Multi-GPU

1. **`render_volume_multigpu()`**: New function that splits rendering across GPUs
   ```python
   def render_volume_multigpu(sigma, rgb, K, poses, n_gpus=1):
       # Split poses across GPUs
       # Each GPU renders its subset in parallel
       # Gather results back to cuda:0
   ```

2. **Automatic detection**: Training loop checks `n_gpus` and uses appropriate function
   ```python
   if n_gpus > 1:
       pred_images = render_volume_multigpu(...)
   else:
       pred_images = render_volume(...)
   ```

3. **Thread-based parallelism**: Uses `ThreadPoolExecutor` for concurrent GPU operations
   - Python threads work well for GPU operations (releases GIL during CUDA calls)
   - Each thread manages one GPU

## Future Improvements

- [ ] Distributed training (DDP) for even better scaling
- [ ] Gradient accumulation for larger batch sizes
- [ ] Mixed precision (FP16) for 2x memory savings
- [ ] Model parallelism for grids > 512³

