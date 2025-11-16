# GPU-Accelerated Voxel Reconstruction on Lambda

This guide shows how to run the **voxel reconstruction training** on Lambda GPU for massive speedup.

## 🚀 Speed Comparison

### CPU (Local MacBook)
- 21 frames: ~8-10 minutes
- 82 frames: ~30-40 minutes

### GPU (Lambda A10)
- 21 frames: ~1-2 minutes ⚡
- 82 frames: ~4-6 minutes ⚡

**~10x faster on GPU!**

## Quick Setup

### 1. Launch Lambda Instance

```bash
# Create instance with PyTorch template
# Recommended: A10 (24GB) - $0.60/hr
```

### 2. Upload Files

```bash
# From your local machine
scp video_orbit_voxel_recon.py ubuntu@YOUR_LAMBDA_IP:~/
scp your_video.mp4 ubuntu@YOUR_LAMBDA_IP:~/
```

### 3. Connect and Run

```bash
# Connect
ssh ubuntu@YOUR_LAMBDA_IP

# Verify GPU
nvidia-smi

# Run reconstruction (all frames, neighbor growth)
python3 video_orbit_voxel_recon.py your_video.mp4 0 --neighbor-growth

# Should see:
# ============================================================
# Using device: cuda
# GPU: NVIDIA A10
# GPU Memory: 24.0 GB
# CUDA version: 11.8
# ============================================================
```

### 4. Download Results

```bash
# From local machine
scp ubuntu@YOUR_LAMBDA_IP:~/video_voxel_out/recon_volume.npz ./
```

## GPU Utilization Tips

### Monitor GPU Usage

```bash
# Watch GPU in real-time
watch -n 1 nvidia-smi

# Or during training
nvidia-smi dmon -s u
```

### Increase Batch Size / Resolution

Since you have more memory on GPU, you can increase quality:

```python
# In video_orbit_voxel_recon.py, edit main() call:
train_from_video(
    ...
    grid_size=64,           # More voxels! (was 32)
    img_res=(128, 128),     # Higher res! (was 64x64)
    n_samples=128,          # More ray samples! (was 64)
    ...
)
```

**Warning**: 64³ grid uses ~16x more memory than 32³!

### Use Mixed Precision (Optional)

For even faster training:

```python
# At the top of train_from_video()
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    sigma_rec, rgb_rec = recon_vol()
    pred_images = render_volume(...)
    ...
    loss = loss_mse

scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()
```

## Recommended Lambda GPU Options

### For Development ($0.60-1.00/hr)
- **A10** (24GB) - Best value, handles 32³ grids easily
- **RTX 6000 Ada** (48GB) - Can do 64³ grids

### For Large Reconstructions ($1.50-3.00/hr)
- **A100** (40GB/80GB) - Fastest training
- **H100** (80GB) - Overkill but blazing fast

## Example Session

```bash
# 1. Connect
ssh ubuntu@YOUR_LAMBDA_IP

# 2. Upload video (if not done)
# (from another terminal on local machine)
scp video.mp4 ubuntu@YOUR_LAMBDA_IP:~/

# 3. Run reconstruction with all frames
time python3 video_orbit_voxel_recon.py video.mp4 0 --neighbor-growth

# Output:
# ============================================================
# Using device: cuda
# GPU: NVIDIA A10
# GPU Memory: 24.0 GB
# CUDA version: 11.8
# ============================================================
# [AUTO] Estimated orbit period: 82 frames, fps=24.00
# ...
# [0/2000] loss=1.876965e-01 ...
# [100/2000] loss=5.433442e-02 ...
# ...
# Done. Check: video_voxel_out
#
# real    4m32s  ⚡ (vs 35m on CPU)

# 4. Download results
# (from local machine)
scp ubuntu@YOUR_LAMBDA_IP:~/video_voxel_out/recon_volume.npz ./

# 5. View locally
python voxel_volume_viewer.py recon_volume.npz
```

## Training Multiple Videos

```bash
# Create a batch script
cat > run_batch.sh << 'EOF'
#!/bin/bash
for video in video1.mp4 video2.mp4 video3.mp4; do
    echo "Processing $video..."
    python3 video_orbit_voxel_recon.py "$video" 0 --neighbor-growth
    mv video_voxel_out "output_${video%.*}"
done
EOF

chmod +x run_batch.sh
./run_batch.sh
```

## Troubleshooting

### "Numpy is not available" or NumPy 2.x compatibility error

```bash
# Fix: Downgrade NumPy to 1.x
pip install "numpy<2" --force-reinstall

# Verify
python3 -c "import torch; import numpy; print('✓ Fixed!')"
```

This happens because PyTorch was compiled with NumPy 1.x and doesn't work with NumPy 2.x yet.

### "CUDA out of memory"

```python
# Reduce grid size
grid_size=16  # or 24

# OR reduce image resolution
img_res=(32, 32)

# OR reduce batch of frames loaded at once
# (edit load_frames_as_tensors to load fewer frames)
```

### "No GPU detected"

```bash
# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Still getting hollow shell?

With all 82 frames, if you still get a hollow funnel, try:

1. **Tighten neighbor threshold**:
   ```python
   # In video_orbit_voxel_recon.py, around line 830:
   target_alpha = 0.5  # Increase from 0.3 to 0.5 or 0.6
   ```

2. **Stronger Gaussian seed**:
   ```python
   # Around line 813:
   peak_sigma = 300.0  # Increase from 200.0
   gaussian_std = 3.0  # Increase from 2.0
   ```

3. **Train longer**:
   ```python
   n_iters=5000  # Increase from 2000
   ```

## Cost Optimization

```bash
# Use spot instances (50-70% cheaper, can be interrupted)
# Or set a reminder:
echo "sudo shutdown -h now" | at now + 30 minutes

# Download results before instance stops!
```

## Performance Tips

1. **Use tmux** so training continues if disconnected:
   ```bash
   tmux new -s training
   python3 video_orbit_voxel_recon.py video.mp4 0 --neighbor-growth
   # Ctrl+B, then D to detach
   # Reconnect: tmux attach -t training
   ```

2. **Log output**:
   ```bash
   python3 video_orbit_voxel_recon.py video.mp4 0 --neighbor-growth 2>&1 | tee training.log
   ```

3. **Compress before download**:
   ```bash
   tar czf results.tar.gz video_voxel_out/
   scp ubuntu@YOUR_LAMBDA_IP:~/results.tar.gz ./
   ```

## Summary

The reconstruction code **already runs on GPU automatically** - just run it on a Lambda instance with CUDA available. You'll see ~10x speedup compared to CPU!

Key points:
- ✅ Automatic GPU detection (no code changes needed)
- ✅ Much faster training (minutes instead of tens of minutes)
- ✅ Can use higher resolution and more voxels
- ✅ All frames (82 instead of 21) for better constraints

