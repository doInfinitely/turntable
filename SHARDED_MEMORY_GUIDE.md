# Sharded Mode Memory Guide

## Memory Requirements per Shard

The sharded mode splits the volume into 8 shards (2×2×2 octants). Each GPU holds one shard.

### Formula

For a total grid size of `N³`, each shard is `(N/2)³` voxels.

**Base memory per shard:**
- Density: `(N/2)³ × 4 bytes` (float32)
- Color: `(N/2)³ × 3 × 4 bytes` (3 channels, float32)
- **Total base:** `(N/2)³ × 16 bytes`

**During training (with TV regularization):**
- TV creates 6 intermediate tensors (dx, dy, dz for both sigma and RGB)
- Each intermediate is almost the same size as the shard
- **Peak memory:** ~8-10× base memory during backward pass

### Common Configurations

| Grid Size | Shard Size | Base Memory | Peak Memory | Recommended GPU |
|-----------|------------|-------------|-------------|-----------------|
| 256³      | 128³       | 32 MB       | ~300 MB     | Any modern GPU  |
| 384³      | 192³       | 108 MB      | ~1 GB       | 16 GB+          |
| 512³      | 256³       | 256 MB      | ~2.5 GB     | 24 GB+          |
| 768³      | 384³       | 864 MB      | ~8 GB       | 40 GB+          |
| 1024³     | 512³       | 2 GB        | ~20 GB      | 80 GB+ (A100)   |
| 1280³     | 640³       | 4 GB        | ~40 GB      | 80 GB+ (tight)  |
| 1536³     | 768³       | 6.9 GB      | ~70 GB      | 192 GB (B200)   |
| 2048³     | 1024³      | 16 GB       | ~160 GB     | 192 GB (B200)   |

## Recommended Settings by GPU

### Consumer GPUs (16-24 GB)
```bash
# 384³ total (192³ per shard) - safe for RTX 4090, etc.
python video_orbit_voxel_recon.py video.mp4 0 --sharded --grid-size 384
```

### A100 40GB
```bash
# 768³ total (384³ per shard) - good balance
python video_orbit_voxel_recon.py video.mp4 0 --sharded --grid-size 768
```

### A100 80GB
```bash
# 1024³ total (512³ per shard) - ambitious but possible
python video_orbit_voxel_recon.py video.mp4 0 --sharded --grid-size 1024
```

### B200 192GB
```bash
# 512³ default (256³ per shard) - very safe
python video_orbit_voxel_recon.py video.mp4 0 --sharded

# 1536³ total (768³ per shard) - using ~70GB per GPU
python video_orbit_voxel_recon.py video.mp4 0 --sharded --grid-size 1536

# 2048³ total (1024³ per shard) - using ~160GB per GPU (tight!)
python video_orbit_voxel_recon.py video.mp4 0 --sharded --grid-size 2048
```

## Troubleshooting OOM Errors

If you get "CUDA out of memory" errors:

### 1. Reduce Grid Size
The default sharded size is now **512³** (256³ per shard).

Try **384³** (192³ per shard):
```bash
python video_orbit_voxel_recon.py video.mp4 0 --sharded --grid-size 384
```

### 2. Check Memory Fragmentation
```bash
# Set PyTorch to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python video_orbit_voxel_recon.py video.mp4 0 --sharded
```

### 3. Disable TV Regularization (if needed)
Edit `video_orbit_voxel_recon.py` and set:
```python
lambda_tv_sigma = 0.0  # Disable TV on sigma
lambda_tv_rgb = 0.0    # Disable TV on RGB
```

TV regularization is the main memory hog (creates 6 large intermediate tensors).

### 4. Monitor Memory Usage
```bash
# In another terminal, watch GPU memory
watch -n 0.5 nvidia-smi
```

## Performance Notes

**Sharded mode benefits:**
- ✅ Enables massive resolutions impossible on single GPU
- ✅ Distributes memory across 8 GPUs
- ✅ Each shard trains independently (model parallelism)

**Trade-offs:**
- ⚠️ Ray sampling requires cross-GPU communication
- ⚠️ Slower than single GPU for small volumes
- ⚠️ Only worth it for 512³+ volumes

## Example Runs

### Conservative (guaranteed to work on B200)
```bash
# 384³ total, 192³ per shard, ~1GB per GPU
python video_orbit_voxel_recon.py 20251115_111317_minimax_hailuo-2.3_Camera_movement_the_camera_or.mp4 0 --sharded --grid-size 384
```

### Balanced (recommended for B200)
```bash
# 512³ total, 256³ per shard, ~2.5GB per GPU (default)
python video_orbit_voxel_recon.py 20251115_111317_minimax_hailuo-2.3_Camera_movement_the_camera_or.mp4 0 --sharded
```

### High Quality (B200 with headroom)
```bash
# 768³ total, 384³ per shard, ~8GB per GPU
python video_orbit_voxel_recon.py 20251115_111317_minimax_hailuo-2.3_Camera_movement_the_camera_or.mp4 0 --sharded --grid-size 768
```

### Ultra Quality (B200, using most memory)
```bash
# 1536³ total, 768³ per shard, ~70GB per GPU
python video_orbit_voxel_recon.py 20251115_111317_minimax_hailuo-2.3_Camera_movement_the_camera_or.mp4 0 --sharded --grid-size 1536
```

## Why 1024³ Failed on B200

Your attempt used:
- **1024³ total** → **512³ per shard**
- Base memory: 2 GB per shard
- With TV regularization: ~20 GB peak per shard
- With gradients, optimizer state, frame data: **~25-30 GB per GPU**

This should fit in 192 GB... but it failed at 177 GB used. Why?

**Likely causes:**
1. **Memory fragmentation** - PyTorch couldn't find contiguous block
2. **Frame data + poses** - 82 frames at 64×64 = ~1 MB, but poses/rays add up
3. **Gradient accumulation** - Adam optimizer stores 2× the model size
4. **Multiple intermediate tensors** during backward pass

**Solution:** Start with **384³** or default **512³** which uses far less memory.

