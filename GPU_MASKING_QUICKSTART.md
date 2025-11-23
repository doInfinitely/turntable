# GPU Masking Tool Quick Start

Run the masking tool on Lambda's GPU, control from your Mac.

## One-Time Setup

### On Mac
```bash
./setup_x11_mac.sh
# Follow the prompts
```

### On Lambda
```bash
./setup_x11_lambda.sh
```

## Every Session

### 1. Start X11 on Mac
```bash
open -a XQuartz
xhost + localhost
```

### 2. Connect to Lambda
```bash
ssh -XC lambda-gpu
# Or: ssh -XC ubuntu@YOUR_LAMBDA_IP
```

### 3. Test X11
```bash
xclock
# Should see a clock on your Mac
# Ctrl+C to close
```

### 4. Run GPU Masking Tool
```bash
cd ~/turntable
python voxel_masking_tool_gpu.py recon_volume.npz
```

The window appears on your Mac, rendering happens on Lambda's GPU!

## Performance

- **GPU**: NVIDIA A100
- **FPS**: 30-60 (vs 5-10 on CPU)
- **Render time**: ~10-20ms/frame (vs ~100-200ms on CPU)
- **Network overhead**: ~20-50ms per frame

## Troubleshooting

**No display**:
```bash
echo $DISPLAY  # Should show "localhost:10.0" or similar
```

**Still no display**: Reconnect with `-XC` flag
```bash
exit
ssh -XC lambda-gpu
```

**Slow**: Reduce resolution
```python
# In voxel_masking_tool_gpu.py, line ~288
img_res = (512, 512)  # Instead of (800, 800)
```

## Cost

- **Lambda A100**: ~$1.10/hour
- **Typical session**: 15-30 minutes
- **Cost**: ~$0.28-$0.55 per session

**10-60x faster** than local CPU!

## Full Documentation

- `X11_FORWARDING_GUIDE.md` - Complete setup guide
- `MASKING_TOOL_GUIDE.md` - Usage instructions
- `LAMBDA_GPU_SETUP.md` - Initial Lambda setup


