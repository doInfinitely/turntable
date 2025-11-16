# Lambda GPU - Quick Start Guide

## One-Command Setup

```bash
# 1. Connect to Lambda instance (replace with your IP)
ssh -X -C ubuntu@YOUR_INSTANCE_IP

# 2. Run setup script
curl -O https://raw.githubusercontent.com/YOUR_REPO/main/setup_lambda.sh
bash setup_lambda.sh

# Or if you uploaded the script:
bash setup_lambda.sh
```

## Upload Files (from your local machine)

```bash
# Upload setup script
scp setup_lambda.sh ubuntu@YOUR_INSTANCE_IP:~/

# Upload GPU viewer
scp voxel_volume_viewer_gpu.py ubuntu@YOUR_INSTANCE_IP:~/

# Upload reconstruction
scp video_voxel_out/recon_volume.npz ubuntu@YOUR_INSTANCE_IP:~/
```

## Run the Viewer

```bash
# On Lambda instance
python3 voxel_volume_viewer_gpu.py recon_volume.npz
```

## macOS X11 Setup (First Time Only)

```bash
# Install XQuartz
brew install --cask xquartz

# Start XQuartz
open -a XQuartz

# Enable "Allow connections from network clients"
# XQuartz > Preferences > Security tab

# Restart XQuartz
```

## Quick Commands

```bash
# Check GPU
nvidia-smi

# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Test X11
xclock  # Should show a clock on your screen

# Run reconstruction on GPU
python3 video_orbit_voxel_recon.py video.mp4 0 --neighbor-growth

# Download results
scp ubuntu@YOUR_INSTANCE_IP:~/video_voxel_out/recon_volume.npz ./
```

## Recommended Lambda GPU Types

- **A10** (24GB) - Best price/performance - ~$0.60/hr
- **RTX 6000 Ada** (48GB) - More VRAM for larger grids - ~$1.00/hr
- **A100** (40GB/80GB) - Fastest training - ~$1.50/hr

## Controls

- **V** = Volumetric mode (NeRF)
- **C** = Cube mode (Minecraft)
- **Arrows** = Rotate
- **+/-** = Zoom
- **ESC** = Quit

## Troubleshooting

```bash
# Can't see window?
echo $DISPLAY  # Should show localhost:10.0 or similar
ssh -X -C ubuntu@YOUR_IP  # Reconnect

# Slow/laggy?
# Use -C for compression (already in command above)
# Or switch to VNC (see full guide)

# Out of memory?
# Increase thresh_factor in viewer code
# Or use smaller render resolution
```

## Cost Savings

- **Stop instance** when done (loses data!)
- **Download results first**: `scp ubuntu@IP:~/recon_volume.npz ./`
- **Persistent storage** costs extra but saves data
- Lambda charges by the hour (rounded up)

## Full Documentation

See `LAMBDA_GPU_SETUP.md` for detailed instructions.

