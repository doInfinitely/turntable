# X11 Forwarding Guide: Run GPU Masking Tool on Lambda, Display on Mac

## Overview

Run the GPU-accelerated voxel masking tool on your Lambda GPU instance while displaying the window on your Mac. This gives you:

- **GPU acceleration** (10-60x faster rendering)
- **Local control** (use your Mac's mouse/keyboard)
- **Remote computation** (leverage Lambda's A100 GPUs)

## Prerequisites

### On Your Mac

1. **Install XQuartz** (X11 server for macOS)
   ```bash
   brew install --cask xquartz
   ```

2. **Start XQuartz**
   - Open XQuartz from Applications
   - In XQuartz preferences (Cmd+,):
     - Go to "Security" tab
     - Check "Allow connections from network clients"
   - Restart XQuartz

3. **Allow X11 forwarding in SSH**
   ```bash
   # Add to ~/.ssh/config
   cat >> ~/.ssh/config << EOF
   
   Host lambda-gpu
       HostName YOUR_LAMBDA_IP
       User ubuntu
       ForwardX11 yes
       ForwardX11Trusted yes
       Compression yes
   EOF
   ```

### On Lambda

1. **Install X11 libraries**
   ```bash
   sudo apt-get update
   sudo apt-get install -y xauth x11-apps
   ```

2. **Enable X11 forwarding in SSHD** (usually enabled by default)
   ```bash
   # Check if enabled
   grep "X11Forwarding" /etc/ssh/sshd_config
   # Should show: X11Forwarding yes
   ```

## Setup

### 1. On Your Mac: Set DISPLAY

```bash
# Allow X11 connections from localhost
xhost + localhost
```

### 2. Upload Files to Lambda

```bash
# From your Mac
scp voxel_masking_tool_gpu.py ubuntu@YOUR_LAMBDA_IP:~/
scp video_voxel_out/recon_volume.npz ubuntu@YOUR_LAMBDA_IP:~/
```

### 3. Connect with X11 Forwarding

```bash
# Use -X flag for X11 forwarding, -C for compression
ssh -XC ubuntu@YOUR_LAMBDA_IP

# Or if you set up ~/.ssh/config:
ssh -C lambda-gpu
```

### 4. Test X11 Forwarding

```bash
# On Lambda, try opening a simple X11 app
xclock

# You should see a clock window appear on your Mac
# Press Ctrl+C to close it
```

## Running the GPU Masking Tool

### Basic Usage

```bash
# On Lambda (connected via SSH with X11 forwarding)
cd ~/turntable
python voxel_masking_tool_gpu.py recon_volume.npz
```

The pygame window will appear on your Mac, but rendering happens on Lambda's GPU!

### Expected Performance

- **FPS**: 30-60 FPS (depending on network latency)
- **Rendering**: ~10-50ms per frame on A100 GPU
- **Network overhead**: ~20-50ms per frame over X11

## Troubleshooting

### "cannot open display"

**Problem**: X11 forwarding not working

**Solutions**:
```bash
# On Lambda, check DISPLAY is set
echo $DISPLAY
# Should show something like "localhost:10.0"

# If empty, X11 forwarding failed. Reconnect with:
ssh -XC ubuntu@YOUR_LAMBDA_IP

# On Mac, make sure XQuartz is running
ps aux | grep XQuartz
```

### "Error: pygame.error: No available video device"

**Problem**: X11 display not available to pygame

**Solution**:
```bash
# On Lambda, explicitly set display
export DISPLAY=:0

# Or reconnect SSH with X11 forwarding
```

### Slow/Laggy Display

**Problem**: Network latency

**Solutions**:
1. Use compression (already in command: `-C`)
2. Reduce resolution in the script:
   ```python
   img_res = (512, 512)  # Instead of (800, 800)
   ```
3. Use lower quality X11:
   ```bash
   # Add to ~/.ssh/config
   ForwardX11Trusted yes
   Compression yes
   CompressionLevel 9
   ```

### "X11 connection rejected because of wrong authentication"

**Solution**:
```bash
# On Lambda
rm ~/.Xauthority
# Reconnect SSH
```

### Black Screen or Garbled Display

**Problem**: X11 visual mismatch

**Solution**:
```bash
# On Mac before connecting
export XAUTHORITY=~/.Xauthority
xauth generate $DISPLAY . trusted

# Then connect
ssh -XC ubuntu@YOUR_LAMBDA_IP
```

## Advanced: Faster Alternative with VNC

If X11 forwarding is too slow, use VNC for better performance:

### Setup VNC on Lambda

```bash
# On Lambda
sudo apt-get install -y x11vnc xvfb

# Start a virtual display
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# Start VNC server
x11vnc -display :99 -forever -nopw -bg
```

### Connect from Mac

```bash
# Create SSH tunnel for VNC
ssh -L 5900:localhost:5900 ubuntu@YOUR_LAMBDA_IP

# On Mac, open VNC viewer and connect to:
# localhost:5900
```

Then run the masking tool and it will appear in the VNC viewer.

## Performance Comparison

| Method | FPS | Latency | Setup |
|--------|-----|---------|-------|
| Local (CPU) | 5-10 | 0ms | Easy |
| X11 Forward | 20-40 | 20-50ms | Medium |
| VNC | 30-60 | 10-30ms | Complex |
| **GPU Local** | 60+ | 0ms | None needed |

## Tips

### Optimize X11 Forwarding

```bash
# Add to ~/.ssh/config for best X11 performance
Host lambda-gpu
    HostName YOUR_LAMBDA_IP
    User ubuntu
    ForwardX11 yes
    ForwardX11Trusted yes
    Compression yes
    CompressionLevel 9
    TCPKeepAlive yes
    ServerAliveInterval 60
```

### Monitor Performance

```bash
# On Lambda, watch GPU usage while masking
watch -n 1 nvidia-smi

# Should show:
# - GPU utilization: 40-80%
# - Memory usage: ~1-4GB depending on volume size
```

### Batch Masking

For multiple volumes:
```bash
# On Lambda
for vol in recon_volume*.npz; do
    echo "Masking $vol..."
    python voxel_masking_tool_gpu.py "$vol"
    # Mask interactively, save (Ctrl+S), close (ESC)
done
```

## Quick Reference

```bash
# Mac: Start XQuartz and allow connections
open -a XQuartz
xhost + localhost

# Mac: Connect to Lambda with X11
ssh -XC ubuntu@YOUR_LAMBDA_IP

# Lambda: Test X11 (should see clock on Mac)
xclock

# Lambda: Run GPU masking tool
cd ~/turntable
python voxel_masking_tool_gpu.py recon_volume.npz

# Lambda: Check GPU usage
nvidia-smi
```

## Keyboard Shortcuts (Same as CPU version)

| Key | Action |
|-----|--------|
| ← → | Orbit left/right |
| ↑ ↓ | Orbit up/down |
| + - | Zoom in/out |
| Left Click | Mask (remove) voxels |
| Right Click | Unmask (restore) voxels |
| U | Undo |
| Ctrl+S | Save |
| ESC | Quit |

## Cost Considerations

**Lambda GPU time**: ~$1.10/hour (A100)
**Typical masking session**: 10-30 minutes
**Cost per session**: ~$0.18-$0.55

Much faster than local CPU, worth it for large volumes!

## See Also

- `voxel_masking_tool.py` - CPU version (slower, local)
- `voxel_masking_tool_gpu.py` - GPU version (this guide)
- `LAMBDA_GPU_SETUP.md` - Initial Lambda setup
- `MASKING_TOOL_GUIDE.md` - Detailed usage guide


