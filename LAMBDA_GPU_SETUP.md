# Lambda GPU Setup Guide for Voxel Viewer

This guide will help you set up and run the GPU-accelerated voxel viewer on Lambda GPU instances.

## Step 1: Create a Lambda GPU Instance

1. **Sign up / Log in to Lambda Labs**
   - Go to https://lambdalabs.com/
   - Create an account or log in

2. **Launch an Instance**
   - Click "GPU Instances" in the dashboard
   - Choose a GPU instance (recommended: **A10** or **RTX 6000 Ada** for best price/performance)
   - Select region closest to you
   - Choose **PyTorch** as the instance type (comes with PyTorch pre-installed)
   - Add your SSH key
   - Launch the instance

3. **Note the IP address**
   - Once launched, note the instance IP address (e.g., `104.171.200.62`)

## Step 2: Connect with X11 Forwarding

X11 forwarding allows you to display the Pygame window on your local machine.

### On macOS:

```bash
# Install XQuartz (X11 server for macOS)
brew install --cask xquartz

# Start XQuartz
open -a XQuartz

# In XQuartz preferences, enable "Allow connections from network clients"
# (XQuartz > Preferences > Security tab)

# Connect to Lambda instance with X11 forwarding
ssh -X -C ubuntu@YOUR_INSTANCE_IP
```

### On Linux:

```bash
# X11 is usually pre-installed
# Connect with X11 forwarding
ssh -X -C ubuntu@YOUR_INSTANCE_IP
```

### On Windows:

```bash
# Install VcXsrv or Xming for Windows X server
# Download from: https://sourceforge.net/projects/vcxsrv/

# Start VcXsrv with default settings

# Connect using MobaXterm (has built-in X11) or:
ssh -X ubuntu@YOUR_INSTANCE_IP
```

## Step 3: Install Dependencies on Lambda Instance

```bash
# Update package lists
sudo apt-get update

# Install system dependencies for Pygame
sudo apt-get install -y python3-pygame libsdl2-dev libsdl2-image-dev \
    libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev \
    libportmidi-dev python3-dev

# Install PyTorch3D (for GPU-accelerated rendering)
# This may take a few minutes
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install other Python dependencies
pip install pygame opencv-python matplotlib numpy

# Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import pytorch3d; print('PyTorch3D installed successfully')"
```

## Step 4: Upload Your Code and Data

### Option A: Using scp (from your local machine)

```bash
# Upload the voxel viewer
scp voxel_volume_viewer_gpu.py ubuntu@YOUR_INSTANCE_IP:~/

# Upload your reconstruction file
scp video_voxel_out/recon_volume.npz ubuntu@YOUR_INSTANCE_IP:~/
```

### Option B: Using git (on Lambda instance)

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### Option C: Download from cloud storage

```bash
# If you have files on S3, Google Cloud, etc.
# aws s3 cp s3://your-bucket/recon_volume.npz ~/
# or
# wget https://your-url.com/recon_volume.npz
```

## Step 5: Run the GPU-Accelerated Viewer

```bash
# Make sure X11 forwarding is working
echo $DISPLAY  # Should show something like "localhost:10.0"

# Test X11 with a simple app
xclock  # A clock should appear on your local screen

# Run the GPU viewer
python3 voxel_volume_viewer_gpu.py recon_volume.npz
```

## Controls

- **V** - Switch to volumetric rendering mode (NeRF-style)
- **C** - Switch to cube mode (Minecraft-style with PyTorch3D)
- **Arrow keys** - Rotate view (left/right = yaw, up/down = pitch)
- **+/-** - Zoom in/out
- **ESC** - Quit

## Performance Tips

1. **Resolution**: The viewer renders at 256x256 internally. You can increase this for better quality:
   ```python
   # Edit line ~388 in the viewer
   render_H, render_W = 512, 512  # Higher resolution
   ```

2. **GPU Memory**: If you run out of memory with large voxel grids:
   - Reduce render resolution
   - Increase `thresh_factor` to show fewer voxels
   - Use a GPU with more VRAM (A100, H100)

3. **Latency**: X11 forwarding has some latency. For smoother experience:
   - Use compression: `ssh -X -C` (already enabled above)
   - Connect from a region close to the Lambda datacenter
   - Consider VNC instead (see below)

## Alternative: VNC Setup (Lower Latency)

If X11 forwarding is too slow, use VNC:

```bash
# On Lambda instance
sudo apt-get install -y tightvncserver

# Start VNC server
vncserver :1 -geometry 1920x1080 -depth 24

# Set a password when prompted

# Create SSH tunnel (from your local machine)
ssh -L 5901:localhost:5901 ubuntu@YOUR_INSTANCE_IP

# On your local machine, connect with VNC client to:
# localhost:5901
```

Then run the viewer inside the VNC session.

## Troubleshooting

### "cannot open display" error
```bash
# Check DISPLAY variable
echo $DISPLAY

# If empty, set it
export DISPLAY=:0

# Reconnect with ssh -X
```

### PyTorch3D import error
```bash
# Reinstall PyTorch3D
pip uninstall pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### CUDA out of memory
```bash
# Use CPU fallback
# The viewer will automatically fall back to CPU if PyTorch3D isn't available
# or manually set device to CPU in the code
```

### Slow rendering
- Reduce resolution
- Increase `thresh_factor` to show fewer voxels
- Use VNC instead of X11 forwarding

## Cost Optimization

Lambda charges by the hour. To minimize costs:

1. **Stop instance when not in use** (saves money but loses data)
2. **Download results** before stopping:
   ```bash
   scp ubuntu@YOUR_INSTANCE_IP:~/recon_volume.npz ./
   ```
3. **Use spot instances** if available (much cheaper but can be interrupted)

## Running the Reconstruction on Lambda

You can also run the full reconstruction pipeline on Lambda for faster training:

```bash
# Upload your video
scp your_video.mp4 ubuntu@YOUR_INSTANCE_IP:~/

# Upload the training script
scp video_orbit_voxel_recon.py ubuntu@YOUR_INSTANCE_IP:~/

# Run reconstruction (no viewer, just training)
python3 video_orbit_voxel_recon.py your_video.mp4 0 --neighbor-growth

# Download results
scp ubuntu@YOUR_INSTANCE_IP:~/video_voxel_out/recon_volume.npz ./
```

Then view locally or on Lambda with the GPU viewer!

