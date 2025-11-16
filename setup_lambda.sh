#!/bin/bash
# Setup script for Lambda GPU instances
# Run this on your Lambda instance after connecting

set -e

echo "=========================================="
echo "Lambda GPU Setup for Voxel Reconstruction"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pygame \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev \
    python3-dev \
    git

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install torch torchvision "numpy<2" opencv-python matplotlib pillow

echo ""
echo "Note: Using NumPy 1.x for compatibility with PyTorch"
echo ""

# Optional: Install pygame and PyTorch3D for GPU-accelerated viewer
# (Not needed for reconstruction, only for viewing)
# pip install pygame
# pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Verify installations
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if python3 -c "import cv2" 2>/dev/null; then
    echo "✓ OpenCV installed successfully"
else
    echo "✗ OpenCV installation failed"
    exit 1
fi

if python3 -c "import numpy" 2>/dev/null; then
    echo "✓ NumPy installed successfully"
else
    echo "✗ NumPy installation failed"
    exit 1
fi

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload reconstruction script: scp video_orbit_voxel_recon.py ubuntu@YOUR_IP:~/"
echo "2. Upload your video: scp your_video.mp4 ubuntu@YOUR_IP:~/"
echo "3. Run reconstruction: python3 video_orbit_voxel_recon.py your_video.mp4 0 --neighbor-growth"
echo "4. Download results: scp ubuntu@YOUR_IP:~/video_voxel_out/recon_volume.npz ./"
echo ""
echo "The reconstruction will automatically use GPU and be ~10x faster!"

