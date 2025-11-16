#!/bin/bash
# Setup script for Lambda GPU instances
# Run this on your Lambda instance after connecting

set -e

echo "=========================================="
echo "Lambda GPU Setup for Voxel Viewer"
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
pip install torch torchvision numpy opencv-python pygame matplotlib pillow

# Install PyTorch3D (GPU-accelerated rendering)
echo "Installing PyTorch3D (this may take 5-10 minutes)..."
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Verify installations
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if python3 -c "import pytorch3d" 2>/dev/null; then
    echo "✓ PyTorch3D installed successfully"
else
    echo "✗ PyTorch3D installation failed"
    exit 1
fi

if python3 -c "import pygame" 2>/dev/null; then
    echo "✓ Pygame installed successfully"
else
    echo "✗ Pygame installation failed"
    exit 1
fi

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your code: scp voxel_volume_viewer_gpu.py ubuntu@YOUR_IP:~/"
echo "2. Upload your data: scp recon_volume.npz ubuntu@YOUR_IP:~/"
echo "3. Run viewer: python3 voxel_volume_viewer_gpu.py recon_volume.npz"
echo ""
echo "Make sure you connected with: ssh -X -C ubuntu@YOUR_IP"

