#!/bin/bash
# Setup script for X11 forwarding on Lambda GPU instance
# Run this ON THE LAMBDA INSTANCE

set -e

echo "=========================================="
echo "Setting up X11 for GPU Masking Tool"
echo "=========================================="

# Install X11 dependencies
echo "Installing X11 libraries..."
sudo apt-get update
sudo apt-get install -y \
    xauth \
    x11-apps \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxrandr-dev \
    libxi-dev \
    libxcursor-dev \
    libxinerama-dev

# Verify SSHD config
echo ""
echo "Checking SSH X11 forwarding configuration..."
if grep -q "^X11Forwarding yes" /etc/ssh/sshd_config; then
    echo "✓ X11Forwarding is enabled"
else
    echo "⚠ X11Forwarding not explicitly enabled"
    echo "  (It may be enabled by default)"
fi

# Test X11 availability
echo ""
echo "Testing X11 setup..."
if [ -z "$DISPLAY" ]; then
    echo "⚠ DISPLAY not set"
    echo "  This is normal if you haven't connected with SSH -X yet"
else
    echo "✓ DISPLAY is set to: $DISPLAY"
fi

# Make sure PyGame can find X11
echo ""
echo "Configuring SDL/PyGame for X11..."
echo "export SDL_VIDEODRIVER=x11" >> ~/.bashrc
echo "export SDL_AUDIODRIVER=dummy" >> ~/.bashrc

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. On your Mac, install XQuartz: brew install --cask xquartz"
echo "2. On your Mac, start XQuartz and enable network connections"
echo "3. Disconnect from this SSH session"
echo "4. Reconnect with: ssh -XC ubuntu@YOUR_LAMBDA_IP"
echo "5. Test with: xclock (should show a clock on your Mac)"
echo "6. Run: python voxel_masking_tool_gpu.py recon_volume.npz"
echo ""
echo "For detailed instructions, see X11_FORWARDING_GUIDE.md"
echo ""


