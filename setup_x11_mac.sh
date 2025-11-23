#!/bin/bash
# Setup script for X11 forwarding on Mac
# Run this ON YOUR MAC

set -e

echo "=========================================="
echo "Setting up X11 on Mac for Lambda GPU"
echo "=========================================="

# Check if XQuartz is installed
if ! command -v xquartz &> /dev/null; then
    echo "Installing XQuartz..."
    if command -v brew &> /dev/null; then
        brew install --cask xquartz
    else
        echo "Error: Homebrew not found"
        echo "Please install XQuartz manually from: https://www.xquartz.org/"
        exit 1
    fi
else
    echo "✓ XQuartz is already installed"
fi

# Get Lambda IP
read -p "Enter your Lambda instance IP: " LAMBDA_IP

if [ -z "$LAMBDA_IP" ]; then
    echo "Error: No IP provided"
    exit 1
fi

# Create/update SSH config
echo ""
echo "Configuring SSH..."

SSH_CONFIG=~/.ssh/config

# Backup existing config
if [ -f "$SSH_CONFIG" ]; then
    cp "$SSH_CONFIG" "$SSH_CONFIG.backup"
    echo "✓ Backed up existing SSH config"
fi

# Check if lambda-gpu entry exists
if grep -q "Host lambda-gpu" "$SSH_CONFIG" 2>/dev/null; then
    echo "⚠ lambda-gpu entry already exists in SSH config"
    echo "  Skipping SSH config modification"
else
    # Add lambda-gpu configuration
    cat >> "$SSH_CONFIG" << EOF

# Lambda GPU instance for voxel masking
Host lambda-gpu
    HostName $LAMBDA_IP
    User ubuntu
    ForwardX11 yes
    ForwardX11Trusted yes
    Compression yes
    CompressionLevel 9
    TCPKeepAlive yes
    ServerAliveInterval 60
EOF
    echo "✓ Added lambda-gpu to SSH config"
fi

# Set proper permissions
chmod 600 "$SSH_CONFIG"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start XQuartz:"
echo "   open -a XQuartz"
echo ""
echo "2. In XQuartz preferences (XQuartz menu → Preferences):"
echo "   - Go to 'Security' tab"
echo "   - Check 'Allow connections from network clients'"
echo "   - Restart XQuartz"
echo ""
echo "3. Allow local connections:"
echo "   xhost + localhost"
echo ""
echo "4. Upload GPU masking tool to Lambda:"
echo "   scp voxel_masking_tool_gpu.py ubuntu@$LAMBDA_IP:~/"
echo "   scp video_voxel_out/recon_volume.npz ubuntu@$LAMBDA_IP:~/"
echo ""
echo "5. Connect to Lambda with X11:"
echo "   ssh -XC lambda-gpu"
echo ""
echo "6. On Lambda, test X11:"
echo "   xclock"
echo "   (You should see a clock appear on your Mac)"
echo ""
echo "7. Run the GPU masking tool:"
echo "   python voxel_masking_tool_gpu.py recon_volume.npz"
echo ""
echo "For detailed troubleshooting, see X11_FORWARDING_GUIDE.md"
echo ""


