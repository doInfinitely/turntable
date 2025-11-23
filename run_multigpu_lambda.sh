#!/bin/bash
# run_multigpu_lambda.sh
# Deploy and run multi-GPU training on Lambda

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <lambda-ip> <video-path-on-lambda>"
    echo ""
    echo "Example:"
    echo "  $0 167.234.213.62 ~/turntable/video.mp4"
    echo ""
    echo "Steps:"
    echo "  1. Syncs code to Lambda instance"
    echo "  2. Tests GPU detection"
    echo "  3. Runs training with all available GPUs"
    exit 1
fi

LAMBDA_IP=$1
VIDEO_PATH=$2
REMOTE_USER="ubuntu"

echo "============================================================"
echo "Multi-GPU Training on Lambda"
echo "============================================================"
echo "Target: $REMOTE_USER@$LAMBDA_IP"
echo "Video: $VIDEO_PATH"
echo ""

# Step 1: Sync code
echo "Step 1/3: Syncing code to Lambda..."
rsync -avz --progress \
    --exclude 'video_voxel_out/' \
    --exclude '*.mp4' \
    --exclude '__pycache__/' \
    --exclude '.git/' \
    ./ $REMOTE_USER@$LAMBDA_IP:~/turntable/

echo ""
echo "✓ Code synced"
echo ""

# Step 2: Test GPU detection
echo "Step 2/3: Testing GPU detection..."
ssh $REMOTE_USER@$LAMBDA_IP "cd ~/turntable && python test_multigpu.py"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ GPU test failed. Please check your Lambda instance."
    exit 1
fi

echo ""
echo "✓ GPU test passed"
echo ""

# Step 3: Run training
echo "Step 3/3: Starting training..."
echo ""
echo "Training will run with:"
echo "  - Resolution: 128×128×128 voxels (2M voxels)"
echo "  - Iterations: 8000"
echo "  - L1 regularization: 3e-2 (aggressive anti-cloud)"
echo "  - TV smoothness: 2e-3 (density), 1e-3 (color)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting training..."
echo "  Expected time: ~5-10 minutes on 8 GPUs, ~40 minutes on 1 GPU"
echo ""
echo "Monitor GPU usage with:"
echo "  ssh $REMOTE_USER@$LAMBDA_IP 'watch -n 0.5 nvidia-smi'"
echo ""

ssh $REMOTE_USER@$LAMBDA_IP "cd ~/turntable && python video_orbit_voxel_recon.py $VIDEO_PATH 0"

echo ""
echo "============================================================"
echo "Training complete!"
echo "============================================================"
echo ""
echo "Download results:"
echo "  scp $REMOTE_USER@$LAMBDA_IP:~/turntable/video_voxel_out/recon_volume.npz ."
echo "  scp $REMOTE_USER@$LAMBDA_IP:~/turntable/video_voxel_out/recon_voxels.ply ."
echo ""
echo "View locally:"
echo "  python voxel_volume_viewer.py video_voxel_out/recon_volume.npz"
echo ""

