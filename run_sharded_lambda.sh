#!/bin/bash
# run_sharded_lambda.sh
# Deploy and run 1024³ sharded training on Lambda 8-GPU instance

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <lambda-ip> <video-path>"
    echo ""
    echo "Example:"
    echo "  $0 159.54.172.71 ~/video.mp4"
    echo ""
    echo "This will:"
    echo "  1. Sync code to Lambda"
    echo "  2. Test GPU detection (must have 8 GPUs)"
    echo "  3. Run 1024³ sharded training (~30 minutes)"
    echo "  4. Download results"
    exit 1
fi

LAMBDA_IP=$1
VIDEO_PATH=$2
REMOTE_USER="ubuntu"

echo "============================================================"
echo "Sharded Volume Training (1024³ across 8 GPUs)"
echo "============================================================"
echo "Target: $REMOTE_USER@$LAMBDA_IP"
echo "Video: $VIDEO_PATH"
echo ""

# Step 1: Sync code
echo "Step 1/4: Syncing code to Lambda..."
rsync -avz --progress \
    --exclude 'video_voxel_out*/' \
    --exclude '*.mp4' \
    --exclude '__pycache__/' \
    --exclude '.git/' \
    ./ $REMOTE_USER@$LAMBDA_IP:~/turntable/

echo ""
echo "✓ Code synced"
echo ""

# Step 2: Test GPU detection
echo "Step 2/4: Testing GPU detection..."
ssh $REMOTE_USER@$LAMBDA_IP << 'EOF'
cd ~/turntable
python << 'PYEOF'
import torch
n_gpus = torch.cuda.device_count()
print(f"GPUs detected: {n_gpus}")
if n_gpus != 8:
    print(f"ERROR: Sharded mode requires exactly 8 GPUs, but found {n_gpus}")
    exit(1)
print("✓ All 8 GPUs detected")
for i in range(n_gpus):
    name = torch.cuda.get_device_name(i)
    mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {name} ({mem_gb:.0f} GB)")
PYEOF
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ GPU test failed. Sharded mode requires exactly 8 GPUs."
    exit 1
fi

echo ""
echo "✓ GPU test passed"
echo ""

# Step 3: Run training
echo "Step 3/4: Starting sharded training..."
echo ""
echo "Training configuration:"
echo "  Resolution: 1024×1024×1024 voxels (1 billion voxels!)"
echo "  Iterations: 8000"
echo "  Expected time: ~30 minutes"
echo "  GPU memory: ~2 GB per GPU"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting training..."
echo ""
echo "Monitor GPU usage with:"
echo "  ssh $REMOTE_USER@$LAMBDA_IP 'watch -n 0.5 nvidia-smi'"
echo ""
echo "Press Ctrl+C to interrupt (but training will stop)"
echo ""

# Run training
ssh $REMOTE_USER@$LAMBDA_IP "cd ~/turntable && python video_orbit_voxel_recon.py $VIDEO_PATH 0 --sharded"

echo ""
echo "✓ Training complete!"
echo ""

# Step 4: Download results
echo "Step 4/4: Downloading results..."

# Create local output directory
mkdir -p video_voxel_out_1024

# Download results
echo "Downloading volume..."
scp $REMOTE_USER@$LAMBDA_IP:~/turntable/video_voxel_out/recon_volume.npz \
    video_voxel_out_1024/

echo "Downloading voxels..."
scp $REMOTE_USER@$LAMBDA_IP:~/turntable/video_voxel_out/recon_voxels.ply \
    video_voxel_out_1024/

echo ""
echo "============================================================"
echo "Sharded Training Complete!"
echo "============================================================"
echo ""
echo "Results saved to: video_voxel_out_1024/"
echo ""
echo "View the results:"
echo "  python voxel_volume_viewer.py video_voxel_out_1024/recon_volume.npz"
echo ""
echo "Volume statistics:"
python3 << 'PYEOF'
import numpy as np
vol = np.load('video_voxel_out_1024/recon_volume.npz')
sigma = vol['sigma']
rgb = vol['rgb']
print(f"  Grid size: {sigma.shape}")
print(f"  Total voxels: {sigma.size:,}")
print(f"  Non-empty voxels (σ>0.1): {(sigma > 0.1).sum():,}")
print(f"  Memory: {(sigma.nbytes + rgb.nbytes) / 1e9:.2f} GB")
PYEOF

echo ""
echo "Cleanup Lambda instance:"
echo "  ssh $REMOTE_USER@$LAMBDA_IP 'rm -rf ~/turntable/video_voxel_out'"
echo ""

