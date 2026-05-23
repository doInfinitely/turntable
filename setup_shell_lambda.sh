#!/bin/bash
# setup_shell_lambda.sh
# Minimal setup for running gaussian_shell.py on a Lambda GPU instance
# (assumes Lambda's ML AMI which already has CUDA + recent PyTorch).
#
# Usage:
#   chmod +x setup_shell_lambda.sh
#   ./setup_shell_lambda.sh

set -e

echo "=========================================="
echo "GaussianShell setup for Lambda A100 80GB"
echo "=========================================="

# Pinned-pip + system deps (ffmpeg is needed by view_shell.py for .webm)
echo "[1/4] Installing system deps (ffmpeg)..."
sudo apt-get update -qq
sudo apt-get install -y ffmpeg

# Python deps — most are already on the Lambda ML AMI; this is idempotent
echo "[2/4] Installing Python deps..."
pip install --upgrade pip
pip install \
    torch \
    "numpy<2" \
    opencv-python \
    "pillow>=9.1.0" \
    requests

# Verify GPU count + CUDA
echo "[3/4] Verifying GPUs..."
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count:      {n}')
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name}, {p.total_memory/1e9:.1f} GB')
assert n >= 1, 'No GPU detected'
"

# Quick import smoke test
echo "[4/4] Importing gaussian_shell..."
python3 -c "
from gaussian_shell import GaussianShell, render_shell_dispatch, _make_shell_replicas
print('  imports OK')
shell = GaussianShell(n_per_shell=100, n_shells=2).cuda()
print('  shell instantiation OK on cuda:0')
"

echo ""
echo "=========================================="
echo "Setup complete. Suggested next step:"
echo "=========================================="
cat <<'EOF'

# Real-quality run on 8× A100 80GB (~6h, ~$62 on Lambda):
python3 -u gaussian_shell.py \
  --video YOUR_VIDEO.mp4 \
  --n-shells 4 --init-radii 0.4 0.8 1.2 1.6 \
  --n-per-shell 32000 \
  --img-res 384 384 \
  --n-iters 4000 \
  --views-per-iter 64 \
  --tile-size 32 --tile-mode grid \
  --chunk-size 4096 \
  --n-gpus 8 \
  --snapshot-every 200 \
  --out-dir shell_out 2>&1 | tee shell_train.log

# If multi-GPU hangs or behaves strangely on first run, add --debug-sync
# (runs per-GPU workers sequentially — slower, easier to debug).

# Before tearing down the instance:
#   1. Final outputs:    shell_out/shell.npz, shell.ply, final.turntable.webm
#   2. Snapshots:        shell_out/snapshots/iter*_shell.npz, .png
#   Upload to S3 / scp them off — Lambda storage is ephemeral.
EOF
