#!/bin/bash
# batch_train_8gpu.sh
# Train multiple videos in parallel on 8-GPU instance (true 8x throughput)

if [ $# -lt 1 ]; then
    echo "Usage: $0 <video1.mp4> [video2.mp4] [video3.mp4] ..."
    echo ""
    echo "Example: Train 8 videos in parallel on 8-GPU instance:"
    echo "  $0 video1.mp4 video2.mp4 video3.mp4 video4.mp4 \\"
    echo "     video5.mp4 video6.mp4 video7.mp4 video8.mp4"
    echo ""
    echo "This gives TRUE 8x speedup (each GPU works independently)"
    exit 1
fi

VIDEOS=("$@")
NUM_VIDEOS=${#VIDEOS[@]}
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "============================================================"
echo "Batch Training on Multi-GPU"
echo "============================================================"
echo "Videos to process: $NUM_VIDEOS"
echo "Available GPUs: $NUM_GPUS"
echo ""

if [ $NUM_VIDEOS -gt $NUM_GPUS ]; then
    echo "⚠️  WARNING: More videos ($NUM_VIDEOS) than GPUs ($NUM_GPUS)"
    echo "   Will process in batches of $NUM_GPUS"
fi

echo "Starting parallel training..."
echo ""

# Function to train a single video on a specific GPU
train_video() {
    local gpu_id=$1
    local video=$2
    local video_name=$(basename "$video" .mp4)
    local output_dir="video_voxel_out_${video_name}"
    
    echo "[GPU $gpu_id] Starting: $video → $output_dir"
    
    # Create output directory and run from there
    mkdir -p "$output_dir"
    cd "$output_dir"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python ../video_orbit_voxel_recon.py \
        "../$video" 0 \
        > "../${output_dir}.log" 2>&1
    
    cd ..
    
    if [ $? -eq 0 ]; then
        echo "[GPU $gpu_id] ✓ Complete: $video"
    else
        echo "[GPU $gpu_id] ✗ Failed: $video (check ${output_dir}.log)"
    fi
}

# Launch videos on GPUs in parallel
pids=()
for i in $(seq 0 $((NUM_VIDEOS - 1))); do
    gpu_id=$((i % NUM_GPUS))
    video="${VIDEOS[$i]}"
    
    # If we've filled all GPUs, wait for the batch to complete
    if [ $i -ge $NUM_GPUS ] && [ $((i % NUM_GPUS)) -eq 0 ]; then
        echo ""
        echo "Waiting for batch to complete..."
        for pid in "${pids[@]}"; do
            wait $pid
        done
        pids=()
        echo "Batch complete. Starting next batch..."
        echo ""
    fi
    
    # Launch training in background
    train_video $gpu_id "$video" &
    pids+=($!)
    
    # Small delay to avoid race conditions
    sleep 2
done

# Wait for final batch
echo ""
echo "Waiting for final batch..."
for pid in "${pids[@]}"; do
    wait $pid
done

echo ""
echo "============================================================"
echo "All videos processed!"
echo "============================================================"
echo ""
echo "Results:"
for video in "${VIDEOS[@]}"; do
    video_name=$(basename "$video" .mp4)
    output_dir="video_voxel_out_${video_name}"
    if [ -f "${output_dir}/recon_volume.npz" ]; then
        echo "  ✓ $video_name → $output_dir/"
    else
        echo "  ✗ $video_name (failed, check ${output_dir}.log)"
    fi
done
echo ""

