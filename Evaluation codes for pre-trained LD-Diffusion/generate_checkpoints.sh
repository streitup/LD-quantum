#!/bin/bash
set -e

# Base directory for checkpoints
CHECKPOINT_DIR="/date/zzn_data/quantum-panda-32/00004-panda_32-uncond-ncsnpp-edm-gpus1-batch32-fp32"
# Script path
SCRIPT_DIR="/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/Evaluation codes for pre-trained LD-Diffusion"

# Checkpoints mapping (Approx epoch -> filename)
# 1000 epoch -> network-snapshot-001001.pkl
# 2000 epoch -> network-snapshot-002003.pkl
# 3000 epoch -> network-snapshot-003004.pkl
# 4000 epoch -> network-snapshot-004006.pkl

declare -A CHECKPOINTS
CHECKPOINTS["1000"]="network-snapshot-001001.pkl"
CHECKPOINTS["2000"]="network-snapshot-002003.pkl"
CHECKPOINTS["3000"]="network-snapshot-003004.pkl"
CHECKPOINTS["4000"]="network-snapshot-004006.pkl"
CHECKPOINTS["5000"]="network-snapshot-005008.pkl"

cd "$SCRIPT_DIR"

for EPOCH in 5000; do
    PKL_FILE="${CHECKPOINTS[$EPOCH]}"
    FULL_PATH="$CHECKPOINT_DIR/$PKL_FILE"
    OUT_DIR="out_images_$EPOCH"
    
    echo "Generating images for Epoch $EPOCH ($PKL_FILE)..."
    
    python generate.py \
        --outdir="$OUT_DIR" \
        --seeds=0-499 \
        --batch=64 \
        --network="$FULL_PATH" \
        --mask_pos=True \
        --resolution=32
        
    echo "Done generating for Epoch $EPOCH. Saved to $OUT_DIR"
done

echo "All generations complete."
