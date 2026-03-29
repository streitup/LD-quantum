#!/bin/bash
set -e

# Classical Comparison Experiment Script (Batch 32)
# Target Run: /home/zzn/qfl_tq/LD-Diffusion-quantum-v3/Training Codes of LD-Diffusion/training-runs-classical-b32/00000-panda_32-uncond-ncsnpp-edm-gpus1-batch32-fp32

# Paths
CLASSICAL_RUN_DIR="/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/Training Codes of LD-Diffusion/training-runs-classical-b32/00000-panda_32-uncond-ncsnpp-edm-gpus1-batch32-fp32"
FID_REF="fid-refs/panda-32-real.npz"
REAL_IMGS="/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/panda_32_real/temp_real_32x32"
CALC_PR_SCRIPT="../calc_pr.py"
OUTPUT_REPORT="classical_b32_comparison_report.txt"

# Checkpoints to evaluate (Epoch/Kimg -> Filename)
# New training config: tick=50 kimg. snap=20 ticks.
# Snapshot every 20 * 50 = 1000 kimg.
# Filenames are in ticks.
declare -A CHECKPOINTS
CHECKPOINTS["1000"]="network-snapshot-000020.pkl"
CHECKPOINTS["2000"]="network-snapshot-000040.pkl"
CHECKPOINTS["3000"]="network-snapshot-000060.pkl"
CHECKPOINTS["4000"]="network-snapshot-000080.pkl"
CHECKPOINTS["5000"]="network-snapshot-000100.pkl"

echo "Classical Comparison Report (Batch 32)" > "$OUTPUT_REPORT"
echo "======================================" >> "$OUTPUT_REPORT"
echo "Baseline: Classical CNN (Batch 32)" >> "$OUTPUT_REPORT"
echo "" >> "$OUTPUT_REPORT"

for EPOCH in 1000 2000 3000 4000 5000; do
    PKL_FILE="${CHECKPOINTS[$EPOCH]}"
    FULL_PATH="$CLASSICAL_RUN_DIR/$PKL_FILE"
    OUT_DIR="out_images_classical_b32_$EPOCH"
    
    echo "Processing Classical (B32) Epoch $EPOCH ($PKL_FILE)..."
    
    # Check if checkpoint exists
    if [ ! -f "$FULL_PATH" ]; then
        echo "Checkpoint $PKL_FILE not found yet. Skipping."
        continue
    fi
    
    # 1. Generate Images
    if [ ! -d "$OUT_DIR" ]; then
        echo "Generating 500 images..."
        python generate.py \
            --outdir="$OUT_DIR" \
            --seeds=0-499 \
            --batch=32 \
            --network="$FULL_PATH" \
            --mask_pos=True \
            --resolution=32 > /dev/null 2>&1
    else
        echo "Images already generated in $OUT_DIR. Skipping generation."
    fi
    
    echo "----------------------------------------" >> "$OUTPUT_REPORT"
    echo "Epoch: $EPOCH" >> "$OUTPUT_REPORT"
    
    # 2. Calculate FID
    echo "Calculating FID..."
    FID_OUT=$(python fid.py calc --images="$OUT_DIR" --ref="$FID_REF" --num=500)
    echo "$FID_OUT"
    echo "FID Output: $FID_OUT" >> "$OUTPUT_REPORT"
    
    # 3. Calculate Precision/Recall
    echo "Calculating Precision/Recall..."
    PR_OUT=$(python "$CALC_PR_SCRIPT" --real_dir="$REAL_IMGS" --fake_dir="$OUT_DIR" --k=3)
    echo "$PR_OUT"
    echo "Precision/Recall Output: $PR_OUT" >> "$OUTPUT_REPORT"
    
    echo "" >> "$OUTPUT_REPORT"
    echo "Done with Epoch $EPOCH."
done

echo "Comparison complete (for available checkpoints). Results saved to $OUTPUT_REPORT."
