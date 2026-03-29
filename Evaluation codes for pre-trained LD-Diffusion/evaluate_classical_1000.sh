#!/bin/bash
set -e

# Classical B32 Evaluation Script for 1000 kimg
CLASSICAL_RUN_DIR="/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/Training Codes of LD-Diffusion/training-runs-classical-b32/00000-panda_32-uncond-ncsnpp-edm-gpus1-batch32-fp32"
FID_REF="fid-refs/panda-32-real.npz"
REAL_IMGS="/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/panda_32_real/temp_real_32x32"
CALC_PR_SCRIPT="../calc_pr.py"
OUTPUT_REPORT="classical_b32_1000_report.txt"

# Checkpoint 1000 kimg
PKL_FILE="network-snapshot-001000.pkl"
FULL_PATH="$CLASSICAL_RUN_DIR/$PKL_FILE"
OUT_DIR="out_images_classical_b32_1000"

echo "Evaluating Classical B32 at 1000 kimg..."
echo "Checkpoint: $PKL_FILE"

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
    echo "Images already generated in $OUT_DIR."
fi

echo "Calculating Metrics..." > "$OUTPUT_REPORT"

# 2. FID
echo "Calculating FID..."
FID_OUT=$(python fid.py calc --images="$OUT_DIR" --ref="$FID_REF" --num=500)
echo "FID: $FID_OUT"
echo "FID: $FID_OUT" >> "$OUTPUT_REPORT"

# 3. Precision/Recall
echo "Calculating Precision/Recall..."
PR_OUT=$(python "$CALC_PR_SCRIPT" --real_dir="$REAL_IMGS" --fake_dir="$OUT_DIR" --k=3)
echo "$PR_OUT"
echo "$PR_OUT" >> "$OUTPUT_REPORT"

cat "$OUTPUT_REPORT"
