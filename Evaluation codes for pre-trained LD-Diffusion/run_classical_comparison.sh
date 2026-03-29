#!/bin/bash
set -e

# Classical Comparison Experiment Script
# Uses existing classical run: /date/zzn_data/classical-panda-32/00002-panda_32-uncond-ncsnpp-edm-gpus1-batch64-fp32

# Paths
CLASSICAL_RUN_DIR="/date/zzn_data/classical-panda-32/00002-panda_32-uncond-ncsnpp-edm-gpus1-batch64-fp32"
FID_REF="fid-refs/panda-32-real.npz"
REAL_IMGS="/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/panda_32_real/temp_real_32x32"
CALC_PR_SCRIPT="../calc_pr.py"
OUTPUT_REPORT="classical_comparison_report.txt"

# Checkpoints to evaluate (Epoch/Kimg -> Filename)
declare -A CHECKPOINTS
CHECKPOINTS["1000"]="network-snapshot-001001.pkl"
CHECKPOINTS["2000"]="network-snapshot-002001.pkl"
CHECKPOINTS["3000"]="network-snapshot-003002.pkl"
CHECKPOINTS["4000"]="network-snapshot-004003.pkl"
CHECKPOINTS["5000"]="network-snapshot-005004.pkl"

echo "Classical Comparison Report" > "$OUTPUT_REPORT"
echo "===========================" >> "$OUTPUT_REPORT"
echo "Baseline: Classical CNN (Batch 64)" >> "$OUTPUT_REPORT"
echo "" >> "$OUTPUT_REPORT"

for EPOCH in 1000 2000 3000 4000 5000; do
    PKL_FILE="${CHECKPOINTS[$EPOCH]}"
    FULL_PATH="$CLASSICAL_RUN_DIR/$PKL_FILE"
    OUT_DIR="out_images_classical_$EPOCH"
    
    echo "Processing Classical Epoch $EPOCH ($PKL_FILE)..."
    
    # 1. Generate Images
    if [ ! -d "$OUT_DIR" ]; then
        echo "Generating 500 images..."
        # We assume generate.py is in current dir
        python generate.py             --outdir="$OUT_DIR"             --seeds=0-499             --batch=64             --network="$FULL_PATH"             --mask_pos=True             --resolution=32 > /dev/null 2>&1
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

echo "Comparison complete. Results saved to $OUTPUT_REPORT."
cat "$OUTPUT_REPORT"
