#!/bin/bash
set -e

# Script to calculate FID and Precision/Recall for generated images

# Directories
REF_IMGS="/home/zzn/qfl_tq/LD-Diffusion-quantum-v3/panda_32_real/temp_real_32x32"
FID_REF="fid-refs/panda-32-real.npz"
CALC_PR_SCRIPT="../calc_pr.py"
OUTPUT_FILE="metrics_report.txt"

# Ensure calc_pr.py is executable or run with python
# Assuming python environment is set

echo "Metrics Calculation Report" > "$OUTPUT_FILE"
echo "==========================" >> "$OUTPUT_FILE"
echo "Reference Images: $REF_IMGS" >> "$OUTPUT_FILE"
echo "FID Reference Stats: $FID_REF" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

for EPOCH in 5000; do
    GEN_DIR="out_images_$EPOCH"
    
    if [ ! -d "$GEN_DIR" ]; then
        echo "Warning: Directory $GEN_DIR not found. Skipping."
        continue
    fi
    
    echo "Processing Epoch $EPOCH ($GEN_DIR)..."
    echo "----------------------------------------" >> "$OUTPUT_FILE"
    echo "Epoch: $EPOCH" >> "$OUTPUT_FILE"
    
    # 1. Calculate FID
    echo "Calculating FID..."
    # Capture FID output
    FID_OUT=$(python fid.py calc --images="$GEN_DIR" --ref="$FID_REF" --num=500)
    echo "$FID_OUT"
    # Extract FID value if possible, or just save the output
    echo "FID Output:" >> "$OUTPUT_FILE"
    echo "$FID_OUT" >> "$OUTPUT_FILE"
    
    # 2. Calculate Precision and Recall
    echo "Calculating Precision and Recall..."
    # Capture PR output
    PR_OUT=$(python "$CALC_PR_SCRIPT" --real_dir="$REF_IMGS" --fake_dir="$GEN_DIR" --k=3)
    echo "$PR_OUT"
    echo "Precision/Recall Output:" >> "$OUTPUT_FILE"
    echo "$PR_OUT" >> "$OUTPUT_FILE"
    
    echo "" >> "$OUTPUT_FILE"
    echo "Done with Epoch $EPOCH."
done

echo "All metrics calculated. Results saved to $OUTPUT_FILE."
cat "$OUTPUT_FILE"
