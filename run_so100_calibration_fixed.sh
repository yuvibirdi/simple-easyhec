#!/bin/bash
# Fixed SO100 calibration script
# This fixes the major issues causing 56k loss

set -e

echo "🔧 Running SO100 calibration with fixes applied"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Key fixes:"
echo "  ✓ Resolution: 640x480 (matching camera_intrinsic.json)"
echo "  ✓ Batch size: None (matching paper example)"
echo "  ✓ More calibration poses (6 instead of 2)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sam

# Run the fixed calibration
cd /home/yb/git/lerobot-sim2real/simple-easyhec

python easyhec/examples/real/so100_fixed.py \
    --camera-intrinsics-path camera_intrinsic.json \
    --opencv-camera-id 0 \
    --output-dir results/so100_fixed \
    --train-steps 5000 \
    --early-stopping-steps 200

echo ""
echo "✅ Calibration complete! Check results/so100_fixed/ for outputs"
