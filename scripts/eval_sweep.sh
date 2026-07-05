#!/bin/bash
# Usage: bash scripts/eval_sweep.sh <log_dir> <output_file>
# e.g.   bash scripts/eval_sweep.sh logs/agvpreid_qtemporal_deep/40ep /tmp/deep_sweep.txt
cd /home/ubuntu/TF-CLIP-quantum
LOG_DIR=$1
OUT=$2

for ckpt in $LOG_DIR/checkpoint_ep*.pth.tar; do
    [ -f "$ckpt" ] || continue
    echo "=== $ckpt ===" | tee -a "$OUT"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_agvpreid.py \
        --config_file configs/vit_clipreid_agvpreid.yml \
        --checkpoint "$ckpt" \
        --case 1 \
        DATASETS.ROOT_DIR DATA/subset_250 \
        INPUT.SEQ_LEN 8 2>&1 \
        | grep -E "Rank-1|Rank-5|mAP|Error|Traceback" | tee -a "$OUT"
done
echo "=== SWEEP DONE: $LOG_DIR ===" | tee -a "$OUT"
