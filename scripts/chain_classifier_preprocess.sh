#!/bin/bash
# QClassifier dense_angle qubit sweep (4/6/8) → QPCA preprocessing
# All with no LR boost, GPU fix applied
set -e
cd /home/ubuntu/TF-CLIP-quantum

for NQ in 4 6 8; do
    OUTDIR="logs/agvpreid_qclassifier/80ep_dense_nq${NQ}"
    mkdir -p "$OUTDIR"
    echo "[chain] $(date) — QClassifier dense_angle n_qubits=${NQ} 80ep..."
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python train_qclassifier.py \
        --config_file configs/vit_clipreid_agvpreid.yml \
        --n_qubits $NQ --n_layers 2 --encoding dense_angle \
        SOLVER.STAGE2.MAX_EPOCHS 80 \
        SOLVER.STAGE2.EVAL_PERIOD 5 \
        SOLVER.STAGE2.CHECKPOINT_PERIOD 5 \
        DATASETS.ROOT_DIR DATA/subset_250 \
        OUTPUT_DIR "$OUTDIR" \
        2>&1 | tee "$OUTDIR/train_log.txt"
    echo "[chain] $(date) — sweep nq${NQ}..."
    bash scripts/eval_sweep.sh "$OUTDIR" "logs/eval/qclassifier_dense_nq${NQ}_sweep.txt"
done

OUTDIR="logs/agvpreid_qpreprocess/80ep_noboost"
mkdir -p "$OUTDIR"
echo "[chain] $(date) — QPCA preprocessing 80ep..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_qpreprocess.py \
    --config_file configs/vit_clipreid_agvpreid.yml \
    --n_qubits 8 --n_layers 2 \
    SOLVER.STAGE2.MAX_EPOCHS 80 \
    SOLVER.STAGE2.EVAL_PERIOD 5 \
    SOLVER.STAGE2.CHECKPOINT_PERIOD 5 \
    DATASETS.ROOT_DIR DATA/subset_250 \
    OUTPUT_DIR "$OUTDIR" \
    2>&1 | tee "$OUTDIR/train_log.txt"

echo "[chain] $(date) — QPCA sweep..."
bash scripts/eval_sweep.sh "$OUTDIR" "logs/eval/qpreprocess_noboost_sweep.txt"

echo "[chain] $(date) — ALL DONE."
