#!/bin/bash
# Chain: ring n3 80ep (with inline eval every 5ep) → SQP+BER 80ep (with inline eval every 5ep)
# Eval runs inline after each epoch % 5 == 0 — results visible in train_log.txt live.
# Kill this script (or the active training PID) to stop early if R1 has plateaued.

set -e
cd /home/ubuntu/TF-CLIP-quantum

RING_N3_PID=9612_PLACEHOLDER  # overwritten below — ring n3 is started fresh in this script

RING_N3_DIR=logs/agvpreid_qtemporal_ent/40ep_ring_n3
BER_DIR=logs/agvpreid_sqp_ber/80ep_n2

mkdir -p "$RING_N3_DIR" "$BER_DIR"

# ── Ring n3 80ep with eval every 5ep ────────────────────────────────────────
echo "[chain] $(date) — starting ring n3 80ep training (eval every 5ep)..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_qtemporal_ent.py \
    --config_file configs/vit_clipreid_agvpreid.yml \
    --entanglement ring --n_layers 3 \
    SOLVER.STAGE2.MAX_EPOCHS 80 \
    SOLVER.STAGE2.EVAL_PERIOD 5 \
    SOLVER.STAGE2.CHECKPOINT_PERIOD 5 \
    DATASETS.ROOT_DIR DATA/subset_250 \
    OUTPUT_DIR "$RING_N3_DIR" \
    2>&1 | tee "$RING_N3_DIR/train_log.txt"

echo "[chain] $(date) — ring n3 training done."

# ── SQP+BER 80ep with eval every 5ep ────────────────────────────────────────
echo "[chain] $(date) — starting SQP+BER 80ep training (eval every 5ep)..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_qtemporal_ber.py \
    --config_file configs/vit_clipreid_agvpreid.yml \
    --n_qubits 8 \
    --n_layers 2 \
    --entropy_reg 0.02 \
    --noise_sigma 0.15 \
    --noise_epochs 20 \
    SOLVER.STAGE2.MAX_EPOCHS 80 \
    SOLVER.STAGE2.EVAL_PERIOD 5 \
    SOLVER.STAGE2.CHECKPOINT_PERIOD 5 \
    DATASETS.ROOT_DIR DATA/subset_250 \
    OUTPUT_DIR "$BER_DIR" \
    2>&1 | tee "$BER_DIR/train_log.txt"

echo "[chain] $(date) — SQP+BER training done."
echo "[chain] $(date) — ALL DONE."
