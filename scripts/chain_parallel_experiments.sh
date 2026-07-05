#!/bin/bash
# Queue: 1) parallel+ham  2) parallel+gated  3) parallel+deep
set -e
cd /home/ubuntu/TF-CLIP-quantum

YACS_OPTS="SOLVER.STAGE2.MAX_EPOCHS 80 \
    SOLVER.STAGE2.EVAL_PERIOD 5 \
    SOLVER.STAGE2.CHECKPOINT_PERIOD 5 \
    DATASETS.ROOT_DIR DATA/subset_250"

BER_OPTS="--config_file configs/vit_clipreid_agvpreid.yml \
    --entropy_reg 0.02 --noise_sigma 0.15 --noise_epochs 20"

run_experiment() {
    local name=$1; shift
    local outdir="logs/agvpreid_sqp_ber/$name"
    local evalout="logs/eval/${name}_sweep.txt"
    mkdir -p "$outdir"
    echo "[chain] $(date) — starting $name..."
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python train_qtemporal_ber.py $BER_OPTS "$@" \
        $YACS_OPTS \
        OUTPUT_DIR "$outdir" \
        2>&1 | tee "$outdir/train_log.txt"
    echo "[chain] $(date) — $name training done. Eval sweep..."
    bash scripts/eval_sweep.sh "$outdir" "$evalout"
    echo "[chain] $(date) — $name sweep done."
}

# 1) Parallel + Hamiltonian (n_qubits=5 for ham)
run_experiment "80ep_parallel_ham" \
    --n_qubits 5 --n_layers 2 --parallel --hamiltonian

# 2) Parallel + Gated fusion
run_experiment "80ep_parallel_gated" \
    --n_qubits 8 --n_layers 2 --parallel --fusion_mode gated

# 3) Parallel + deep quantum branch (n_layers=3)
run_experiment "80ep_parallel_deep" \
    --n_qubits 8 --n_layers 3 --parallel

echo "[chain] $(date) — ALL EXPERIMENTS DONE."
