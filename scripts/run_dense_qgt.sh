#!/bin/bash
set -e
cd /home/ubuntu/TF-CLIP-quantum

echo "=== [1/4] Dense 8q no-boost 80ep ==="
python train_qtemporal_dense.py --config_file configs/vit_clipreid_agvpreid.yml --n_qubits 8 \
    SOLVER.STAGE2.MAX_EPOCHS 80 SOLVER.STAGE2.EVAL_PERIOD 999 SOLVER.STAGE2.CHECKPOINT_PERIOD 10 \
    DATASETS.ROOT_DIR DATA/subset_250 OUTPUT_DIR logs/agvpreid_qtemporal_dense/8q_80ep_noboost

echo "=== [2/4] Eval sweep: dense 8q ==="
bash scripts/eval_sweep.sh logs/agvpreid_qtemporal_dense/8q_80ep_noboost /tmp/dense_8q_noboost_sweep.txt

echo "=== [3/4] QGT no-boost 80ep ==="
python train_qgt.py --config_file configs/vit_clipreid_agvpreid.yml \
    SOLVER.STAGE2.MAX_EPOCHS 80 SOLVER.STAGE2.EVAL_PERIOD 999 SOLVER.STAGE2.CHECKPOINT_PERIOD 10 \
    DATASETS.ROOT_DIR DATA/subset_250 OUTPUT_DIR logs/agvpreid_qgt/80ep_noboost

echo "=== [4/4] Eval sweep: QGT ==="
bash scripts/eval_sweep.sh logs/agvpreid_qgt/80ep_noboost /tmp/qgt_noboost_sweep.txt

echo "=== ALL DONE ==="
