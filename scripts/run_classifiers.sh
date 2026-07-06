#!/bin/bash
set -e
cd /home/ubuntu/TF-CLIP-quantum

echo "=== [1/4] QClassifier dense_angle 8q no-boost 80ep ==="
python train_qclassifier.py --config_file configs/vit_clipreid_agvpreid.yml \
    --n_qubits 8 --n_layers 2 --encoding dense_angle \
    SOLVER.STAGE2.MAX_EPOCHS 80 SOLVER.STAGE2.EVAL_PERIOD 999 SOLVER.STAGE2.CHECKPOINT_PERIOD 10 \
    DATASETS.ROOT_DIR DATA/subset_250 OUTPUT_DIR logs/agvpreid_qclassifier/dense_8q_80ep_noboost

echo "=== [2/4] Eval sweep: dense classifier ==="
bash scripts/eval_sweep.sh logs/agvpreid_qclassifier/dense_8q_80ep_noboost /tmp/qclassifier_dense_noboost_sweep.txt

echo "=== [3/4] QClassifier hamiltonian 5q no-boost 80ep ==="
python train_qclassifier_ham.py --config_file configs/vit_clipreid_agvpreid.yml \
    --n_qubits 5 --n_layers 2 \
    SOLVER.STAGE2.MAX_EPOCHS 80 SOLVER.STAGE2.EVAL_PERIOD 999 SOLVER.STAGE2.CHECKPOINT_PERIOD 10 \
    DATASETS.ROOT_DIR DATA/subset_250 OUTPUT_DIR logs/agvpreid_qclassifier_ham/5q_80ep_noboost

echo "=== [4/4] Eval sweep: ham classifier ==="
bash scripts/eval_sweep.sh logs/agvpreid_qclassifier_ham/5q_80ep_noboost /tmp/qclassifier_ham_noboost_sweep.txt

echo "=== ALL DONE ==="
