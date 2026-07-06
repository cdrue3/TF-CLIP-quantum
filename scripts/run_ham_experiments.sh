#!/bin/bash
set -e
cd /home/ubuntu/TF-CLIP-quantum

echo "=== [1/4] Ham fair (no LR boost) 80ep ==="
python train_qtd_ham.py --config_file configs/vit_clipreid_agvpreid.yml --lr_mode none \
    SOLVER.STAGE2.MAX_EPOCHS 80 SOLVER.STAGE2.EVAL_PERIOD 999 SOLVER.STAGE2.CHECKPOINT_PERIOD 10 \
    DATASETS.ROOT_DIR DATA/subset_250 OUTPUT_DIR logs/agvpreid_qtd_ham_80ep_fair_v2

echo "=== [2/4] Eval sweep: fair ==="
bash scripts/eval_sweep.sh logs/agvpreid_qtd_ham_80ep_fair_v2 /tmp/ham_fair_v2_sweep.txt

echo "=== [3/4] Ham boost-decay (3x/10x → 1x at ep30) 80ep ==="
python train_qtd_ham.py --config_file configs/vit_clipreid_agvpreid.yml --lr_mode decay \
    SOLVER.STAGE2.MAX_EPOCHS 80 SOLVER.STAGE2.EVAL_PERIOD 999 SOLVER.STAGE2.CHECKPOINT_PERIOD 10 \
    DATASETS.ROOT_DIR DATA/subset_250 OUTPUT_DIR logs/agvpreid_qtd_ham_80ep_boost_decay

echo "=== [4/4] Eval sweep: boost-decay ==="
bash scripts/eval_sweep.sh logs/agvpreid_qtd_ham_80ep_boost_decay /tmp/ham_boost_decay_sweep.txt

echo "=== ALL DONE ==="
