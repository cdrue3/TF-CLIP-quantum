#!/bin/bash
cd /home/ubuntu/TF-CLIP-quantum

echo "[$(date)] Waiting for swap test to finish..."
until grep -q "RESULTS SUMMARY\|Error\|Traceback" logs/quantum_retrieval_swap_classical/eval_log_4q.txt 2>/dev/null; do
    sleep 30
done
echo "[$(date)] Swap test done. Starting q-triplet training..."

mkdir -p logs/agvpreid_q_triplet_loss_20ep
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_q_triplet_loss.py \
    --config_file configs/vit_clipreid_agvpreid.yml \
    --n_qubits 6 --n_layers 1 \
    DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8 \
    SOLVER.STAGE2.MAX_EPOCHS 20 SOLVER.STAGE2.EVAL_PERIOD 999 \
    SOLVER.STAGE2.CHECKPOINT_PERIOD 10 \
    OUTPUT_DIR logs/agvpreid_q_triplet_loss_20ep 2>&1 | tee logs/agvpreid_q_triplet_loss_20ep/train_log.txt

echo "[$(date)] Q-triplet training done. Starting eval..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_agvpreid.py \
    --config_file configs/vit_clipreid_agvpreid.yml \
    --checkpoint logs/agvpreid_q_triplet_loss_20ep/checkpoint_ep20.pth.tar \
    DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8 2>&1 | tee logs/agvpreid_q_triplet_loss_20ep/eval_ep20.txt

echo "[$(date)] All done."
