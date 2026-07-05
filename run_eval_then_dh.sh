#!/bin/bash
cd /home/ubuntu/TF-CLIP-quantum

echo "[$(date)] Starting QGT eval..."
mkdir -p logs/agvpreid_qgt_40ep
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_agvpreid.py \
    --config_file configs/vit_clipreid_agvpreid.yml \
    --checkpoint logs/agvpreid_qgt_40ep/checkpoint_ep40.pth.tar \
    DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8 2>&1 | tee logs/agvpreid_qgt_40ep/eval_ep40.txt

echo "[$(date)] QGT eval done. Starting Durr-Hoyer..."
mkdir -p logs/quantum_retrieval_dh_classical
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_agvpreid_quantum.py \
    --config_file configs/vit_clipreid_agvpreid.yml \
    --checkpoint logs/agvpreid_classical_80ep/best_model.pth.tar \
    --retrieval durr_hoyer \
    --output_dir logs/quantum_retrieval_dh_classical \
    DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8 2>&1 | tee logs/quantum_retrieval_dh_classical/eval_log.txt

echo "[$(date)] All done."
