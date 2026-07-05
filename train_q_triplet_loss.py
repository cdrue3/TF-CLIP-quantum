"""
train_q_triplet_loss.py

Classical TF-CLIP model with QuantumTripletLoss replacing the standard
TripletLoss at training time. The model architecture is unchanged —
the VQC only lives inside the loss function. At eval time, standard
L2 retrieval is used (no quantum involvement).

No LR boost. q_triplet is attached as model.q_triplet so make_optimizer_2stage
picks up its params at the same base LR as the backbone.

Usage:
    python train_q_triplet_loss.py \\
        --config_file configs/vit_clipreid_agvpreid.yml \\
        --n_qubits 6 --n_layers 1 \\
        SOLVER.STAGE2.MAX_EPOCHS 80 SOLVER.STAGE2.EVAL_PERIOD 999 \\
        SOLVER.STAGE2.CHECKPOINT_PERIOD 10 \\
        DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8 \\
        OUTPUT_DIR logs/agvpreid_q_triplet_loss_80ep
"""

import os
import random
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import cfg
from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from loss.make_loss import make_loss_q_triplet
from solver.make_optimizer_prompt import make_optimizer_2stage
from solver.lr_scheduler import WarmupMultiStepLR
from processor.processor_clipreid_stage2 import do_train_stage2


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QuantumTripletLoss Training')
    parser.add_argument('--config_file', default='configs/vit_clipreid_agvpreid.yml')
    parser.add_argument('--n_qubits',    default=6,  type=int)
    parser.add_argument('--n_layers',    default=1,  type=int)
    parser.add_argument('--encoding',    default='angle', choices=['angle', 'amplitude'])
    parser.add_argument('--fast_schedule', action='store_true', default=False)
    parser.add_argument('--local_rank',  default=0,  type=int)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('TFCLIP', output_dir, if_train=True)
    logger.info(f"Saving model in the path: {output_dir}")
    logger.info(args)
    logger.info(f"QuantumTripletLoss: n_qubits={args.n_qubits}, n_layers={args.n_layers}, "
                f"encoding={args.encoding} (no LR boost — q_triplet at backbone BASE_LR)")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader_stage2, train_loader_stage1, val_loader, num_query, \
        num_classes, camera_num, view_num = make_dataloader(cfg)

    # ── Model (classical — no quantum in architecture) ────────────────────────
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num)

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_func, center_criterion, q_triplet = make_loss_q_triplet(
        cfg, num_classes=num_classes,
        feat_dim=None,              # lazy init on first forward
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        encoding=args.encoding,
    )

    # Attach to model so make_optimizer_2stage picks up q_triplet params
    # at base LR — no boost, no separate param group needed
    model.q_triplet = q_triplet

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer, optimizer_center = make_optimizer_2stage(cfg, model, center_criterion)
    n_q_params = sum(p.numel() for p in q_triplet.parameters())
    logger.info(f"QuantumTripletLoss params: {n_q_params:,} "
                f"(pre_net deferred until first batch)")

    # ── LR scheduler ──────────────────────────────────────────────────────────
    sched_steps = list(cfg.SOLVER.STAGE2.STEPS)
    if args.fast_schedule:
        total = cfg.SOLVER.STAGE2.MAX_EPOCHS
        sched_steps = [max(1, int(total * 0.75)), max(2, int(total * 0.90))]
        logger.info(f"[fast_schedule] MAX_EPOCHS={total}, scaled steps={sched_steps}")

    scheduler = WarmupMultiStepLR(
        optimizer, sched_steps,
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage1,
        train_loader_stage2,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query,
        args.local_rank,
        num_classes,
        use_amp=True,
    )
