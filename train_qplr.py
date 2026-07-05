"""
train_qplr.py

Quantum Probabilistic Label Refining (QPLR) — training script.

QPLR adds a VQC-based KL loss on top of the standard CE+triplet+I2T losses.
The VQC processes top-K class logits and refines the soft label distribution,
capturing inter-class quantum correlations.

Options:
  --pretrain_checkpoint  warm-start backbone from an existing checkpoint
                         (use with 80ep classical checkpoint for option 1)
  --epochs               total training epochs (option 2: try 60-80ep)
  --kl_weight            blend weight for QPLR KL loss (default 0.5)
  --top_k_classes        number of top logits fed to VQC (default 32)

Usage (option 1 — warm-start from 80ep backbone):
  python train_qplr.py \\
      --config_file configs/vit_clipreid_agvpreid.yml \\
      --pretrain_checkpoint logs/agvpreid_classical_80ep/best_model.pth.tar \\
      SOLVER.STAGE2.MAX_EPOCHS 40 SOLVER.STAGE2.EVAL_PERIOD 5 \\
      SOLVER.STAGE2.CHECKPOINT_PERIOD 5 \\
      DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8 \\
      OUTPUT_DIR logs/agvpreid_qplr/80ep_warmstart_40ep

Usage (option 2 — longer run from scratch):
  python train_qplr.py \\
      --config_file configs/vit_clipreid_agvpreid.yml \\
      SOLVER.STAGE2.MAX_EPOCHS 80 SOLVER.STAGE2.EVAL_PERIOD 5 \\
      SOLVER.STAGE2.CHECKPOINT_PERIOD 5 \\
      DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8 \\
      OUTPUT_DIR logs/agvpreid_qplr/80ep
"""

import os
import sys
import random
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import cfg
from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from loss.make_loss import make_loss_qplr
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
    parser = argparse.ArgumentParser(description='QPLR Training')
    parser.add_argument('--config_file',           default='configs/vit_clipreid_agvpreid.yml')
    parser.add_argument('--pretrain_checkpoint',   default=None,
                        help='Warm-start backbone from this checkpoint before training')
    parser.add_argument('--n_qubits',              default=8,   type=int)
    parser.add_argument('--n_layers',              default=2,   type=int)
    parser.add_argument('--top_k_classes',         default=32,  type=int)
    parser.add_argument('--kl_weight',             default=0.5, type=float)
    parser.add_argument('--fast_schedule',         action='store_true', default=False)
    parser.add_argument('--local_rank',            default=0,   type=int)
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
    logger.info(f"QPLR: n_qubits={args.n_qubits}, n_layers={args.n_layers}, "
                f"top_k_classes={args.top_k_classes}, kl_weight={args.kl_weight}")
    if args.pretrain_checkpoint:
        logger.info(f"Warm-start: {args.pretrain_checkpoint}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader_stage2, train_loader_stage1, val_loader, num_query, \
        num_classes, camera_num, view_num = make_dataloader(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num)

    if args.pretrain_checkpoint:
        state = torch.load(args.pretrain_checkpoint, map_location='cpu')
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded pretrain checkpoint "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # ── Loss (QPLR) ───────────────────────────────────────────────────────────
    loss_func, center_criterion, q_refiner = make_loss_qplr(
        cfg,
        num_classes=num_classes,
        top_k=args.top_k_classes,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        kl_weight=args.kl_weight,
    )

    # ── Optimizer — add q_refiner params at backbone LR ──────────────────────
    optimizer, optimizer_center = make_optimizer_2stage(cfg, model, center_criterion)
    for p in q_refiner.parameters():
        optimizer.add_param_group({
            'params': [p],
            'lr': cfg.SOLVER.STAGE2.BASE_LR,
            'weight_decay': cfg.SOLVER.STAGE2.WEIGHT_DECAY,
        })

    # ── LR scheduler ─────────────────────────────────────────────────────────
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
