"""
train_classical_single_head.py

Classical ablation for QuantumTripletLoss — identical setup but uses standard
TripletLoss on only feat[0] (primary head) instead of all heads.

Isolates whether the Q-Triplet improvement comes from the VQC or from
the single-head change.
"""

import os
import random
import argparse

import numpy as np
import torch

from config import cfg
from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from loss.make_loss import make_loss_single_head
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
    parser = argparse.ArgumentParser(description='Classical Single-Head Triplet Ablation')
    parser.add_argument('--config_file', default='configs/vit_clipreid_agvpreid.yml')
    parser.add_argument('--fast_schedule', action='store_true', default=False)
    parser.add_argument('--local_rank', default=0, type=int)
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
    logger.info("Classical single-head ablation: TripletLoss on feat[0] only")

    train_loader_stage2, train_loader_stage1, val_loader, num_query, \
        num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num)

    loss_func, center_criterion = make_loss_single_head(cfg, num_classes)

    optimizer, optimizer_center = make_optimizer_2stage(cfg, model, center_criterion)

    sched_steps = list(cfg.SOLVER.STAGE2.STEPS)
    if args.fast_schedule:
        total = cfg.SOLVER.STAGE2.MAX_EPOCHS
        sched_steps = [max(1, int(total * 0.75)), max(2, int(total * 0.90))]

    scheduler = WarmupMultiStepLR(
        optimizer, sched_steps,
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD,
    )

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
