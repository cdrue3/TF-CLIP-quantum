"""
train_q_triplet_loss.py

Optimisation-stage quantum experiment: replaces TripletLoss with QuantumTripletLoss.
A trainable quantum kernel defines the training-time distance metric.
Zero quantum at inference — classical L2 / cosine retrieval unchanged.

Usage:
    python train_q_triplet_loss.py \
        --config_file configs/vit_clipreid_agvpreid.yml \
        --n_qubits 6 --n_layers 1 \
        DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8 \
        SOLVER.STAGE2.MAX_EPOCHS 20 SOLVER.STAGE2.EVAL_PERIOD 999 \
        SOLVER.STAGE2.CHECKPOINT_PERIOD 10 \
        OUTPUT_DIR logs/agvpreid_q_triplet_loss_20ep
"""

import os
import numpy as np
import random
import torch

import argparse
from config import cfg

from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from loss.make_loss import make_loss_q_triplet
from solver.make_optimizer_prompt import make_optimizer_2stage
from solver.lr_scheduler import WarmupMultiStepLR
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_train_stage2


class _LimitedLoader:
    def __init__(self, loader, max_batches):
        self._loader = loader
        self._max = max_batches
    def __len__(self):
        return min(self._max, len(self._loader))
    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if i >= self._max:
                break
            yield batch
    def __getattr__(self, name):
        return getattr(self._loader, name)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quantum Triplet Loss Training")
    parser.add_argument("--config_file", default="configs/vit_clipreid_agvpreid.yml", type=str)
    parser.add_argument("--n_qubits",  default=6, type=int,
                        help="Qubits in quantum kernel VQC.")
    parser.add_argument("--n_layers",  default=1, type=int,
                        help="StronglyEntanglingLayers in quantum kernel.")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument("--max_mem_batches", default=None, type=int)
    parser.add_argument("--fast_schedule", action="store_true", default=False)
    parser.add_argument("--no_amp", action="store_true", default=False)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("TFCLIP", output_dir, if_train=True)
    logger.info(f"[QuantumTripletLoss] n_qubits={args.n_qubits}, n_layers={args.n_layers}")
    logger.info("Running with config:\n{}".format(cfg))

    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    if args.max_mem_batches is not None:
        train_loader_stage1 = _LimitedLoader(train_loader_stage1, args.max_mem_batches)
    if args.max_batches is not None:
        train_loader_stage2 = _LimitedLoader(train_loader_stage2, args.max_batches)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # Build quantum triplet loss and register on model so its params are
    # automatically picked up by make_optimizer_2stage → model.parameters()
    loss_func, center_criterion, q_triplet = make_loss_q_triplet(
        cfg, num_classes, feat_dim=None,  # auto-detected on first forward pass
        n_qubits=args.n_qubits, n_layers=args.n_layers,
    )
    model.q_triplet = q_triplet   # registers as nn.Module submodule
    logger.info(f"[QuantumTripletLoss] registered on model — "
                f"params: pre_net {q_triplet.pre_net.weight.shape}, "
                f"q_weights {q_triplet.q_weights.shape}")

    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)

    sched_steps = list(cfg.SOLVER.STAGE2.STEPS)
    if args.fast_schedule:
        total = cfg.SOLVER.STAGE2.MAX_EPOCHS
        sched_steps = [max(1, int(total * 0.75)), max(2, int(total * 0.90))]
        logger.info(f"[fast_schedule] steps={sched_steps}")

    scheduler_2stage = WarmupMultiStepLR(
        optimizer_2stage, sched_steps, cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR, cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD,
    )

    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage1,
        train_loader_stage2,
        val_loader,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query, args.local_rank,
        num_classes,
        use_amp=not args.no_amp,
    )
