"""
train_qfeatext.py

Training script for the TF-CLIP Quantum Feature Extractor variant.

Architecture: Parallel integration pattern (survey §3.3.2).
Each classifier head concatenates the original CLIP feature with VQC-derived
quantum features before classifying:
    [CLIP_feat (768) || quantum_feat (2^n_qubits)] → nn.Linear → 625

This is different from train_qclassifier.py where the VQC REPLACES the
linear classifier (sequential pattern). Here the VQC AUGMENTS the classical
features — avoiding classical overshadowing.

Usage:
    python train_qfeatext.py \
        --config_file configs/vit_clipreid_qclassifier.yml \
        [--n_qubits 8] [--n_layers 2]

Differences from train_qclassifier.py:
- Imports make_model from quantum_models.make_model_qfeatext.
- No --classical_ablation or --encoding flags (not applicable).
- LR boost recalibrated for fused_dim = in_features + 2^n_qubits.
- Log label updated to show parallel architecture.
"""

import os
import os.path as osp
import sys
import datetime

import scipy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import argparse
from config import cfg

from utils.logger import setup_logger
import datasets.make_dataloader_clipreid as _dm_module
import torch.utils.data as _tud
from datasets.make_dataloader_clipreid import make_dataloader, train_collate_fn
from datasets.video_loader_xh import VideoDataset
from datasets.samplers import RandomIdentitySampler
from loss.make_loss import make_loss
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_train_stage2

# Quantum feature extractor model
from quantum_models.make_model_qfeatext import make_model


class _LimitedLoader:
    """Caps a DataLoader at max_batches iterations; preserves __len__ and .batch_size."""
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

    parser = argparse.ArgumentParser(description="TF-CLIP Quantum Feature Extractor Training")

    parser.add_argument(
        "--config_file",
        default="configs/vit_clipreid_qclassifier.yml",
        help="Path to YACS config file.",
        type=str,
    )
    parser.add_argument(
        "--n_qubits",
        default=8,
        type=int,
        help="Number of qubits in each VQC feature extractor head. (default: 8)",
    )
    parser.add_argument(
        "--n_layers",
        default=2,
        type=int,
        help="Number of variational entangler layers in each VQC. (default: 2)",
    )
    parser.add_argument(
        "opts",
        help="Override config options via command line (key value pairs).",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument(
        "--classical_ablation",
        action="store_true",
        default=False,
        help="Replace each QuantumAugmentedClassifier with a pure nn.Linear(in_features→num_classes). "
             "Proper linear probe baseline: no VQC, no bottleneck, no feature concat. "
             "Use this to isolate whether qfeatext gains come from quantum features "
             "or from the 768-dim classical bypass path.",
    )
    parser.add_argument(
        "--n_ids",
        default=None,
        type=int,
        help="Restrict training to the first N identity classes (expressibility diagnostic). "
             "Filters the stage-2 loader and sets num_classes=N. "
             "Use with --max_mem_batches 1 to suppress I2T loss.",
    )
    parser.add_argument(
        "--max_batches",
        default=None,
        type=int,
        help="Limit the stage-2 training loader to this many batches per epoch "
             "(quick smoke-test). Default: use the full dataset.",
    )
    parser.add_argument(
        "--max_mem_batches",
        default=None,
        type=int,
        help="Limit the stage-1 CLIP-memory loader to this many batches. "
             "When fewer than all identities are covered, the I2T loss is "
             "automatically skipped. Default: use the full stage-1 dataset.",
    )

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("TFCLIP", output_dir, if_train=True)
    logger.info("Saving model in the path: {}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    n_q_feat = 2 ** args.n_qubits
    if args.classical_ablation:
        logger.info(
            f"[QuantumFeatExt] mode=linear_probe (bypass_quantum=True), "
            f"pattern=none (nn.Linear(in_features→num_classes)), no VQC, no concat"
        )
    else:
        logger.info(
            f"[QuantumFeatExt] n_qubits={args.n_qubits}, n_layers={args.n_layers}, "
            f"pattern=parallel, fused_dim=768+{n_q_feat}={768+n_q_feat}"
        )

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            logger.info("\n" + cf.read())
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    class _NoPinDataLoader(_tud.DataLoader):
        def __init__(self, *args, **kwargs):
            kwargs['pin_memory'] = False
            super().__init__(*args, **kwargs)

    _dm_module.DataLoader = _NoPinDataLoader
    train_loader_stage2, train_loader_stage1, val_loader, \
        num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    _dm_module.DataLoader = _tud.DataLoader

    # ------------------------------------------------------------------ #
    # N-identity expressibility diagnostic  (--n_ids)
    # ------------------------------------------------------------------ #
    if args.n_ids is not None:
        raw_train = train_loader_stage2.dataset.dataset
        filtered = [(p, pid, cam, tid) for p, pid, cam, tid in raw_train if pid < args.n_ids]
        if not filtered:
            raise ValueError(
                f"--n_ids {args.n_ids}: no tracklets found for pids 0..{args.n_ids - 1}."
            )
        vds = train_loader_stage2.dataset
        filtered_dataset = VideoDataset(
            filtered, seq_len=vds.seq_len, sample=vds.sample, transform=vds.transform
        )
        filtered_loader = _tud.DataLoader(
            filtered_dataset,
            sampler=RandomIdentitySampler(
                filtered,
                batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                num_instances=cfg.DATALOADER.NUM_INSTANCE,
            ),
            batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=True,
            collate_fn=train_collate_fn,
        )
        logger.info(
            f"[n_ids] Expressibility diagnostic: restricting to {args.n_ids} identities "
            f"({len(filtered)}/{len(raw_train)} tracklets, "
            f"{len(filtered_loader)} batches/epoch). num_classes overridden to {args.n_ids}."
        )
        train_loader_stage2 = filtered_loader
        num_classes = args.n_ids

    if args.max_mem_batches is not None:
        logger.info(
            f"[quick-test] --max_mem_batches={args.max_mem_batches}: "
            f"stage1 {len(train_loader_stage1)} → {min(args.max_mem_batches, len(train_loader_stage1))} batches "
            f"(I2T loss auto-skipped if cluster_features are incomplete)."
        )
        train_loader_stage1 = _LimitedLoader(train_loader_stage1, args.max_mem_batches)

    if args.max_batches is not None:
        logger.info(
            f"[quick-test] --max_batches={args.max_batches}: "
            f"stage2 {len(train_loader_stage2)} → {min(args.max_batches, len(train_loader_stage2))} batches."
        )
        train_loader_stage2 = _LimitedLoader(train_loader_stage2, args.max_batches)

    # ------------------------------------------------------------------ #
    # Model  (quantum feature extractor variant)
    # ------------------------------------------------------------------ #
    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        bypass_quantum=args.classical_ablation,
    )

    # ------------------------------------------------------------------ #
    # Loss, optimiser, scheduler
    # ------------------------------------------------------------------ #
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(
        cfg, model, center_criterion
    )

    # ------------------------------------------------------------------ #
    # LR boost for quantum feature extractor heads.
    #
    # post_net: nn.Linear(fused_dim, 625).
    #   fused_dim = in_features + 2^n_qubits (e.g. 768+256=1024 for 8q).
    #   Target shift ≈ baseline Linear(768→625) with LARGE_FC_LR=10×.
    #   POST_NET_LR_FACTOR = max(5, int(768 * 10 / fused_dim))
    #     8q (fused=1024): floor(7680/1024)=7 → 7×
    #   This is lower than the sequential classifier (30× for 8q) because
    #   fused_dim is larger than n_measurements=256.
    #
    # quantum_extractor (pre_net + qlayer): 3× — same conservative factor
    #   as QuantumClassifier, prevents sigmoid embedding drift.
    # ------------------------------------------------------------------ #
    if args.classical_ablation:
        # Linear probe: post_net is nn.Linear(768→625) — same as baseline LARGE_FC_LR.
        # No VQC or concat; fused_dim = in_features = 768, so target factor = 10×.
        POST_NET_LR_FACTOR = 10
    else:
        fused_dim = 768 + (2 ** args.n_qubits)   # representative (768 = ViT-B-16 in_planes)
        POST_NET_LR_FACTOR = max(5, int(768 * 10 / fused_dim))
    VQC_LR_FACTOR      = 3
    n_post, n_vqc = 0, 0

    param_to_name = {id(p): n for n, p in model.named_parameters()}
    for pg in optimizer_2stage.param_groups:
        for p in pg["params"]:
            name = param_to_name.get(id(p), "")
            if name.startswith("classifier"):
                if name.endswith("post_net.weight"):
                    pg["lr"] *= POST_NET_LR_FACTOR
                    n_post += 1
                else:
                    pg["lr"] *= VQC_LR_FACTOR
                    n_vqc += 1

    mid_label = "n/a (linear probe)" if args.classical_ablation else "pre_net/qlayer"
    logger.info(
        f"[QuantumFeatExtLR] post_net {POST_NET_LR_FACTOR}× "
        f"(LR={cfg.SOLVER.STAGE2.BASE_LR * POST_NET_LR_FACTOR:.2e}, {n_post} params); "
        f"{mid_label} {VQC_LR_FACTOR}× "
        f"(LR={cfg.SOLVER.STAGE2.BASE_LR * VQC_LR_FACTOR:.2e}, {n_vqc} params)."
    )

    scheduler_2stage = WarmupMultiStepLR(
        optimizer_2stage,
        cfg.SOLVER.STAGE2.STEPS,
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD,
    )

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
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
        num_query,
        args.local_rank,
        num_classes,
    )

    # Always save the final model so standalone eval scripts have a checkpoint,
    # even if eval_period > max_epochs (which prevents in-loop checkpoint saving).
    from utils.iotools import save_slim_checkpoint as _save_slim
    _save_slim(model, fpath=os.path.join(cfg.OUTPUT_DIR, 'last_model.pth.tar'))
    logger.info(f"Final model (slim) saved to {cfg.OUTPUT_DIR}/last_model.pth.tar")
