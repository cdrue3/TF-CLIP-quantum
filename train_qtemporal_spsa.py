"""
train_qtemporal.py

Training script for the TF-CLIP Temporal Quantum Aggregation (TQA) variant.

Architecture: QuantumTemporalAgg replaces mean-pool on the primary temporal path.
    [B, T, 768] → TQA (VQC sequential upload + skip) → [B, 768] → BN → nn.Linear

    img_feature_proj (512) and TMD output retain plain mean-pool.
    All 4 classifier heads: classical nn.Linear.

This is different from:
  train_qclassifier.py  — VQC REPLACES classifier heads (sequential)
  train_qfeatext.py     — VQC AUGMENTS classifier heads via concat (parallel)
  train_qadapter.py     — VQC ADAPTS post-pool features (residual, shared)
  train_qtemporal.py    — VQC AGGREGATES temporal frames (replaces mean-pool)

Usage:
    python train_qtemporal.py \\
        --config_file configs/vit_clipreid_qtemporal.yml \\
        [--n_qubits 8] [--n_layers 2] [--classical_ablation]
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
from datasets.make_dataloader_clipreid import make_dataloader, make_eval_all_dataloader, train_collate_fn
from datasets.video_loader_xh import VideoDataset
from datasets.samplers import RandomIdentitySampler
from loss.make_loss import make_loss
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_train_stage2

# Temporal quantum aggregation model
from quantum_models.make_model_qtemporal import make_model
from quantum_models.optimisation.spsa_optimizer import make_hybrid_optimizer


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

    parser = argparse.ArgumentParser(description="TF-CLIP SPSA Hybrid Optimizer Training (QTemporal + SPSA for circuit params)")

    parser.add_argument(
        "--config_file",
        default="configs/vit_clipreid_agvpreid.yml",
        help="Path to YACS config file.",
        type=str,
    )
    parser.add_argument(
        "--n_qubits",
        default=8,
        type=int,
        help="Number of qubits in the TQA VQC. (default: 8)",
    )
    parser.add_argument(
        "--n_layers",
        default=2,
        type=int,
        help="Number of variational entangler layers per frame. (default: 2)",
    )
    parser.add_argument(
        "--classical_ablation",
        action="store_true",
        default=False,
        help="Replace the VQC in TQA with plain mean-pool (bypass_quantum=True). "
             "Ablation: tests mean-pool baseline with same architecture wrapper.",
    )
    parser.add_argument(
        "opts",
        help="Override config options via command line (key value pairs).",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--fast_schedule",
        action="store_true",
        default=False,
        help="Scale LR decay steps proportionally to MAX_EPOCHS (for short runs). "
             "Steps [30,50,70]/80ep become proportional fractions of total epochs.",
    )
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument(
        "--n_ids",
        default=None,
        type=int,
        help="Restrict training to the first N identity classes (expressibility diagnostic).",
    )
    parser.add_argument(
        "--max_batches",
        default=None,
        type=int,
        help="Limit stage-2 training loader to this many batches per epoch (quick test).",
    )
    parser.add_argument(
        "--max_mem_batches",
        default=None,
        type=int,
        help="Limit stage-1 CLIP-memory loader to this many batches.",
    )
    parser.add_argument(
        "--max_eval_batches",
        default=None,
        type=int,
        help="Limit eval loader to this many tracklets (smoke test only).",
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
    mode_str = "classical_ablation (bypass_quantum=True, plain mean-pool)" if args.classical_ablation else "quantum VQC"
    logger.info(
        f"[TQA] n_qubits={args.n_qubits}, n_layers={args.n_layers}, "
        f"seq_len={cfg.INPUT.SEQ_LEN}, mode={mode_str}, "
        f"pattern=temporal re-uploading (replaces mean-pool), "
        f"tqa: 768→{args.n_qubits}→{n_q_feat}→768 (skip + mean-pool), "
        f"all 4 classifier heads: classical nn.Linear"
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
            f"[n_ids] Restricting to {args.n_ids} identities "
            f"({len(filtered)}/{len(raw_train)} tracklets). num_classes → {args.n_ids}."
        )
        train_loader_stage2 = filtered_loader
        num_classes = args.n_ids

    if args.max_mem_batches is not None:
        logger.info(
            f"[quick-test] --max_mem_batches={args.max_mem_batches}: "
            f"stage1 → {min(args.max_mem_batches, len(train_loader_stage1))} batches "
            f"(I2T loss auto-skipped if cluster_features incomplete)."
        )
        train_loader_stage1 = _LimitedLoader(train_loader_stage1, args.max_mem_batches)

    if args.max_batches is not None:
        logger.info(
            f"[quick-test] --max_batches={args.max_batches}: "
            f"stage2 → {min(args.max_batches, len(train_loader_stage2))} batches."
        )
        train_loader_stage2 = _LimitedLoader(train_loader_stage2, args.max_batches)

    # ------------------------------------------------------------------ #
    # Eval loader
    # ------------------------------------------------------------------ #
    val_loader_eval, num_query_eval, _, _, _ = make_eval_all_dataloader(cfg)
    logger.info(
        f"[eval-loader] {len(val_loader_eval)} tracklets total, "
        f"num_query={num_query_eval}."
    )
    if args.max_eval_batches is not None:
        actual = min(args.max_eval_batches, len(val_loader_eval))
        num_query_eval = min(num_query_eval, max(1, actual // 2))
        logger.info(
            f"[quick-eval] --max_eval_batches={args.max_eval_batches}: "
            f"limiting to {actual} tracklets, effective num_query={num_query_eval} "
            f"(metrics not statistically valid — smoke test only)."
        )
        val_loader_eval = _LimitedLoader(val_loader_eval, actual)

    # ------------------------------------------------------------------ #
    # Model  (TQA variant)
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

    # Hybrid optimizer: Adam for classical params, SPSA for VQC circuit weights
    optimizer_adam, optimizer_spsa = make_hybrid_optimizer(model, cfg)
    optimizer_2stage = optimizer_adam
    _, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    # Attach SPSA to model so processor can call it (non-standard — see processor notes)
    model._spsa_optimizer = optimizer_spsa

    # ------------------------------------------------------------------ #
    # LR boost for TQA components.
    #
    # tqa.pre_net / qlayer: 3× — conservative, avoids sigmoid drift.
    # tqa.upscale: standard LR (1×) — takes gradient from classical heads.
    # classifier heads (nn.Linear): 10× — same as baseline LARGE_FC_LR.
    # ------------------------------------------------------------------ #
    TQA_LR_FACTOR        = 3
    CLASSIFIER_LR_FACTOR = 10
    n_tqa, n_cls = 0, 0

    param_to_name = {id(p): n for n, p in model.named_parameters()}
    for pg in optimizer_2stage.param_groups:
        for p in pg["params"]:
            name = param_to_name.get(id(p), "")
            if name.startswith("tqa"):
                if "upscale" not in name:
                    pg["lr"] *= TQA_LR_FACTOR
                    n_tqa += 1
            elif name.startswith("classifier"):
                pg["lr"] *= CLASSIFIER_LR_FACTOR
                n_cls += 1

    logger.info(
        f"[TQA_LR] tqa pre_net/qlayer {TQA_LR_FACTOR}× "
        f"(LR={cfg.SOLVER.STAGE2.BASE_LR * TQA_LR_FACTOR:.2e}, {n_tqa} params); "
        f"classifiers {CLASSIFIER_LR_FACTOR}× "
        f"(LR={cfg.SOLVER.STAGE2.BASE_LR * CLASSIFIER_LR_FACTOR:.2e}, {n_cls} params)."
    )

    sched_steps = list(cfg.SOLVER.STAGE2.STEPS)
    if args.fast_schedule:
        total = cfg.SOLVER.STAGE2.MAX_EPOCHS
        sched_steps = [max(1, int(total * 0.75)), max(2, int(total * 0.90))]
        logger.info(f"[fast_schedule] MAX_EPOCHS={total}, scaled steps={sched_steps} "
                    f"(proportional to [30,50,70]/80)")

    scheduler_2stage = WarmupMultiStepLR(
        optimizer_2stage,
        sched_steps,
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
        val_loader_eval,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query_eval,
        args.local_rank,
        num_classes,
    )

    from utils.iotools import save_slim_checkpoint as _save_slim
    _save_slim(model, fpath=os.path.join(cfg.OUTPUT_DIR, 'last_model.pth.tar'))
    logger.info(f"Final model (slim) saved to {cfg.OUTPUT_DIR}/last_model.pth.tar")
