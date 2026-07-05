"""
train_qframe.py

Training script for the TF-CLIP Quantum Frame Attention variant.

Architecture: VQC generates T soft attention weights for temporal aggregation.
Replaces plain .mean(1) with attention-weighted sum over T frames:

    [B, T, 768] → QuantumFrameAttention → [B, 768] → BN → nn.Linear → cls_score
    Other 3 heads: unchanged from original TF-CLIP.

This is different from:
  train_qadapter.py    — VQC ADAPTS features (additive residual)
  train_qtemporal.py   — VQC REPLACES temporal aggregation AND classifier head
  train_qframe.py      — VQC WEIGHTS T frames (soft attention, T scalars only)

Usage:
    python train_qframe.py \\
        --config_file configs/vit_clipreid_qclassifier.yml \\
        [--n_qubits 8] [--n_layers 2]
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

# Quantum frame attention model
from quantum_models.make_model_qframe import make_model


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

    parser = argparse.ArgumentParser(description="TF-CLIP Quantum Frame Attention Training")

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
        help="Number of qubits in the shared VQC adapter. (default: 8)",
    )
    parser.add_argument(
        "--n_layers",
        default=2,
        type=int,
        help="Number of variational entangler layers in the VQC. (default: 2)",
    )
    parser.add_argument(
        "--classical_ablation",
        action="store_true",
        default=False,
        help="Replace the VQC in the adapter with a classical Linear(n_qubits→2^n_qubits)+ReLU. "
             "Ablation test: checks whether the residual architecture alone (not the quantum "
             "circuit) explains the adapter's performance gains. Default: False (use VQC).",
    )
    parser.add_argument(
        "opts",
        help="Override config options via command line (key value pairs).",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", default=0, type=int)
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
    parser.add_argument(
        "--max_eval_batches",
        default=None,
        type=int,
        help="Limit the eval (val_loader) to this many tracklets per eval pass. "
             "For smoke testing only — Rank-1/mAP are not statistically valid. "
             "Default: use the full query+gallery set.",
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        default=None,
        type=str,
        help="Path to a pretrained TF-CLIP checkpoint (.pth.tar) to warm-start from. "
             "Keys matching the current model are loaded; new keys (e.g. quantum_adapter.*) "
             "keep their initialised values. Useful for fine-tuning a domain-adapted backbone.",
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
    mode_str = "classical_ablation (bypass_quantum=True)" if args.classical_ablation else "quantum VQC"
    logger.info(
        f"[QuantumFrameAttn] n_qubits={args.n_qubits}, n_layers={args.n_layers}, "
        f"mode={mode_str}, pattern=frame attention (replaces temporal mean), "
        f"768→{args.n_qubits}→{n_q_feat}→T_weights, "
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
    # Eval loader — use make_eval_all_dataloader (dense, batch_size=1)
    # The training val_loader from make_dataloader uses sample='rrs_test'
    # with batch_size=30, which is incompatible with do_inference_dense
    # (designed for dense batch_size=1 producing 6D tensors).
    # ------------------------------------------------------------------ #
    val_loader_eval, num_query_eval, _, _, _ = make_eval_all_dataloader(cfg)
    logger.info(
        f"[eval-loader] {len(val_loader_eval)} tracklets total, "
        f"num_query={num_query_eval} (dense/batch_size=1 format)."
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
    # Model  (quantum adapter variant)
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
    # Optional warm-start from a pretrained checkpoint
    # ------------------------------------------------------------------ #
    if args.pretrained_checkpoint is not None:
        logger.info(f"[pretrained] Loading from: {args.pretrained_checkpoint}")
        ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu")
        model_sd = model.state_dict()
        loaded, skipped = [], []
        for k, v in ckpt.items():
            k_clean = k.replace("module.", "")
            if k_clean in model_sd and model_sd[k_clean].shape == v.shape:
                model_sd[k_clean].copy_(v)
                loaded.append(k_clean)
            else:
                skipped.append(k_clean)
        model.load_state_dict(model_sd)
        logger.info(
            f"[pretrained] Loaded {len(loaded)} keys; "
            f"skipped {len(skipped)} (shape mismatch or new keys). "
            f"quantum_adapter.* keys initialised fresh."
        )

    # ------------------------------------------------------------------ #
    # Loss, optimiser, scheduler
    # ------------------------------------------------------------------ #
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(
        cfg, model, center_criterion
    )

    # ------------------------------------------------------------------ #
    # LR boost for quantum adapter components.
    #
    # quantum_adapter.pre_net / qlayer: 3× — conservative, prevents sigmoid drift.
    # quantum_adapter.upscale: standard LR (1×) — takes gradient from classical
    #   heads which have full gradient flow; upscale learns quickly.
    # classifier heads (nn.Linear, 768→625): 10× — same as baseline LARGE_FC_LR.
    #   Named 'classifier' so the pattern match applies.
    #
    # Note: `classifier` params don't have 'post_net.weight' suffix — they ARE
    # the linear weights directly (e.g. 'classifier2.weight'). Use a different
    # naming convention: boost all 'classifier' params by 10×, adapter params by 3×.
    # ------------------------------------------------------------------ #
    ADAPTER_LR_FACTOR     = 1    # pre_net + qlayer: conservative
    CLASSIFIER_LR_FACTOR = 1   # nn.Linear heads: same as baseline LARGE_FC_LR
    n_adapter, n_cls = 0, 0

    param_to_name = {id(p): n for n, p in model.named_parameters()}
    for pg in optimizer_2stage.param_groups:
        for p in pg["params"]:
            name = param_to_name.get(id(p), "")
            if name.startswith("frame_attn"):
                if "weight_net" not in name:   # pre_net + qlayer get boost; weight_net standard
                    pg["lr"] *= ADAPTER_LR_FACTOR
                    n_adapter += 1
            elif name.startswith("classifier"):
                pg["lr"] *= CLASSIFIER_LR_FACTOR
                n_cls += 1

    logger.info(
        f"[QuantumFrameAttnLR] attn pre_net/qlayer {ADAPTER_LR_FACTOR}× "
        f"(LR={cfg.SOLVER.STAGE2.BASE_LR * ADAPTER_LR_FACTOR:.2e}, {n_adapter} params); "
        f"classifiers {CLASSIFIER_LR_FACTOR}× "
        f"(LR={cfg.SOLVER.STAGE2.BASE_LR * CLASSIFIER_LR_FACTOR:.2e}, {n_cls} params)."
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
    # val_loader_eval uses dense/batch_size=1 format (correct for do_inference_dense).
    # num_query_eval is adjusted when --max_eval_batches limits the loader.
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

    # Always save the final model so standalone eval scripts have a checkpoint,
    # even if eval_period > max_epochs (which prevents in-loop checkpoint saving).
    from utils.iotools import save_slim_checkpoint as _save_slim
    _save_slim(model, fpath=os.path.join(cfg.OUTPUT_DIR, 'last_model.pth.tar'))
    logger.info(f"Final model (slim) saved to {cfg.OUTPUT_DIR}/last_model.pth.tar")
