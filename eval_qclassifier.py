"""
eval_qclassifier.py

Standalone evaluation script for the TF-CLIP Quantum Classifier and Quantum
Feature Extractor (qfeatext) models.

Note on eval mode: during eval, the model does NOT call classifier heads.
Features are extracted from the ViT backbone identically to the classical model.
The quantum circuits (QuantumClassifier / QuantumAugmentedClassifier) are only
active during training (forward returns classification logits). In eval mode,
the model returns concatenated backbone features [img_feature | img_feature_proj |
cls_f_tp] — fully classical, quantum circuits not executed.

This means eval_qclassifier.py is useful for:
  - Verifying the eval pipeline runs without errors
  - Getting Rank-1/mAP from a trained qclassifier checkpoint

Usage:
    # Smoke test (50 tracklets, random weights):
    python eval_qclassifier.py \\
        --config_file configs/vit_clipreid_qclassifier.yml \\
        --n_qubits 8 --n_layers 2 \\
        --max_eval_batches 50

    # Full eval from checkpoint:
    python eval_qclassifier.py \\
        --config_file configs/vit_clipreid_qclassifier.yml \\
        --n_qubits 8 --n_layers 2 \\
        --checkpoint logs/mars_vit_clip_reid_qclassifier/last_model.pth.tar

    # qfeatext (parallel pattern):
    python eval_qclassifier.py \\
        --config_file configs/vit_clipreid_qclassifier.yml \\
        --n_qubits 8 --n_layers 2 --model_variant qfeatext \\
        --checkpoint logs/mars_vit_clip_reid_qclassifier/last_model.pth.tar
"""

import os
import sys
import random
import numpy as np
import torch
import argparse

from config import cfg
from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_eval_all_dataloader
from processor.processor_clipreid_stage2 import do_inference_rrs as do_inference_dense


class _LimitedLoader:
    """Caps a DataLoader at max_batches; preserves __len__ and attribute access."""
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
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="TF-CLIP Quantum Classifier — Standalone Eval")

    parser.add_argument(
        "--config_file",
        default="configs/vit_clipreid_qclassifier.yml",
        type=str,
        help="Path to YACS config file.",
    )
    parser.add_argument(
        "--n_qubits",
        default=8,
        type=int,
        help="Number of qubits. Must match the training configuration. (default: 8)",
    )
    parser.add_argument(
        "--n_layers",
        default=2,
        type=int,
        help="Number of VQC variational layers. Must match training. (default: 2)",
    )
    parser.add_argument(
        "--model_variant",
        default="qclassifier",
        choices=["qclassifier", "qfeatext"],
        type=str,
        help="Which quantum model variant to load. "
             "'qclassifier': VQC heads (train_qclassifier.py). "
             "'qfeatext': parallel concat pattern (train_qfeatext.py). "
             "Default: qclassifier.",
    )
    parser.add_argument(
        "--encoding",
        default="angle",
        choices=["angle", "dense_angle", "iqp"],
        type=str,
        help="Quantum encoding used during training. Must match. Default: angle.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Path to a saved model state dict (.pth.tar). "
             "If omitted, random (untrained) weights are used — smoke test only.",
    )
    parser.add_argument(
        "--max_eval_batches",
        default=None,
        type=int,
        help="Limit val_loader to this many tracklets for smoke testing. "
             "Rank-1/mAP not meaningful when limited. Default: full dataset.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Override config options via command line (KEY VALUE pairs).",
    )
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    # batch_size=1 eval: multiprocessing overhead exceeds prefetch benefit (especially WSL2).
    cfg.merge_from_list(["DATALOADER.NUM_WORKERS", "0"])
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("TFCLIP", output_dir, if_train=False)
    logger.info(
        f"[eval_qclassifier] variant={args.model_variant}, "
        f"n_qubits={args.n_qubits}, n_layers={args.n_layers}, encoding={args.encoding}"
    )
    logger.info(f"[eval_qclassifier] checkpoint={args.checkpoint or 'None (random weights — smoke test)'}")

    # ------------------------------------------------------------------
    # Val loader — dense format, batch_size=1 (required by do_inference_dense)
    # ------------------------------------------------------------------
    val_loader, num_query, num_classes, camera_num, view_num = make_eval_all_dataloader(cfg)
    logger.info(
        f"[eval_qclassifier] {len(val_loader)} tracklets total, "
        f"num_query={num_query}, num_classes={num_classes}, cameras={camera_num}"
    )

    if args.max_eval_batches is not None:
        actual = min(args.max_eval_batches, len(val_loader))
        num_query = min(num_query, max(1, actual // 2))
        logger.info(
            f"[smoke] --max_eval_batches={args.max_eval_batches}: limiting to {actual} tracklets, "
            f"effective num_query={num_query}. Metrics NOT statistically valid."
        )
        val_loader = _LimitedLoader(val_loader, actual)

    # ------------------------------------------------------------------
    # Model — choose variant
    # ------------------------------------------------------------------
    if args.model_variant == "qfeatext":
        from quantum_models.make_model_qfeatext import make_model
        model = make_model(
            cfg,
            num_class=num_classes,
            camera_num=camera_num,
            view_num=view_num,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
        )
    else:  # qclassifier (default)
        from quantum_models.make_model_qclassifier import make_model
        model = make_model(
            cfg,
            num_class=num_classes,
            camera_num=camera_num,
            view_num=view_num,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            encoding=args.encoding,
        )

    if args.checkpoint:
        logger.info(f"Loading checkpoint from: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        logger.info("Checkpoint loaded successfully.")
    else:
        logger.info("No checkpoint — using random (untrained) weights. Metrics will be random.")

    # Note: quantum circuits are NOT executed during eval (model.training=False
    # causes the forward pass to return backbone features directly, skipping
    # classifier heads). No CPU pinning is required here.

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    logger.info("Starting evaluation (do_inference_dense) ...")
    r1, r5 = do_inference_dense(cfg, model, val_loader, num_query)
    logger.info(f"[eval_qclassifier DONE] Rank-1: {r1:.1%}  Rank-5: {r5:.1%}")
