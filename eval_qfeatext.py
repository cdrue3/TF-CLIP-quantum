"""
eval_qfeatext.py

Standalone evaluation script for the TF-CLIP Quantum Feature Extractor model.

Uses make_eval_all_dataloader (dense sampling, batch_size=1), which is the
correct format for do_inference_dense.

Usage:
    # Full eval from a saved checkpoint:
    python eval_qfeatext.py \\
        --config_file configs/vit_clipreid_qclassifier.yml \\
        --n_qubits 8 --n_layers 2 \\
        --checkpoint logs/agreid_qadapter/qfeatext_vqc_80ep/last_model.pth.tar

    # Classical ablation:
    python eval_qfeatext.py \\
        --config_file configs/vit_clipreid_qclassifier.yml \\
        --n_qubits 8 --n_layers 2 --classical_ablation \\
        --checkpoint logs/agreid_qadapter/qfeatext_classical_80ep/last_model.pth.tar
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
from quantum_models.make_model_qfeatext import make_model


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


def _pin_quantum_to_cpu(model):
    """Re-pin all PennyLane TorchLayer weights to CPU after any model.to(device) call."""
    for name, module in model.named_modules():
        if hasattr(module, 'qlayer'):
            module.qlayer.to(device=torch.device("cpu"), dtype=torch.float32)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="TF-CLIP Quantum Feature Extractor — Standalone Eval")

    parser.add_argument(
        "--config_file",
        default="configs/vit_clipreid_qclassifier.yml",
        type=str,
    )
    parser.add_argument("--n_qubits", default=8, type=int,
                        help="Number of qubits. Must match training. (default: 8)")
    parser.add_argument("--n_layers", default=2, type=int,
                        help="VQC variational layers. Must match training. (default: 2)")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="Path to saved model state dict (.pth.tar).")
    parser.add_argument("--classical_ablation", action="store_true", default=False,
                        help="Evaluate classical ablation (bypass_quantum=True).")
    parser.add_argument("--max_eval_batches", default=None, type=int,
                        help="Limit val_loader to N tracklets for smoke testing.")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Override config options (KEY VALUE pairs).")
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.merge_from_list(["DATALOADER.NUM_WORKERS", "0"])
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("TFCLIP", output_dir, if_train=False)
    logger.info(f"[eval_qfeatext] n_qubits={args.n_qubits}, n_layers={args.n_layers}, "
                f"classical_ablation={args.classical_ablation}")
    logger.info(f"[eval_qfeatext] checkpoint={args.checkpoint or 'None (random weights — smoke test)'}")

    val_loader, num_query, num_classes, camera_num, view_num = make_eval_all_dataloader(cfg)
    logger.info(
        f"[eval_qfeatext] {len(val_loader)} tracklets total, "
        f"num_query={num_query}, num_classes={num_classes}, cameras={camera_num}"
    )

    if args.max_eval_batches is not None:
        actual = min(args.max_eval_batches, len(val_loader))
        num_query = min(num_query, max(1, actual // 2))
        logger.info(
            f"[smoke] --max_eval_batches={args.max_eval_batches}: limiting to {actual} tracklets. "
            f"Metrics NOT statistically valid."
        )
        val_loader = _LimitedLoader(val_loader, actual)

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        bypass_quantum=args.classical_ablation,
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
        logger.info("No checkpoint provided — using random weights. Metrics will be random.")

    _pin_quantum_to_cpu(model)

    logger.info("Starting evaluation (do_inference_dense) ...")
    r1, r5 = do_inference_dense(cfg, model, val_loader, num_query)
    logger.info(f"[eval_qfeatext DONE] Rank-1: {r1:.1%}  Rank-5: {r5:.1%}")
