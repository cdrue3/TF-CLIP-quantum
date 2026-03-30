"""
eval_agvpreid.py

Standalone evaluation script for AG-VPReID using the standard TF-CLIP model.
Evaluates Case 1 (aerial→ground) and/or Case 2 (ground→aerial) from a single
trained checkpoint.

Usage:
    # Both cases:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval_agvpreid.py \\
        --checkpoint logs/agvpreid_classical_baseline/best_model.pth.tar

    # Single case:
    python eval_agvpreid.py \\
        --checkpoint logs/agvpreid_classical_baseline/best_model.pth.tar \\
        --case 1
"""

import os
import random
import numpy as np
import torch
import argparse

from config import cfg
from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_eval_all_dataloader
from processor.processor_clipreid_stage2 import do_inference_rrs as do_inference_dense
from model.make_model_clipreid import make_model


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def run_eval(cfg, args, case, logger):
    label = f"Case {case} ({'aerial→ground' if case == 1 else 'ground→aerial'})"
    logger.info(f"\n{'='*60}\nEvaluating {label}\n{'='*60}")

    val_loader, num_query, num_classes, camera_num, view_num = \
        make_eval_all_dataloader(cfg, case=case)
    logger.info(f"{len(val_loader)} tracklets total, num_query={num_query}")

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
        logger.info("Checkpoint loaded.")
    else:
        logger.info("No checkpoint — random weights (smoke test only).")

    r1, r5 = do_inference_dense(cfg, model, val_loader, num_query)
    logger.info(f"[{label}] Rank-1: {r1:.1%}  Rank-5: {r5:.1%}")
    return r1, r5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AG-VPReID Eval — Case 1 and/or Case 2")
    parser.add_argument("--config_file", default="configs/vit_clipreid_agvpreid.yml", type=str)
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="Path to saved checkpoint (.pth.tar). Omit for smoke test.")
    parser.add_argument("--case", default=0, type=int, choices=[0, 1, 2],
                        help="Which case to eval: 1=aerial→ground, 2=ground→aerial, 0=both (default).")
    parser.add_argument("--max_eval_batches", default=None, type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.merge_from_list(["DATALOADER.NUM_WORKERS", "4"])
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("TFCLIP", output_dir, if_train=False)
    logger.info(f"checkpoint={args.checkpoint or 'None (smoke test)'}")

    cases = [1, 2] if args.case == 0 else [args.case]
    results = {}
    for c in cases:
        results[c] = run_eval(cfg, args, c, logger)

    logger.info("\n=== SUMMARY ===")
    for c, (r1, r5) in results.items():
        label = 'aerial→ground' if c == 1 else 'ground→aerial'
        logger.info(f"Case {c} ({label}): Rank-1 {r1:.1%}  Rank-5 {r5:.1%}")
