"""
eval_qpreprocess.py — Standalone evaluation for the Quantum Channel Preprocessing model.
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
from quantum_models.preprocessing.make_model_qpreprocess import make_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="TF-CLIP Quantum Channel Preprocessing — Standalone Eval")
    parser.add_argument("--config_file", default="configs/vit_clipreid_agvpreid.yml", type=str)
    parser.add_argument("--n_qubits", default=4, type=int)
    parser.add_argument("--n_layers", default=1, type=int)
    parser.add_argument("--preprocess_mode", default="qpca", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--bypass_quantum", action="store_true", default=False)
    parser.add_argument("--max_eval_batches", default=None, type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.merge_from_list(["DATALOADER.NUM_WORKERS", "0"])
    cfg.freeze()

    torch.manual_seed(cfg.SOLVER.SEED)
    torch.cuda.manual_seed(cfg.SOLVER.SEED)
    np.random.seed(cfg.SOLVER.SEED)
    random.seed(cfg.SOLVER.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("TFCLIP", output_dir, if_train=False)
    logger.info(f"[eval_qpreprocess] mode={args.preprocess_mode}, n_qubits={args.n_qubits}, n_layers={args.n_layers}")
    logger.info(f"[eval_qpreprocess] checkpoint={args.checkpoint or 'None (random weights)'}")

    val_loader, num_query, num_classes, camera_num, view_num = make_eval_all_dataloader(cfg)

    if args.max_eval_batches is not None:
        from train_qpreprocess import _LimitedLoader
        actual = min(args.max_eval_batches, len(val_loader))
        num_query = min(num_query, max(1, actual // 2))
        val_loader = _LimitedLoader(val_loader, actual)

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        preprocess_mode=args.preprocess_mode,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        bypass_quantum=args.bypass_quantum,
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
        logger.info("No checkpoint — random weights.")

    logger.info("Starting evaluation ...")
    r1, r5 = do_inference_dense(cfg, model, val_loader, num_query)
    logger.info(f"[eval_qpreprocess DONE] Rank-1: {r1:.1%}  Rank-5: {r5:.1%}")
