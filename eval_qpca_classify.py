"""
eval_qpca_classify.py — Eval for the QPCA + VQC Classifier model.
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
from quantum_models.make_model_qpca_classify import make_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="QPCA Classifier — Eval")
    parser.add_argument("--config_file", default="configs/vit_clipreid_agvpreid.yml", type=str)
    parser.add_argument("--n_qubits_pca", default=10, type=int)
    parser.add_argument("--n_qubits_out", default=3, type=int)
    parser.add_argument("--n_qubits_cls", default=8, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--bypass_quantum", action="store_true", default=False)
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
    np.random.seed(cfg.SOLVER.SEED)
    random.seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("TFCLIP", output_dir, if_train=False)
    logger.info(
        f"[eval_qpca_classify] QPCA: {args.n_qubits_pca}q → {2**args.n_qubits_out} components, "
        f"VQC: {args.n_qubits_cls}q, layers={args.n_layers}"
    )
    logger.info(f"checkpoint={args.checkpoint or 'None'}")

    val_loader, num_query, num_classes, camera_num, view_num = make_eval_all_dataloader(cfg)

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        n_qubits_pca=args.n_qubits_pca,
        n_qubits_out=args.n_qubits_out,
        n_qubits_cls=args.n_qubits_cls,
        n_layers=args.n_layers,
        bypass_quantum=args.bypass_quantum,
    )

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}")
        logger.info("Checkpoint loaded.")

    r1, r5 = do_inference_dense(cfg, model, val_loader, num_query)
    logger.info(f"[eval_qpca_classify DONE] Rank-1: {r1:.1%}  Rank-5: {r5:.1%}")
