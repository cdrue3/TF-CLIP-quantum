"""
train_qtl.py

Proper Quantum Transfer Learning (QTL) training script for TF-CLIP.

What makes this proper QTL (vs train_qclassifier.py):
    1. FROZEN backbone: a pretrained classical TF-CLIP checkpoint is loaded
       and all backbone parameters are frozen. Only the single QTLClassifier
       head trains — the quantum circuit IS the fine-tuner.
    2. Single head: one QTLClassifier (not 4 redundant ones).
    3. Simple loss: CrossEntropy on the QTL head only. Triplet and CLIP i2t
       losses are omitted because they require backbone gradients.
    4. Optional PCA init (--use_pca): after extracting all training features
       from the frozen backbone, PCA is computed and the top n_qubits principal
       components initialise dress_layer.weight. This gives the VQC angles
       that maximise variance in the input distribution from the first step,
       rather than a random linear projection.

Usage:
    # Quick 15-epoch test:
    conda run -n tfclip python train_qtl.py \\
        --config_file configs/vit_clipreid_qclassifier.yml \\
        --pretrained_checkpoint logs/mars_vit_clip_reid_qclassifier/last_model.pth.tar \\
        --n_qubits 8 --n_layers 2 \\
        --max_mem_batches 5 --max_batches 500 \\
        SOLVER.STAGE2.MAX_EPOCHS 15 SOLVER.STAGE2.LOG_PERIOD 100 \\
        OUTPUT_DIR logs/mars_vit_clip_reid_qtl

    # With PCA dress_layer initialisation:
    conda run -n tfclip python train_qtl.py \\
        --config_file configs/vit_clipreid_qclassifier.yml \\
        --pretrained_checkpoint logs/mars_vit_clip_reid_qclassifier/last_model.pth.tar \\
        --n_qubits 8 --n_layers 2 --use_pca \\
        --max_mem_batches 5 --max_batches 500 \\
        SOLVER.STAGE2.MAX_EPOCHS 15 SOLVER.STAGE2.LOG_PERIOD 100 \\
        OUTPUT_DIR logs/mars_vit_clip_reid_qtl
"""

import os
import sys
import time
import datetime
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
from tqdm import tqdm
from config import cfg
from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from processor.processor_clipreid_stage2 import do_inference_dense
from datasets.make_dataloader_clipreid import make_eval_all_dataloader
from solver.lr_scheduler import WarmupMultiStepLR

from quantum_models.make_model_qtl import make_model


# ============================================================================
# Helpers
# ============================================================================

class _LimitedLoader:
    def __init__(self, loader, max_batches):
        self._loader  = loader
        self._max     = max_batches
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
    """Re-pin PennyLane TorchLayer weights to CPU after any model.to(device) call."""
    for name, module in model.named_modules():
        if hasattr(module, 'qlayer'):
            module.qlayer.to(device=torch.device("cpu"), dtype=torch.float32)


# ============================================================================
# PCA initialisation
# ============================================================================

def _init_dress_pca(model, train_loader, logger, max_batches=None):
    """
    Compute PCA on frozen backbone features extracted from the training set,
    then initialise dress_layer.weight with the top n_qubits principal components.

    This ensures the dress layer projects CLIP features onto the directions of
    maximum variance — the most informative n_qubits dimensions — rather than
    a random linear projection.

    Args:
        model        : QTL model (backbone frozen).
        train_loader : Stage-2 training loader (video batches).
        logger       : Logger instance.
        max_batches  : If set, limit to this many batches for PCA fitting.
    """
    n_qubits = model.qtl_classifier.n_qubits
    logger.info(f"[PCA init] Extracting training features for PCA (n_qubits={n_qubits}) ...")

    model.eval()
    feats = []
    limit = max_batches or len(train_loader)

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= limit:
                break
            img, target, cam_label, view_label = batch
            img = img.cuda()
            cam_label  = cam_label.cuda()
            view_label = view_label.cuda()
            # Run backbone only (training=False path skips QTL head)
            feat, img_feature, img_feature_proj, cls_f_tp = model._extract_features(
                img, cam_label, view_label
            )
            feats.append(feat.cpu().float())

    feats = torch.cat(feats, dim=0)   # [N, in_features]
    logger.info(f"[PCA init] Collected {feats.shape[0]} feature vectors — fitting PCA ...")

    # Centre features
    mean = feats.mean(dim=0, keepdim=True)
    feats_c = feats - mean

    # SVD-based PCA (torch.linalg.svd is stable on large matrices)
    try:
        _, _, Vt = torch.linalg.svd(feats_c, full_matrices=False)
        components = Vt[:n_qubits]   # [n_qubits, in_features] — top principal directions
    except Exception as e:
        logger.warning(f"[PCA init] SVD failed ({e}), falling back to numpy.")
        U, S, Vt_np = np.linalg.svd(feats_c.numpy(), full_matrices=False)
        components = torch.from_numpy(Vt_np[:n_qubits]).float()

    # Set dress_layer.weight to PCA components.
    # weight shape is [out_features, in_features] = [n_qubits, in_features].
    with torch.no_grad():
        model.qtl_classifier.dress_layer.weight.copy_(components)
        nn.init.zeros_(model.qtl_classifier.dress_layer.bias)

    logger.info(f"[PCA init] dress_layer.weight initialised with top-{n_qubits} PCA components.")
    model.train()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="TF-CLIP Quantum Transfer Learning Training")

    parser.add_argument(
        "--config_file",
        default="configs/vit_clipreid_qclassifier.yml",
        type=str,
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        default=None,
        type=str,
        help="Path to a pretrained classical TF-CLIP checkpoint (.pth.tar). "
             "REQUIRED for proper QTL — the backbone must be pre-trained before "
             "the quantum head is attached and frozen.",
    )
    parser.add_argument("--n_qubits", default=8, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument(
        "--use_pca",
        action="store_true",
        default=False,
        help="Initialise dress_layer.weight with PCA components computed from "
             "frozen backbone features of the training set. "
             "If False (default), Kaiming random initialisation is used.",
    )
    parser.add_argument(
        "--max_batches",
        default=None,
        type=int,
        help="Limit stage-2 training loader to this many batches per epoch.",
    )
    parser.add_argument(
        "--max_mem_batches",
        default=None,
        type=int,
        help="Limit stage-1 CLIP-memory loader (not used for QTL training loss, "
             "but still passed to make_dataloader for compatibility).",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Override config options (KEY VALUE pairs).",
    )
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

    logger = setup_logger("TFCLIP", output_dir, if_train=True)
    pca_str = "PCA" if args.use_pca else "Kaiming"
    logger.info(
        f"[train_qtl] n_qubits={args.n_qubits}, n_layers={args.n_layers}, "
        f"dress_init={pca_str}, "
        f"checkpoint={args.pretrained_checkpoint or 'NONE (not proper QTL)'}"
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, \
        camera_num, view_num = make_dataloader(cfg)

    if args.max_mem_batches is not None:
        train_loader_stage1 = _LimitedLoader(train_loader_stage1, args.max_mem_batches)
    if args.max_batches is not None:
        train_loader_stage2 = _LimitedLoader(train_loader_stage2, args.max_batches)

    logger.info(
        f"[train_qtl] {len(train_loader_stage2)} batches/epoch, "
        f"{num_classes} classes, {camera_num} cameras"
    )

    # ------------------------------------------------------------------
    # Model — frozen backbone + single QTL head
    # ------------------------------------------------------------------
    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        pretrained_checkpoint=args.pretrained_checkpoint,
    )
    model.cuda()
    _pin_quantum_to_cpu(model)

    # ------------------------------------------------------------------
    # Optional PCA initialisation of dress_layer
    # ------------------------------------------------------------------
    if args.use_pca:
        _init_dress_pca(
            model, train_loader_stage2, logger,
            max_batches=args.max_mem_batches,  # reuse mem_batches limit for PCA fitting
        )
        _pin_quantum_to_cpu(model)

    # ------------------------------------------------------------------
    # Optimizer — only QTL head parameters
    # ------------------------------------------------------------------
    # LR schedule: dress_layer gets 3× boost (small module, needs faster learning);
    # qlayer and output_layer use base LR.
    base_lr    = cfg.SOLVER.STAGE2.BASE_LR
    qtl_params = []
    for name, param in model.qtl_classifier.named_parameters():
        lr = base_lr * 3 if "dress_layer" in name else base_lr
        qtl_params.append({"params": [param], "lr": lr, "name": name})

    optimizer = torch.optim.Adam(qtl_params, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)

    scheduler = WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STAGE2.STEPS,
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD,
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    max_epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD

    logger.info(f"[train_qtl] Starting QTL training for {max_epochs} epochs ...")
    model.train()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        running_acc  = 0.0
        n_batches    = 0

        pbar = tqdm(train_loader_stage2, desc=f"Epoch {epoch}/{max_epochs}", dynamic_ncols=True)
        for i, batch in enumerate(pbar):
            img, target, cam_label, view_label = batch
            img        = img.cuda()
            target     = target.cuda()
            cam_label  = cam_label.cuda()
            view_label = view_label.cuda()

            optimizer.zero_grad()

            # Backbone: frozen — no_grad applied inside model.forward (training=True)
            cls_score, feat = model(
                img, cam_label=cam_label, view_label=view_label
            )

            loss = loss_fn(cls_score, target)
            loss.backward()

            # Gradient clip to stabilise VQC training
            nn.utils.clip_grad_norm_(model.qtl_classifier.parameters(), max_norm=1.0)

            optimizer.step()
            _pin_quantum_to_cpu(model)

            # Accuracy
            with torch.no_grad():
                acc = (cls_score.argmax(dim=1) == target).float().mean().item()

            running_loss += loss.item()
            running_acc  += acc
            n_batches    += 1

            pbar.set_postfix(loss=f"{running_loss/n_batches:.3f}", acc=f"{running_acc/n_batches:.3f}")

            if (i + 1) % log_period == 0:
                avg_loss = running_loss / n_batches
                avg_acc  = running_acc  / n_batches
                lr_now   = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch[{epoch}] Iteration[{i+1}/{len(train_loader_stage2)}] "
                    f"Loss: {avg_loss:.3f}, Acc_id: {avg_acc:.3f}, "
                    f"Base Lr: {lr_now:.2e}"
                )

        scheduler.step()

        elapsed = time.time() - epoch_start
        avg_loss = running_loss / max(n_batches, 1)
        avg_acc  = running_acc  / max(n_batches, 1)
        logger.info(
            f"Epoch {epoch}/{max_epochs} done. "
            f"Loss: {avg_loss:.3f}, Acc_id: {avg_acc:.3f}, "
            f"Time: {elapsed:.1f}s ({elapsed/max(n_batches,1):.3f}s/batch)"
        )

        # Save checkpoint every epoch
        ckpt_path = os.path.join(output_dir, f"qtl_ep{epoch}.pth.tar")
        torch.save(model.state_dict(), ckpt_path)

    # Save final
    torch.save(model.state_dict(), os.path.join(output_dir, "qtl_last.pth.tar"))
    logger.info(f"[train_qtl] Training complete. Checkpoint saved to {output_dir}/qtl_last.pth.tar")
