"""
train_qtemporal_ber.py

Training script for TF-CLIP with Stochastic Quantum Perturbation + Born Entropy Regularization.

Based on the noise-induced regularization literature (Kuzmin 2025, arXiv:2410.19921):
  - SQP: Gaussian noise injected into VQC weights each forward pass, decaying over epochs.
    Acts as quantum weight perturbation — approximates Bayesian posterior over circuit params.
  - BER: Born entropy maximization loss term (-λ * H(probs)) prevents the quantum circuit
    from collapsing to a low-entropy state at the ep30 LR drop, which is the primary
    cause of R1 collapse in previous experiments.

Key arguments:
  --entropy_reg   (float): λ for entropy regularization. Default 0.02.
  --noise_sigma   (float): Initial Gaussian noise std on circuit weights. Default 0.15.
  --noise_epochs  (int):   Epochs over which noise decays to zero (linear decay). Default 20.
"""

import os
import os.path as osp
import sys
import datetime
import time
import logging
import collections

import scipy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.utils.data as _tud
import argparse
from tqdm import tqdm
from torch.cuda import amp

from config import cfg
from utils.logger import setup_logger
import datasets.make_dataloader_clipreid as _dm_module
from datasets.make_dataloader_clipreid import make_dataloader, make_eval_all_dataloader, train_collate_fn
from datasets.video_loader_xh import VideoDataset
from datasets.samplers import RandomIdentitySampler
from loss.make_loss import make_loss
from loss.softmax_loss import CrossEntropyLabelSmooth
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.lr_scheduler import WarmupMultiStepLR
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_inference_rrs
from utils.iotools import save_slim_checkpoint
from utils.meter import AverageMeter

from quantum_models.feature_extraction.make_model_qtemporal_sqp import make_model as make_model_sqp
from quantum_models.feature_extraction.make_model_qtemporal_ham import make_model as make_model_ham
from quantum_models.feature_extraction.make_model_qtemporal_parallel import make_model as make_model_parallel


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
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def do_train_ber(
    cfg, model, train_loader_stage1, train_loader_stage2,
    optimizer_2stage, optimizer_center_2stage, scheduler_2stage,
    loss_fn, center_criterion, num_classes,
    entropy_reg: float, noise_sigma: float, noise_epochs: int,
    logger, val_loader=None, num_query=0, use_amp: bool = True,
):
    """Custom training loop with SQP noise schedule and BER entropy regularization."""
    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD

    model.to(device)

    loss_meter    = AverageMeter()
    acc_meter     = AverageMeter()
    acc_meter_id1 = AverageMeter()
    acc_meter_id2 = AverageMeter()
    scaler        = amp.GradScaler(enabled=use_amp)
    xent_frame    = CrossEntropyLabelSmooth(num_classes=num_classes)

    # ── CLIP Memory (from cache) ──────────────────────────────────────────
    _cache_path = os.path.join(cfg.DATASETS.ROOT_DIR, 'clip_memory_cache.pt')
    if os.path.exists(_cache_path):
        logger.info(f"Loading CLIP-Memory cache from {_cache_path}")
        cluster_features = torch.load(_cache_path, map_location='cuda').detach()
    else:
        logger.info("Generating CLIP-Memory (no cache found)…")
        image_features, labels = [], []
        with torch.no_grad():
            for img, vid, _, _ in tqdm(train_loader_stage1, desc="CLIP-Memory"):
                img = img.to(device)
                if len(img.size()) == 6:
                    b, n, s, c, h, w = img.size()
                    img = img.view(b * n, s, c, h, w)
                    feats = []
                    with amp.autocast(enabled=use_amp):
                        for ci in range(0, n, 16):
                            feats.append(model(img[ci:ci+16], get_image=True).cpu())
                    feat = torch.cat(feats).mean(0, keepdim=True)
                    for v, f in zip(vid, feat):
                        labels.append(v); image_features.append(f)
                else:
                    with amp.autocast(enabled=use_amp):
                        feat = model(img, get_image=True)
                    for v, f in zip(vid, feat):
                        labels.append(v); image_features.append(f.cpu())
        labels_t = torch.stack(labels).cuda()
        feats_t  = torch.stack(image_features).cuda()
        centers  = collections.defaultdict(list)
        for i, lbl in enumerate(labels_t.cpu().numpy()):
            centers[lbl].append(feats_t[i])
        cluster_features = torch.stack(
            [torch.stack(centers[k]).mean(0) for k in sorted(centers)]
        ).detach()
        torch.save(cluster_features.cpu(), _cache_path)
        logger.info(f"Saved CLIP-Memory to {_cache_path}")

    logger.info(
        f"[SQP+BER] entropy_reg={entropy_reg}, noise_sigma={noise_sigma}, "
        f"noise_epochs={noise_epochs}"
    )

    for epoch in range(1, epochs + 1):
        # ── SQP noise schedule: linear decay from noise_sigma to 0 ──────────
        noise_scale = noise_sigma * max(0.0, 1.0 - (epoch - 1) / max(noise_epochs, 1))
        if hasattr(model, 'tqa'):
            model.tqa._noise_scale = noise_scale

        loss_meter.reset()
        acc_meter.reset()
        acc_meter_id1.reset()
        acc_meter_id2.reset()
        model.train()

        pbar = tqdm(
            enumerate(train_loader_stage2),
            total=len(train_loader_stage2),
            desc=f"Epoch {epoch}/{epochs} σ={noise_scale:.3f}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
        for n_iter, (img, vid, target_cam, target_view) in pbar:
            optimizer_2stage.zero_grad()
            optimizer_center_2stage.zero_grad()

            img    = img.to(device)
            target = vid.to(device)
            target_cam  = target_cam.to(device) if cfg.MODEL.SIE_CAMERA else None
            target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None

            with amp.autocast(enabled=use_amp):
                B, T, C, H, W = img.shape
                score, feat, logits1 = model(
                    x=img, cam_label=target_cam, view_label=target_view,
                    text_features2=cluster_features
                )
                score1 = score[0:3]
                i2t_arg = logits1 if logits1.size(1) >= score[0].size(1) else None

                if (n_iter + 1) % log_period == 0:
                    loss1 = loss_fn(score1, feat, target, target_cam, i2t_arg, isprint=True)
                else:
                    loss1 = loss_fn(score1, feat, target, target_cam, i2t_arg)

                targetX = target.unsqueeze(1).expand(B, T).contiguous().view(B * T)
                loss_frame = xent_frame(score[3], targetX)
                loss = loss1 + loss_frame / T

            # ── BER: Born entropy regularization ────────────────────────────
            if entropy_reg > 0 and hasattr(model, 'tqa') and model.tqa._last_probs is not None:
                probs = model.tqa._last_probs.float().to(device)
                # Shannon entropy H = -Σ p log p; maximise → add negative to loss
                H = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                loss = loss - entropy_reg * H
            # ────────────────────────────────────────────────────────────────

            scaler.scale(loss).backward()

            # gradient diagnostic on first iter
            if n_iter == 0 and epoch == 1:
                logger.info("[GradCheck] First backward pass norms:")
                for name, param in model.named_parameters():
                    if 'tqa' in name and param.grad is not None:
                        logger.info(
                            f"  {name}: grad_norm={param.grad.norm().item():.4e}"
                        )

            scaler.step(optimizer_2stage)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center_2stage)
                scaler.update()

            acc1    = (logits1.max(1)[1] == target).float().mean()
            acc_id1 = (score[0].max(1)[1] == target).float().mean()
            acc_id2 = (score[3].max(1)[1] == targetX).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc1, 1)
            acc_meter_id1.update(acc_id1, 1)
            acc_meter_id2.update(acc_id2, 1)

            pbar.set_postfix(
                loss=f"{loss_meter.avg:.3f}",
                acc_clip=f"{acc_meter.avg:.3f}",
                acc_id=f"{acc_meter_id1.avg:.3f}",
                lr=f"{scheduler_2stage.get_lr()[0]:.1e}",
            )

            if (n_iter + 1) % log_period == 0:
                logger.info(
                    f"Epoch[{epoch}] Iteration[{n_iter+1}/{len(train_loader_stage2)}] "
                    f"Loss: {loss_meter.avg:.3f}, Acc_clip: {acc_meter.avg:.3f}, "
                    f"Acc_id1: {acc_meter_id1.avg:.3f}, Acc_id2: {acc_meter_id2.avg:.3f}, "
                    f"Base Lr: {scheduler_2stage.get_lr()[0]:.2e}, noise_σ: {noise_scale:.3f}"
                )

        if hasattr(scheduler_2stage, 'set_metric'):
            scheduler_2stage.set_metric(loss_meter.avg)
        scheduler_2stage.step()

        if eval_period > 0 and epoch % eval_period == 0 and val_loader is not None:
            model.eval()
            do_inference_rrs(cfg, model, val_loader, num_query)
            model.train()

        if epoch % checkpoint_period == 0:
            save_slim_checkpoint(
                model,
                fpath=os.path.join(cfg.OUTPUT_DIR, f'checkpoint_ep{epoch:02d}.pth.tar')
            )
            logger.info(f"Checkpoint saved: ep{epoch:02d}")

    logger.info("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TF-CLIP SQP+BER Quantum Training")
    parser.add_argument("--config_file", default="configs/vit_clipreid_agvpreid.yml", type=str)
    parser.add_argument("--n_qubits",    default=8,    type=int)
    parser.add_argument("--n_layers",    default=2,    type=int)
    parser.add_argument("--entropy_reg", default=0.02, type=float,
                        help="BER: λ for -λ*H(probs) entropy term (default 0.02)")
    parser.add_argument("--noise_sigma", default=0.15, type=float,
                        help="SQP: initial noise std on VQC weights (default 0.15)")
    parser.add_argument("--noise_epochs", default=20, type=int,
                        help="SQP: epochs over which noise decays to 0 (default 20)")
    parser.add_argument("--dense_encoding", action="store_true", default=False,
                        help="Use dense angle encoding (RY+RZ, 2 features/qubit)")
    parser.add_argument("--hamiltonian", action="store_true", default=False,
                        help="Use Hamiltonian encoding (all 768 features, no compression)")
    parser.add_argument("--parallel", action="store_true", default=False,
                        help="Parallel quantum-classical architecture (PQCNN-style)")
    parser.add_argument("--fusion_mode", default="concat", choices=["concat", "gated"],
                        help="Fusion mode for parallel arch (default: concat)")
    parser.add_argument("--classical_ablation", action="store_true", default=False)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("TFCLIP", output_dir, if_train=True)
    # Wire transreid.test logger (used by do_inference_rrs) to our handlers
    import logging as _logging
    _tl = _logging.getLogger("transreid.test")
    for _h in logger.handlers:
        _tl.addHandler(_h)
    _tl.setLevel(_logging.DEBUG)
    logger.info(args)
    logger.info(
        f"[SQP+BER] n_qubits={args.n_qubits}, n_layers={args.n_layers}, "
        f"entropy_reg={args.entropy_reg}, noise_sigma={args.noise_sigma}, "
        f"noise_epochs={args.noise_epochs}"
    )

    # ── Data ────────────────────────────────────────────────────────────────
    class _NoPinDataLoader(_tud.DataLoader):
        def __init__(self, *args, **kwargs):
            kwargs['pin_memory'] = False
            super().__init__(*args, **kwargs)

    _dm_module.DataLoader = _NoPinDataLoader
    train_loader_stage2, train_loader_stage1, val_loader, \
        num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    _dm_module.DataLoader = _tud.DataLoader

    # ── Model ───────────────────────────────────────────────────────────────
    if args.parallel:
        model = make_model_parallel(
            cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num,
            n_qubits=args.n_qubits, n_layers=args.n_layers,
            bypass_quantum=args.classical_ablation,
            dense_encoding=args.dense_encoding,
            hamiltonian=args.hamiltonian,
            fusion_mode=args.fusion_mode,
        )
    elif args.hamiltonian:
        model = make_model_ham(
            cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num,
            n_qubits=args.n_qubits, n_layers=args.n_layers,
            bypass_quantum=args.classical_ablation,
        )
    else:
        model = make_model_sqp(
            cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num,
            n_qubits=args.n_qubits, n_layers=args.n_layers,
            bypass_quantum=args.classical_ablation,
            dense_encoding=args.dense_encoding,
        )

    # ── Loss / Optimizer ────────────────────────────────────────────────────
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(
        cfg, model, center_criterion
    )

    # LR boost for TQA components (same as train_qtemporal_deep.py)
    TQA_LR_FACTOR        = 1
    CLASSIFIER_LR_FACTOR = 1
    param_to_name = {id(p): n for n, p in model.named_parameters()}
    for pg in optimizer_2stage.param_groups:
        for p in pg["params"]:
            name = param_to_name.get(id(p), "")
            if name.startswith("tqa") and "upscale" not in name:
                pg["lr"] *= TQA_LR_FACTOR
            elif name.startswith("classifier"):
                pg["lr"] *= CLASSIFIER_LR_FACTOR

    scheduler_2stage = WarmupMultiStepLR(
        optimizer_2stage,
        list(cfg.SOLVER.STAGE2.STEPS),
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    do_train_ber(
        cfg=cfg,
        model=model,
        train_loader_stage1=train_loader_stage1,
        train_loader_stage2=train_loader_stage2,
        optimizer_2stage=optimizer_2stage,
        optimizer_center_2stage=optimizer_center_2stage,
        scheduler_2stage=scheduler_2stage,
        loss_fn=loss_func,
        center_criterion=center_criterion,
        num_classes=num_classes,
        entropy_reg=args.entropy_reg,
        noise_sigma=args.noise_sigma,
        noise_epochs=args.noise_epochs,
        logger=logger,
        val_loader=val_loader,
        num_query=num_query,
    )

    save_slim_checkpoint(model, fpath=os.path.join(cfg.OUTPUT_DIR, 'last_model.pth.tar'))
    logger.info(f"Final model saved to {cfg.OUTPUT_DIR}/last_model.pth.tar")
