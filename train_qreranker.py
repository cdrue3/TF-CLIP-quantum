"""
train_qreranker.py

Post-hoc quantum re-ranker: trains a VQC pairwise similarity model on frozen
features from a classical checkpoint, then evals reranked retrieval.

Pipeline:
  1. Load classical checkpoint (frozen — no backbone training)
  2. Extract features for train set (GPU, one pass, cached to /tmp)
  3. Train VQC binary classifier (same-ID / different-ID pairs) on CPU
  4. Extract query + gallery features
  5. Rerank top-K classical L2 candidates using VQC scores
  6. Report Rank-1 vs classical baseline

Usage:
  python train_qreranker.py \\
      --config_file configs/vit_clipreid_agvpreid.yml \\
      --checkpoint logs/agvpreid_classical_80ep/best_model.pth.tar \\
      DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8
"""

import os
import sys
import random
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import cfg
from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader, make_eval_all_dataloader
from model.make_model_clipreid import make_model
from quantum_models.postprocessing.quantum_reranker import QuantumPairwiseReranker
from quantum_models.postprocessing.classical_reranker import ClassicalPairwiseReranker


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, loader, device, desc=''):
    model.eval()
    feats, pids, camids = [], [], []
    for batch in loader:
        # Train loader: (img, pids, camids, viewids)
        # Val loader:   (img, pids, camids, camids_batch, viewids, img_paths)
        if len(batch) == 4:
            img, pid, camid, target_view = batch
            camids_batch = torch.as_tensor(camid)
        else:
            img, pid, camid, camids_batch, target_view, _ = batch
        img = img.to(device)
        cam = camids_batch.to(device) if cfg.MODEL.SIE_CAMERA else None
        view = torch.as_tensor(target_view).to(device) if cfg.MODEL.SIE_VIEW else None
        f = model(img, cam_label=cam, view_label=view)
        feats.append(f.cpu())
        pids.extend(torch.as_tensor(pid).view(-1).tolist())
        camids.extend(torch.as_tensor(camid).view(-1).tolist())
    feats = torch.cat(feats, 0)
    feats = F.normalize(feats, p=2, dim=1)
    return feats, pids, camids


# ─────────────────────────────────────────────────────────────────────────────
# Pair sampling
# ─────────────────────────────────────────────────────────────────────────────

def build_pairs(feats, pids, n_pairs_per_class=10, neg_ratio=1.0):
    """
    Sample positive (same-ID) and negative (different-ID) feature pairs.
    Returns: pair_feats_q [N, D], pair_feats_g [N, D], labels [N] (float 0/1)
    """
    pid_tensor = torch.tensor(pids)
    unique_ids = pid_tensor.unique().tolist()

    pos_q, pos_g, neg_q, neg_g = [], [], [], []

    for uid in unique_ids:
        idx = (pid_tensor == uid).nonzero(as_tuple=True)[0].tolist()
        if len(idx) < 2:
            continue
        pairs_this = min(n_pairs_per_class, len(idx) * (len(idx) - 1) // 2)
        for _ in range(pairs_this):
            i, j = random.sample(idx, 2)
            pos_q.append(feats[i])
            pos_g.append(feats[j])

    n_pos = len(pos_q)
    n_neg = int(n_pos * neg_ratio)
    all_idx = list(range(len(pids)))
    attempts = 0
    while len(neg_q) < n_neg and attempts < n_neg * 10:
        i, j = random.sample(all_idx, 2)
        if pids[i] != pids[j]:
            neg_q.append(feats[i])
            neg_g.append(feats[j])
        attempts += 1

    fq = torch.stack(pos_q + neg_q)
    fg = torch.stack(pos_g + neg_g)
    labels = torch.tensor([1.0] * len(pos_q) + [0.0] * len(neg_q))
    # Shuffle
    perm = torch.randperm(len(labels))
    return fq[perm], fg[perm], labels[perm]


# ─────────────────────────────────────────────────────────────────────────────
# Eval helpers
# ─────────────────────────────────────────────────────────────────────────────

def classical_rank1(query_feats, gallery_feats, query_pids, gallery_pids,
                    query_camids, gallery_camids):
    dist = torch.cdist(query_feats.float(), gallery_feats.float())
    indices = dist.argsort(dim=1)
    correct = 0
    for q_idx in range(len(query_pids)):
        qpid, qcam = query_pids[q_idx], query_camids[q_idx]
        for g_idx in indices[q_idx].tolist():
            gpid, gcam = gallery_pids[g_idx], gallery_camids[g_idx]
            if gpid == qpid and gcam == qcam:
                continue  # same camera — skip (junk)
            correct += (gpid == qpid)
            break
    return correct / len(query_pids)


@torch.no_grad()
def rerank_with_vqc(model, query_feats, gallery_feats, query_pids, gallery_pids,
                    query_camids, gallery_camids, top_k=20, batch_size=64,
                    alpha=0.3, logger=None):
    """
    Rerank top-K classical L2 candidates using learned VQC similarity.
    final_score = (1-alpha)*vqc_score + alpha*(1 - classical_rank/top_k)
    """
    model.eval()
    dist = torch.cdist(query_feats.float(), gallery_feats.float())
    classical_indices = dist.argsort(dim=1)  # [n_q, n_g]

    n_q = len(query_pids)
    reranked_indices = []

    for q_idx in range(n_q):
        topk_idx = classical_indices[q_idx, :top_k].tolist()
        fq_rep = query_feats[q_idx].unsqueeze(0).expand(len(topk_idx), -1)  # [K, D]
        fg_cands = gallery_feats[topk_idx]                                   # [K, D]

        # Score in batches
        scores = []
        for start in range(0, len(topk_idx), batch_size):
            bq = fq_rep[start:start+batch_size]
            bg = fg_cands[start:start+batch_size]
            s = model(bq, bg).cpu()
            scores.append(s)
        scores = torch.cat(scores)  # [K]

        # Blend: higher = better match
        classical_score = 1.0 - torch.arange(len(topk_idx)).float() / len(topk_idx)
        final_score = (1 - alpha) * scores + alpha * classical_score

        reranked_order = final_score.argsort(descending=True)
        reranked_topk = [topk_idx[i] for i in reranked_order.tolist()]
        # Append rest of gallery unchanged
        rest = [i for i in classical_indices[q_idx, top_k:].tolist()]
        reranked_indices.append(reranked_topk + rest)

    # Compute Rank-1
    correct = 0
    for q_idx in range(n_q):
        qpid, qcam = query_pids[q_idx], query_camids[q_idx]
        for g_idx in reranked_indices[q_idx]:
            gpid, gcam = gallery_pids[g_idx], gallery_camids[g_idx]
            if gpid == qpid and gcam == qcam:
                continue
            correct += (gpid == qpid)
            break

    rank1 = correct / n_q
    return rank1, reranked_indices


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-hoc Quantum Pairwise Reranker')
    parser.add_argument('--config_file', default='configs/vit_clipreid_agvpreid.yml')
    parser.add_argument('--checkpoint',  default='logs/agvpreid_classical_80ep/best_model.pth.tar')
    parser.add_argument('--n_qubits',    default=8,   type=int)
    parser.add_argument('--n_layers',    default=2,   type=int)
    parser.add_argument('--epochs',      default=60,  type=int)
    parser.add_argument('--lr',          default=1e-3, type=float)
    parser.add_argument('--batch_size',  default=64,  type=int)
    parser.add_argument('--top_k',       default=20,  type=int,
                        help='Number of top-K classical candidates to rerank')
    parser.add_argument('--alpha',       default=0.3, type=float,
                        help='Blend weight for classical rank vs VQC score (0=pure VQC)')
    parser.add_argument('--mode',        default='quantum', choices=['quantum', 'classical'],
                        help='quantum: VQC reranker | classical: MLP ablation (same architecture, no VQC)')
    parser.add_argument('--output_dir',  default='logs/quantum_reranker')
    parser.add_argument('--feat_cache',  default='/tmp/qreranker_train_feats.pt',
                        help='Cache path for training set features')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('qreranker', args.output_dir, if_train=True)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"n_qubits={args.n_qubits}, n_layers={args.n_layers}, "
                f"epochs={args.epochs}, top_k={args.top_k}, alpha={args.alpha}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # ── Load classical model (frozen) ────────────────────────────────────────
    train_loader, _, val_loader, num_query, num_classes, camera_num, view_num = \
        make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num)
    state = torch.load(args.checkpoint, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"Loaded checkpoint (missing={len(missing)}, unexpected={len(unexpected)})")
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Extract / load training features ─────────────────────────────────────
    if os.path.exists(args.feat_cache):
        logger.info(f"Loading cached training features from {args.feat_cache}")
        cache = torch.load(args.feat_cache)
        train_feats, train_pids, train_camids = cache['feats'], cache['pids'], cache['camids']
    else:
        logger.info("Extracting training set features...")
        train_feats, train_pids, train_camids = extract_features(model, train_loader, device, 'train')
        torch.save({'feats': train_feats, 'pids': train_pids, 'camids': train_camids},
                   args.feat_cache)
        logger.info(f"Cached to {args.feat_cache}")

    logger.info(f"Train features: {train_feats.shape}, {len(set(train_pids))} IDs")

    # ── Build reranker (quantum VQC or classical MLP ablation) ───────────────
    if args.mode == 'quantum':
        reranker = QuantumPairwiseReranker(
            in_features=train_feats.shape[1],
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
        )
    else:
        reranker = ClassicalPairwiseReranker(
            in_features=train_feats.shape[1],
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
        )
    n_params = sum(p.numel() for p in reranker.parameters() if p.requires_grad)
    logger.info(f"Reranker ({args.mode}): {n_params:,} trainable params")
    optimizer = torch.optim.Adam(reranker.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────────
    from tqdm import tqdm
    logger.info(f"Training VQC reranker for {args.epochs} epochs...")
    epoch_bar = tqdm(range(1, args.epochs + 1), desc='Training', unit='ep')
    for epoch in epoch_bar:
        reranker.train()
        fq, fg, labels = build_pairs(train_feats, train_pids,
                                     n_pairs_per_class=8, neg_ratio=1.0)
        dataset = TensorDataset(fq, fg, labels)
        loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

        total_loss, correct, total = 0.0, 0, 0
        for bq, bg, blabels in loader:
            optimizer.zero_grad()
            scores = reranker(bq, bg)
            loss = F.binary_cross_entropy(scores, blabels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(blabels)
            preds = (scores > 0.5).float()
            correct += (preds == blabels).sum().item()
            total += len(blabels)

        scheduler.step()
        avg_loss = total_loss / total
        acc = correct / total
        epoch_bar.set_postfix(loss=f'{avg_loss:.4f}', acc=f'{acc:.3f}',
                              lr=f'{scheduler.get_last_lr()[0]:.1e}')

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch[{epoch}/{args.epochs}] loss={avg_loss:.4f} acc={acc:.3f} "
                        f"lr={scheduler.get_last_lr()[0]:.1e}")
            ckpt_path = os.path.join(args.output_dir, f'reranker_ep{epoch:02d}.pt')
            torch.save(reranker.state_dict(), ckpt_path)

    # Save final weights
    vqc_path = os.path.join(args.output_dir, 'reranker.pt')
    torch.save(reranker.state_dict(), vqc_path)
    logger.info(f"Saved final reranker weights to {vqc_path}")

    # ── Eval ──────────────────────────────────────────────────────────────────
    logger.info("Extracting query + gallery features for eval...")
    from datasets.make_dataloader_clipreid import make_eval_all_dataloader
    eval_loader, num_q, *_ = make_eval_all_dataloader(cfg, case=1)
    eval_feats, eval_pids, eval_camids = extract_features(model, eval_loader, device, 'eval')

    q_feats   = eval_feats[:num_q]
    g_feats   = eval_feats[num_q:]
    q_pids    = eval_pids[:num_q]
    g_pids    = eval_pids[num_q:]
    q_camids  = eval_camids[:num_q]
    g_camids  = eval_camids[num_q:]

    logger.info(f"Query: {q_feats.shape[0]}, Gallery: {g_feats.shape[0]}")

    # Classical baseline
    classical_r1 = classical_rank1(q_feats, g_feats, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"Classical L2 Rank-1: {classical_r1*100:.1f}%")

    # Reranked (quantum or classical ablation)
    label = 'VQC' if args.mode == 'quantum' else 'MLP'
    logger.info(f"Reranking top-{args.top_k} with {label} (alpha={args.alpha})...")
    vqc_r1, _ = rerank_with_vqc(reranker, q_feats, g_feats, q_pids, g_pids,
                                 q_camids, g_camids,
                                 top_k=args.top_k, batch_size=args.batch_size,
                                 alpha=args.alpha, logger=logger)
    logger.info(f"{label} Reranked Rank-1:  {vqc_r1*100:.1f}%")
    logger.info(f"Delta vs classical:        {(vqc_r1 - classical_r1)*100:+.1f}pp")
