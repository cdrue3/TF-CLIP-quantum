"""
train_qkreranker.py

Quantum K-Reciprocal Re-Ranker: trains a VQC on k-NN distance patterns to
re-rank classical retrieval results. Also evaluates the classical k-reciprocal
baseline (Zhong et al. CVPR 2017) as the proper ablation.

Architecture:
    For each (query q, gallery candidate g) pair:
      N_q = top-k gallery neighbours of q (the shared "neighbourhood")
      v_q [k] = L2 distances from q  to each n ∈ N_q
      v_g [k] = L2 distances from g  to each n ∈ N_q
      cat(v_q, v_g) [2k] → VQC → match_score ∈ (0,1)

Classical k-reciprocal baseline: Zhong et al. Jaccard-distance re-ranking.
The quantum model is trained to predict the same structural similarity that
k-reciprocal encoding captures, but via a learned VQC.

Usage:
    python train_qkreranker.py \\
        --config_file configs/vit_clipreid_agvpreid.yml \\
        --checkpoint logs/agvpreid_classical_80ep/best_model.pth.tar \\
        --epochs 80 --k 20 \\
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
from utils.kreranker import krerank, rank1_from_distmat
from datasets.make_dataloader_clipreid import make_dataloader, make_eval_all_dataloader
from model.make_model_clipreid import make_model
from quantum_models.postprocessing.quantum_kreranker import QuantumKReciprocalReranker


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, loader, device, desc=''):
    model.eval()
    feats, pids, camids = [], [], []
    for batch in loader:
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
# Build training pairs using k-NN distance patterns
# ─────────────────────────────────────────────────────────────────────────────

def build_knn_distance_pairs(feats: torch.Tensor, pids: list, k: int,
                              n_pairs_per_id: int = 8, neg_ratio: float = 1.0):
    """
    For each training pair (i, j):
      - Treat i as the "query" — find its k-NN from feats (excluding self)
      - v_i [k] = distances from i to its k-NN
      - v_j [k] = distances from j to i's same k-NN
      - label = 1 if pid[i]==pid[j]

    Returns: v_q [N, k], v_g [N, k], labels [N]
    """
    n = feats.shape[0]
    pid_tensor = torch.tensor(pids)

    # Full pairwise distance matrix [n, n] (L2 squared; feats are L2-normalised)
    dist_mat = torch.cdist(feats.float(), feats.float())    # [n, n]

    # k-NN for each sample (exclude self by taking indices 1:k+1)
    knn_indices = dist_mat.argsort(dim=1)[:, 1:k + 1]      # [n, k]

    unique_ids = pid_tensor.unique().tolist()

    pos_vq, pos_vg = [], []
    neg_vq, neg_vg = [], []

    # Positive pairs (same ID)
    for uid in unique_ids:
        idx = (pid_tensor == uid).nonzero(as_tuple=True)[0].tolist()
        if len(idx) < 2:
            continue
        n_this = min(n_pairs_per_id, len(idx) * (len(idx) - 1) // 2)
        for _ in range(n_this):
            i, j = random.sample(idx, 2)
            nbs = knn_indices[i]                            # [k] ← i's neighbourhood
            v_i = dist_mat[i, nbs]                         # [k]
            v_j = dist_mat[j, nbs]                         # [k]
            pos_vq.append(v_i)
            pos_vg.append(v_j)

    n_pos = len(pos_vq)
    n_neg = int(n_pos * neg_ratio)
    all_idx = list(range(n))
    attempts = 0

    # Negative pairs (different ID)
    while len(neg_vq) < n_neg and attempts < n_neg * 10:
        i, j = random.sample(all_idx, 2)
        if pids[i] != pids[j]:
            nbs = knn_indices[i]
            neg_vq.append(dist_mat[i, nbs])
            neg_vg.append(dist_mat[j, nbs])
        attempts += 1

    vq = torch.stack(pos_vq + neg_vq)
    vg = torch.stack(pos_vg + neg_vg)
    labels = torch.tensor([1.0] * len(pos_vq) + [0.0] * len(neg_vq))
    perm = torch.randperm(len(labels))
    return vq[perm], vg[perm], labels[perm]


# ─────────────────────────────────────────────────────────────────────────────
# Eval helpers
# ─────────────────────────────────────────────────────────────────────────────

def classical_l2_rank1(q_feats, g_feats, q_pids, g_pids, q_camids, g_camids) -> float:
    dist = torch.cdist(q_feats.float(), g_feats.float()).numpy()
    return rank1_from_distmat(dist, q_pids, g_pids, q_camids, g_camids)


@torch.no_grad()
def quantum_krerank_rank1(reranker, q_feats, g_feats, q_pids, g_pids,
                          q_camids, g_camids, k: int = 20, top_rerank: int = 20,
                          alpha: float = 0.3, batch_size: int = 64,
                          logger=None) -> float:
    """
    Re-rank top_rerank classical L2 candidates using QuantumKReciprocalReranker.

    For each query q:
      1. Find q's k-NN gallery indices (neighbourhood) from L2 distances
      2. Compute v_q = distances from q to those k neighbours
      3. For each of the top_rerank gallery candidates:
             v_g = distances from g to the same k neighbours
      4. score(q, g) = VQC(v_q, v_g)
      5. Blend with classical L2 rank; rest of gallery kept in L2 order
    """
    reranker.eval()
    dist_mat = torch.cdist(q_feats.float(), g_feats.float())     # [n_q, n_g]
    classical_order = dist_mat.argsort(dim=1)                    # [n_q, n_g]
    knn_idx = classical_order[:, :k]                             # [n_q, k] — neighbourhood

    n_q = q_feats.shape[0]
    correct = 0
    g_nb_feats_cache = {}                                        # cache per query

    for q_idx in range(n_q):
        nbs = knn_idx[q_idx]                                     # [k] neighbourhood indices
        v_q = dist_mat[q_idx, nbs]                               # [k]

        # Only score top_rerank candidates through VQC
        topk_g_idx = classical_order[q_idx, :top_rerank].tolist()
        g_cands = g_feats[topk_g_idx]                            # [top_rerank, D]
        g_nb_feats = g_feats[nbs]                                # [k, D]
        v_g = torch.cdist(g_cands.float(), g_nb_feats.float())  # [top_rerank, k]

        v_q_rep = v_q.unsqueeze(0).expand(top_rerank, -1)       # [top_rerank, k]
        scores = reranker(v_q_rep, v_g).cpu()                    # [top_rerank]

        # Blend with classical position score
        classical_pos = torch.arange(top_rerank).float()
        classical_score = 1.0 - classical_pos / top_rerank
        final_score = (1.0 - alpha) * scores + alpha * classical_score

        reranked_top = [topk_g_idx[i] for i in final_score.argsort(descending=True).tolist()]
        rest = classical_order[q_idx, top_rerank:].tolist()
        full_order = reranked_top + rest

        qpid, qcam = q_pids[q_idx], q_camids[q_idx]
        for g_idx in full_order:
            gpid, gcam = g_pids[g_idx], g_camids[g_idx]
            if gpid == qpid and gcam == qcam:
                continue
            correct += int(gpid == qpid)
            break

    return correct / n_q


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantum K-Reciprocal Reranker')
    parser.add_argument('--config_file', default='configs/vit_clipreid_agvpreid.yml')
    parser.add_argument('--checkpoint',  default='logs/agvpreid_classical_80ep/best_model.pth.tar')
    parser.add_argument('--n_qubits',    default=8,   type=int)
    parser.add_argument('--n_layers',    default=2,   type=int)
    parser.add_argument('--epochs',      default=80,  type=int)
    parser.add_argument('--lr',          default=1e-3, type=float)
    parser.add_argument('--batch_size',  default=64,  type=int)
    parser.add_argument('--k',           default=20,  type=int,
                        help='k-NN neighbourhood size for distance pattern extraction')
    parser.add_argument('--k1',          default=20,  type=int,
                        help='k1 for classical k-reciprocal (Zhong et al.)')
    parser.add_argument('--k2',          default=6,   type=int,
                        help='k2 for classical k-reciprocal local query expansion')
    parser.add_argument('--lambda_val',  default=0.3, type=float,
                        help='lambda blend weight for classical k-reciprocal')
    parser.add_argument('--alpha',       default=0.3, type=float,
                        help='Blend weight for quantum reranker (0=pure VQC, 1=pure L2)')
    parser.add_argument('--output_dir',  default='logs/quantum_kreranker/80ep_k20')
    parser.add_argument('--feat_cache',  default='/tmp/qkreranker_train_feats.pt',
                        help='Cache path for training set features')
    parser.add_argument('--skip_train',  action='store_true',
                        help='Skip training — only run classical k-reciprocal + eval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('qkreranker', args.output_dir, if_train=True)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"n_qubits={args.n_qubits}, n_layers={args.n_layers}, "
                f"epochs={args.epochs}, k={args.k}, alpha={args.alpha}")

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

    # ── Extract eval features (query + gallery) ───────────────────────────────
    logger.info("Extracting query + gallery features...")
    eval_loader, num_q, *_ = make_eval_all_dataloader(cfg, case=1)
    eval_feats, eval_pids, eval_camids = extract_features(model, eval_loader, device, 'eval')

    q_feats   = eval_feats[:num_q]
    g_feats   = eval_feats[num_q:]
    q_pids    = eval_pids[:num_q]
    g_pids    = eval_pids[num_q:]
    q_camids  = eval_camids[:num_q]
    g_camids  = eval_camids[num_q:]
    logger.info(f"Query: {q_feats.shape[0]}, Gallery: {g_feats.shape[0]}")

    # ── Classical L2 baseline ─────────────────────────────────────────────────
    l2_r1 = classical_l2_rank1(q_feats, g_feats, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"Classical L2 Rank-1:           {l2_r1*100:.2f}%")

    # ── Classical k-reciprocal baseline (Zhong et al.) ───────────────────────
    logger.info(f"Running classical k-reciprocal (k1={args.k1}, k2={args.k2}, "
                f"lambda={args.lambda_val})...")
    krecip_dist = krerank(q_feats, g_feats, k1=args.k1, k2=args.k2,
                          lambda_value=args.lambda_val)
    krecip_r1 = rank1_from_distmat(krecip_dist, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"Classical k-reciprocal Rank-1: {krecip_r1*100:.2f}%  "
                f"({(krecip_r1 - l2_r1)*100:+.2f}pp vs L2)")

    if args.skip_train:
        logger.info("--skip_train set. Exiting after classical baselines.")
        sys.exit(0)

    # ── Extract / load training features ─────────────────────────────────────
    if os.path.exists(args.feat_cache):
        logger.info(f"Loading cached training features from {args.feat_cache}")
        cache = torch.load(args.feat_cache)
        train_feats, train_pids, train_camids = cache['feats'], cache['pids'], cache['camids']
    else:
        logger.info("Extracting training set features...")
        train_feats, train_pids, train_camids = extract_features(
            model, train_loader, device, 'train')
        torch.save({'feats': train_feats, 'pids': train_pids, 'camids': train_camids},
                   args.feat_cache)
        logger.info(f"Cached to {args.feat_cache}")
    logger.info(f"Train features: {train_feats.shape}, {len(set(train_pids))} IDs")

    # ── Build VQC reranker ────────────────────────────────────────────────────
    reranker = QuantumKReciprocalReranker(
        k=args.k, n_qubits=args.n_qubits, n_layers=args.n_layers)
    n_params = sum(p.numel() for p in reranker.parameters() if p.requires_grad)
    logger.info(f"QuantumKReciprocalReranker: {n_params:,} trainable params")

    optimizer = torch.optim.Adam(reranker.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────────
    from tqdm import tqdm
    logger.info(f"Training for {args.epochs} epochs on k-NN distance patterns (k={args.k})...")
    epoch_bar = tqdm(range(1, args.epochs + 1), desc='Training', unit='ep')

    for epoch in epoch_bar:
        reranker.train()
        vq, vg, labels = build_knn_distance_pairs(
            train_feats, train_pids, k=args.k,
            n_pairs_per_id=8, neg_ratio=1.0)
        dataset = TensorDataset(vq, vg, labels)
        loader  = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, drop_last=False)

        total_loss, correct, total = 0.0, 0, 0
        for bvq, bvg, blabels in loader:
            optimizer.zero_grad()
            scores = reranker(bvq, bvg)
            loss = F.binary_cross_entropy(scores, blabels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(blabels)
            preds = (scores > 0.5).float()
            correct += (preds == blabels).sum().item()
            total += len(blabels)

        scheduler.step()
        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        epoch_bar.set_postfix(loss=f'{avg_loss:.4f}', acc=f'{acc:.3f}',
                              lr=f'{scheduler.get_last_lr()[0]:.1e}')

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch[{epoch}/{args.epochs}] loss={avg_loss:.4f} acc={acc:.3f} "
                        f"lr={scheduler.get_last_lr()[0]:.1e}")
            ckpt_path = os.path.join(args.output_dir, f'reranker_ep{epoch:02d}.pt')
            torch.save(reranker.state_dict(), ckpt_path)

    # Save final weights
    final_path = os.path.join(args.output_dir, 'reranker.pt')
    torch.save(reranker.state_dict(), final_path)
    logger.info(f"Saved final weights to {final_path}")

    # ── Final eval: quantum k-reciprocal ─────────────────────────────────────
    logger.info(f"Evaluating quantum k-reciprocal (alpha={args.alpha}, top_rerank={args.k})...")
    qkr_r1 = quantum_krerank_rank1(
        reranker, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids,
        k=args.k, top_rerank=args.k, alpha=args.alpha,
        batch_size=args.batch_size, logger=logger)
    logger.info(f"Quantum k-reciprocal Rank-1:   {qkr_r1*100:.2f}%  "
                f"({(qkr_r1 - l2_r1)*100:+.2f}pp vs L2)")

    # Summary table
    logger.info("\n" + "=" * 52)
    logger.info(f"{'Method':<35} {'Rank-1':>8}  {'vs L2':>7}")
    logger.info("-" * 52)
    logger.info(f"{'Classical L2':<35} {l2_r1*100:>7.2f}%  {'baseline':>7}")
    logger.info(f"{'Classical k-reciprocal (Zhong 2017)':<35} {krecip_r1*100:>7.2f}%  "
                f"{(krecip_r1-l2_r1)*100:>+6.2f}pp")
    logger.info(f"{'Quantum k-reciprocal VQC':<35} {qkr_r1*100:>7.2f}%  "
                f"{(qkr_r1-l2_r1)*100:>+6.2f}pp")
    logger.info("=" * 52)
