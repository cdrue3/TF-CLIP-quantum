"""
utils/kreranker.py

Classical k-reciprocal re-ranking (Zhong et al. CVPR 2017).

Usage:
    distmat = krerank(q_feats, g_feats, k1=20, k2=6, lambda_value=0.3)
    # distmat is the re-ranked distance matrix [n_q, n_g]; lower = better match
"""

import numpy as np
import torch


def krerank(q_feats, g_feats, k1: int = 20, k2: int = 6,
            lambda_value: float = 0.3) -> np.ndarray:
    """
    Zhong et al. CVPR 2017 k-reciprocal encoding re-ranking.

    Args:
        q_feats:      [n_q, D] L2-normalised query features (torch.Tensor or np.ndarray)
        g_feats:      [n_g, D] L2-normalised gallery features
        k1:           k for k-reciprocal neighbourhood (default 20)
        k2:           k for local query expansion (default 6)
        lambda_value: blend with original L2 distance (default 0.3)

    Returns:
        np.ndarray [n_q, n_g]: final re-ranked distance (lower = better)
    """
    if isinstance(q_feats, torch.Tensor):
        q_feats = q_feats.float().cpu().numpy()
    if isinstance(g_feats, torch.Tensor):
        g_feats = g_feats.float().cpu().numpy()

    n_q, n_g = q_feats.shape[0], g_feats.shape[0]

    # Stack all features: probes = queries + gallery
    all_feats = np.concatenate([q_feats, g_feats], axis=0)  # [n_q+n_g, D]
    n_all = all_feats.shape[0]

    # Pairwise L2 squared distances (features are L2-normalised so ||a-b||^2 = 2 - 2<a,b>)
    sim = all_feats @ all_feats.T                           # [n_all, n_all]
    orig_dist = np.clip(2.0 - 2.0 * sim, 0.0, None)        # squared L2, [n_all, n_all]
    np.fill_diagonal(orig_dist, 0.0)

    # Sort by distance for each probe
    sorted_idx = np.argsort(orig_dist, axis=1)              # [n_all, n_all]

    # Build k-reciprocal neighbours R(i, k1)
    # j ∈ R(i, k1)  iff  j ∈ kNN(i, k1) AND i ∈ kNN(j, k1)
    knn_k1 = sorted_idx[:, 1:k1 + 1]                       # [n_all, k1]  (skip self)

    def get_krecip(i):
        forward = set(knn_k1[i].tolist())
        backward = set(knn_k1[:, :].tolist()[0])            # placeholder
        # j in R(i) iff i in kNN(j)
        r = set()
        for j in forward:
            if i in knn_k1[j]:
                r.add(j)
        return r

    # Pre-compute: for each probe i, which k1-NNs have i as a k1-NN
    knn_k1_sets = [set(knn_k1[i].tolist()) for i in range(n_all)]
    knn_k1_half_sets = [set(sorted_idx[i, 1:k1 // 2 + 1].tolist()) for i in range(n_all)]

    R = []
    for i in range(n_all):
        r_i = set()
        for j in knn_k1_sets[i]:
            if i in knn_k1_sets[j]:
                r_i.add(j)
        R.append(r_i)

    # Expand R: for each r in R(i, k1), add R(r, k1//2) if large overlap
    R_expanded = []
    for i in range(n_all):
        r_i = set(R[i])
        for r in list(R[i]):
            r_r_half = knn_k1_half_sets[r]
            if len(r_r_half & r_i) >= 2 / 3 * len(r_r_half):
                r_i = r_i | r_r_half
        R_expanded.append(list(r_i))

    # Build weighted Jaccard encoding V[i] ∈ R^{n_all}
    # V[i][j] = exp(-d(i,j)) / |R_expanded(i)|  if j in R_expanded(i), else 0
    V = np.zeros((n_all, n_all), dtype=np.float32)
    for i in range(n_all):
        r_exp = R_expanded[i]
        if len(r_exp) == 0:
            continue
        dists = orig_dist[i, r_exp]
        weights = np.exp(-dists)
        V[i, r_exp] = weights / weights.sum()

    # Local query expansion: smooth V with k2-NN
    if k2 > 1:
        V_qe = np.zeros_like(V)
        knn_k2 = sorted_idx[:, 1:k2 + 1]                   # [n_all, k2]
        for i in range(n_all):
            # Average over k2 nearest neighbours (including self)
            neighbors = [i] + knn_k2[i].tolist()
            V_qe[i] = V[neighbors].mean(axis=0)
        V = V_qe

    # Jaccard distance: 1 - sum(min(V_i, V_j)) / sum(max(V_i, V_j))
    # Computed only for query-gallery pairs
    jaccard_dist = np.zeros((n_q, n_g), dtype=np.float32)
    for q_idx in range(n_q):
        v_q = V[q_idx]                                      # [n_all]
        v_g = V[n_q:n_q + n_g]                             # [n_g, n_all]
        intersect = np.minimum(v_q[None, :], v_g).sum(axis=1)
        union = np.maximum(v_q[None, :], v_g).sum(axis=1)
        jaccard_dist[q_idx] = 1.0 - intersect / (union + 1e-12)

    original_dist_qg = orig_dist[:n_q, n_q:]               # [n_q, n_g]

    final_dist = lambda_value * original_dist_qg + (1.0 - lambda_value) * jaccard_dist
    return final_dist


def rank1_from_distmat(distmat, q_pids, g_pids, q_camids, g_camids) -> float:
    """Compute Rank-1 from a [n_q, n_g] distance matrix (lower=better)."""
    indices = np.argsort(distmat, axis=1)
    correct = 0
    for q_idx in range(len(q_pids)):
        qpid, qcam = q_pids[q_idx], q_camids[q_idx]
        for g_idx in indices[q_idx]:
            gpid, gcam = g_pids[g_idx], g_camids[g_idx]
            if gpid == qpid and gcam == qcam:
                continue
            correct += int(gpid == qpid)
            break
    return correct / len(q_pids)
