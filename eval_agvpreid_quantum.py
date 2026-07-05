"""
eval_agvpreid_quantum.py

Quantum retrieval evaluation for AG-VPReID Case 1.
Replaces or augments the classical L2 nearest-neighbour search at eval time.
No changes to the trained model.

Modes:
  --retrieval swap_test   : Rerank top-K with quantum swap test (accuracy focus)
  --retrieval durr_hoyer  : Dürr-Høyer quantum minimum finding (speedup demo)

Usage:
  python eval_agvpreid_quantum.py \\
      --config_file configs/vit_clipreid_agvpreid.yml \\
      --checkpoint logs/agvpreid_classical_qhed_40ep/checkpoint_ep40.pth.tar \\
      --retrieval swap_test --top_k 50 \\
      DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8

  python eval_agvpreid_quantum.py \\
      --retrieval durr_hoyer \\
      ...
"""

import os
import math
import time
import random
import numpy as np
import torch
import argparse

from config import cfg
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval
from datasets.make_dataloader_clipreid import make_eval_all_dataloader
from model.make_model_clipreid import make_model
from utils.quantum_retrieval import QuantumSwapTestRanker, DurrHoyerSearch


def extract_features(cfg, model, val_loader, device='cuda'):
    """Extract all query and gallery features using the trained model."""
    model.eval()
    feats, pids, camids = [], [], []

    with torch.no_grad():
        for img, pid, camid, camids_batch, target_view, _ in val_loader:
            img = img.to(device)
            camids_batch = camids_batch.to(device) if cfg.MODEL.SIE_CAMERA else None
            target_view  = target_view.to(device)  if cfg.MODEL.SIE_VIEW   else None
            feat = model(img, cam_label=camids_batch, view_label=target_view)
            feats.append(feat.cpu())
            pids.extend(torch.as_tensor(pid).view(-1).tolist())
            camids.extend(torch.as_tensor(camid).view(-1).tolist())

    return torch.cat(feats, dim=0), pids, camids


def classical_ranking(query_feats, gallery_feats):
    """L2 nearest-neighbour ranking. Returns [n_query, n_gallery] sorted indices."""
    # Compute pairwise L2 distances
    dist = torch.cdist(query_feats.float(), gallery_feats.float(), p=2)
    return dist.argsort(dim=1)   # [n_query, n_gallery]


def compute_cmc_map(query_feats, gallery_feats, query_pids, gallery_pids,
                    query_camids, gallery_camids, ranked_indices, num_query,
                    feat_norm=True):
    """Evaluate Rank-1/5 and mAP from precomputed ranked indices."""
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=feat_norm)
    evaluator.reset()

    # Feed as (feat, pid, camid) in ranked order — use original feats for eval
    # Rebuild using the evaluator's standard path
    n_q = len(query_pids)
    n_g = len(gallery_pids)

    all_feats  = torch.cat([query_feats, gallery_feats], dim=0)
    all_pids   = query_pids   + gallery_pids
    all_camids = query_camids + gallery_camids

    for i in range(len(query_pids) + len(gallery_pids)):
        evaluator.update((all_feats[i:i+1], [all_pids[i]], [all_camids[i]]))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    return cmc, mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/vit_clipreid_agvpreid.yml')
    parser.add_argument('--checkpoint',  default=None)
    parser.add_argument('--retrieval',   default='swap_test',
                        choices=['swap_test', 'durr_hoyer'])
    parser.add_argument('--top_k',       default=50, type=int,
                        help='Candidates to rerank (swap_test only).')
    parser.add_argument('--n_qubits',    default=8,  type=int,
                        help='Qubits per feature register in swap test.')
    parser.add_argument('--output_dir',  default='/tmp/quantum_retrieval_eval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.merge_from_list(['DATALOADER.NUM_WORKERS', '0'])
    cfg.freeze()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('TFCLIP', args.output_dir, if_train=False)
    logger.info(f"[quantum_retrieval] mode={args.retrieval}")

    # ── Load data & model ──────────────────────────────────────────────────
    val_loader, num_query, num_classes, camera_num, view_num = \
        make_eval_all_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes,
                       camera_num=camera_num, view_num=view_num)
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    model.cuda()

    # ── Extract features ───────────────────────────────────────────────────
    logger.info("Extracting features...")
    t0 = time.time()
    all_feats, all_pids, all_camids = extract_features(cfg, model, val_loader)
    logger.info(f"Features extracted in {time.time()-t0:.1f}s — shape {all_feats.shape}, "
                f"len(pids)={len(all_pids)}, len(camids)={len(all_camids)}, num_query={num_query}")
    # Flatten any nested lists from the dataloader (e.g. camid shape [B,1] → [B])
    all_pids   = [p[0] if isinstance(p, (list, tuple)) else p for p in all_pids]
    all_camids = [c[0] if isinstance(c, (list, tuple)) else c for c in all_camids]
    assert len(all_pids) == all_feats.shape[0], \
        f"pids/feats mismatch: {len(all_pids)} vs {all_feats.shape[0]}"

    query_feats   = all_feats[:num_query]
    gallery_feats = all_feats[num_query:]
    query_pids    = all_pids[:num_query]
    gallery_pids  = all_pids[num_query:]
    query_camids  = all_camids[:num_query]
    gallery_camids= all_camids[num_query:]
    n_gallery     = len(gallery_pids)
    logger.info(f"Split: {num_query} query, {n_gallery} gallery")

    # ── Classical baseline ─────────────────────────────────────────────────
    logger.info("Computing classical L2 ranking (baseline)...")
    t_classical = time.time()
    classical_ranked = classical_ranking(query_feats, gallery_feats)
    classical_time   = time.time() - t_classical
    logger.info(f"Classical ranking: {classical_time*1000:.1f}ms total, "
                f"{classical_time/num_query*1000:.3f}ms/query")

    # ── Quantum retrieval ──────────────────────────────────────────────────
    if args.retrieval == 'swap_test':
        logger.info(f"Running quantum swap test reranking (top-{args.top_k}, "
                    f"n_qubits={args.n_qubits})...")
        ranker = QuantumSwapTestRanker(n_qubits=args.n_qubits, top_k=args.top_k)

        t_quantum = time.time()
        quantum_ranked = ranker.rerank(query_feats, gallery_feats, classical_ranked)
        quantum_time   = time.time() - t_quantum

        # ── Speed metrics ──────────────────────────────────────────────────
        # Note: swap test is O(K) circuit calls per query vs O(N) classical distance ops.
        # Swap test trades speed for a DIFFERENT similarity metric (quantum overlap
        # vs L2 distance). Speed comparison is not apples-to-apples since they compute
        # different things. The key metric is accuracy improvement (or not).
        # Save reranked result so eval crash doesn't lose the 72-min run
        _ckpt = os.path.join(args.output_dir, 'swap_test_ranked.pt')
        torch.save({'quantum_ranked': quantum_ranked,
                    'classical_ranked': classical_ranked,
                    'query_pids': query_pids, 'gallery_pids': gallery_pids,
                    'query_camids': query_camids, 'gallery_camids': gallery_camids},
                   _ckpt)
        logger.info(f"Saved reranked indices to {_ckpt}")

        logger.info(
            f"\n{'='*60}\n"
            f"SWAP TEST SPEED METRICS\n"
            f"  Classical ranking (L2, full gallery):  {classical_time*1000:.1f}ms total\n"
            f"  Quantum reranking (swap test, top-{args.top_k}): {quantum_time:.1f}s total\n"
            f"  Quantum is {quantum_time/classical_time:.0f}x SLOWER on simulator\n"
            f"  (Expected: swap test is O(K) circuit calls, each ~ms on CPU)\n"
            f"  NOTE: Swap test computes quantum overlap |<q|g>|^2, not L2.\n"
            f"  The goal is ACCURACY improvement from the different metric,\n"
            f"  not speed improvement. Speed advantage only applies on hardware\n"
            f"  where circuit calls are O(1) vs O(D) classical dot products.\n"
            f"  On quantum hardware: K circuit calls vs K*D classical ops\n"
            f"  → speedup proportional to D/circuit_depth ≈ {all_feats.shape[1]}/depth\n"
            f"{'='*60}"
        )
        ranked_for_eval = quantum_ranked

    elif args.retrieval == 'durr_hoyer':
        logger.info("Running Dürr-Høyer quantum minimum finding...")
        logger.info(f"Gallery size N={n_gallery}, √N={math.sqrt(n_gallery):.1f}")

        import math
        searcher = DurrHoyerSearch()
        t_quantum  = time.time()
        dh_indices, stats = searcher.search(query_feats, gallery_feats)
        quantum_time = time.time() - t_quantum

        logger.info(
            f"\n{'='*60}\n"
            f"DÜRR-HØYER QUANTUM SEARCH METRICS\n"
            f"  Gallery size N:                 {stats['n_gallery']}\n"
            f"  Classical oracle calls/query:   {stats['classical_oracle_calls']}\n"
            f"  Quantum oracle calls/query:     {stats['quantum_oracle_calls']:.1f}\n"
            f"  √N (theoretical minimum):       {stats['sqrt_N']:.1f}\n"
            f"  Theoretical speedup (N/calls):  {stats['theoretical_speedup']:.1f}x\n"
            f"  Actual sim time (CPU):          {quantum_time:.1f}s\n"
            f"  Classical baseline time:        {classical_time*1000:.1f}ms\n"
            f"  Simulator overhead factor:      {quantum_time/(classical_time+1e-9):.0f}x\n"
            f"  ---\n"
            f"  {stats['note']}\n"
            f"{'='*60}"
        )

        # Build full ranking: DH gives top-1, fill rest with classical order
        ranked_for_eval = classical_ranked.clone()
        for qi in range(num_query):
            dh_top = dh_indices[qi].item()
            # Move DH result to front, shift others
            classical_pos = (classical_ranked[qi] == dh_top).nonzero(as_tuple=True)[0]
            if len(classical_pos) > 0:
                pos = classical_pos[0].item()
                ranked_for_eval[qi] = torch.cat([
                    classical_ranked[qi, pos:pos+1],
                    classical_ranked[qi, :pos],
                    classical_ranked[qi, pos+1:]
                ])

    # ── Evaluate both ──────────────────────────────────────────────────────
    def rank1_from_indices(ranked_idx, q_pids, g_pids, q_cams, g_cams):
        """Compute Rank-1 from ranked gallery indices."""
        correct = 0
        for qi in range(len(q_pids)):
            qpid, qcam = q_pids[qi], q_cams[qi]
            for gidx in ranked_idx[qi].tolist():
                gpid, gcam = g_pids[gidx], g_cams[gidx]
                if gpid == qpid and gcam == qcam:
                    continue  # same camera same id — skip (junk)
                if gpid == qpid:
                    correct += 1
                break
        return correct / len(q_pids)

    r1_classical = rank1_from_indices(
        classical_ranked, query_pids, gallery_pids, query_camids, gallery_camids)
    r1_quantum = rank1_from_indices(
        ranked_for_eval, query_pids, gallery_pids, query_camids, gallery_camids)

    logger.info(
        f"\n{'='*60}\n"
        f"RESULTS SUMMARY\n"
        f"  Classical L2 Rank-1:  {r1_classical:.1%}\n"
        f"  Quantum {args.retrieval} Rank-1: {r1_quantum:.1%}\n"
        f"  Delta:                {(r1_quantum - r1_classical):+.1%}\n"
        f"{'='*60}"
    )
