"""
eval_reranker_sweep.py

Sweep all saved reranker checkpoints (quantum and/or classical) and report
Rank-1 at each epoch alongside the classical L2 baseline.

Usage:
  python eval_reranker_sweep.py \\
      --config_file configs/vit_clipreid_agvpreid.yml \\
      --backbone_checkpoint logs/agvpreid_classical_80ep/best_model.pth.tar \\
      --quantum_dir  logs/quantum_reranker/80ep_k20 \\
      --classical_dir logs/classical_reranker/80ep_k20 \\
      --top_k 20 --alpha 0.3 \\
      DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8
"""

import os
import sys
import argparse
import glob
import re
import logging

import torch
import torch.nn.functional as F

from config import cfg
from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader, make_eval_all_dataloader
from model.make_model_clipreid import make_model
from quantum_models.postprocessing.quantum_reranker import QuantumPairwiseReranker
from quantum_models.postprocessing.classical_reranker import ClassicalPairwiseReranker

# Reuse helpers from train_qreranker
from train_qreranker import extract_features, classical_rank1, rerank_with_vqc


def load_checkpoints(directory, pattern='reranker_ep*.pt'):
    """Return sorted list of (epoch, path) for checkpoints matching pattern."""
    paths = glob.glob(os.path.join(directory, pattern))
    results = []
    for p in paths:
        m = re.search(r'ep(\d+)', os.path.basename(p))
        if m:
            results.append((int(m.group(1)), p))
    results.sort(key=lambda x: x[0])
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reranker Checkpoint Sweep')
    parser.add_argument('--config_file',          default='configs/vit_clipreid_agvpreid.yml')
    parser.add_argument('--backbone_checkpoint',  default='logs/agvpreid_classical_80ep/best_model.pth.tar')
    parser.add_argument('--quantum_dir',          default=None, help='Dir containing quantum reranker_ep*.pt files')
    parser.add_argument('--classical_dir',        default=None, help='Dir containing classical reranker_ep*.pt files')
    parser.add_argument('--n_qubits',             default=8,   type=int)
    parser.add_argument('--n_layers',             default=2,   type=int)
    parser.add_argument('--top_k',                default=20,  type=int)
    parser.add_argument('--alpha',                default=0.3, type=float)
    parser.add_argument('--batch_size',           default=64,  type=int)
    parser.add_argument('--output_dir',           default='logs/reranker_sweep')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.quantum_dir and not args.classical_dir:
        print("ERROR: provide at least one of --quantum_dir or --classical_dir")
        sys.exit(1)

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('reranker_sweep', args.output_dir, if_train=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # ── Load frozen backbone ──────────────────────────────────────────────────
    train_loader, _, val_loader, num_query, num_classes, camera_num, view_num = \
        make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    state = torch.load(args.backbone_checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Extract eval features (once) ─────────────────────────────────────────
    logger.info("Extracting query + gallery features...")
    eval_loader, num_q, *_ = make_eval_all_dataloader(cfg, case=1)
    eval_feats, eval_pids, eval_camids = extract_features(model, eval_loader, device, 'eval')

    q_feats  = eval_feats[:num_q];  g_feats  = eval_feats[num_q:]
    q_pids   = eval_pids[:num_q];   g_pids   = eval_pids[num_q:]
    q_camids = eval_camids[:num_q]; g_camids = eval_camids[num_q:]
    logger.info(f"Query: {len(q_pids)}, Gallery: {len(g_pids)}")

    # Classical L2 baseline (computed once)
    baseline_r1 = classical_rank1(q_feats, g_feats, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"Classical L2 Rank-1: {baseline_r1*100:.1f}%")

    in_features = q_feats.shape[1]

    # ── Sweep helper ─────────────────────────────────────────────────────────
    def sweep(ckpt_dir, mode):
        checkpoints = load_checkpoints(ckpt_dir)
        if not checkpoints:
            logger.warning(f"No reranker_ep*.pt files found in {ckpt_dir}")
            return []
        logger.info(f"\n{'='*55}")
        logger.info(f"Sweeping {mode} reranker: {ckpt_dir}")
        logger.info(f"{'='*55}")
        rows = []
        for epoch, path in checkpoints:
            if mode == 'quantum':
                reranker = QuantumPairwiseReranker(in_features, args.n_qubits, args.n_layers)
            else:
                reranker = ClassicalPairwiseReranker(in_features, args.n_qubits, args.n_layers)
            reranker.load_state_dict(torch.load(path, map_location='cpu'))
            reranker.eval()

            r1, _ = rerank_with_vqc(
                reranker, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids,
                top_k=args.top_k, batch_size=args.batch_size, alpha=args.alpha,
            )
            delta = (r1 - baseline_r1) * 100
            sign = '+' if delta >= 0 else ''
            logger.info(f"ep{epoch:02d}  {mode:9s}  {r1*100:.1f}%  ({sign}{delta:.1f}pp vs classical)")
            rows.append((epoch, mode, r1))
        return rows

    all_rows = []
    if args.quantum_dir:
        all_rows += sweep(args.quantum_dir, 'quantum')
    if args.classical_dir:
        all_rows += sweep(args.classical_dir, 'classical')

    # ── Combined summary table ────────────────────────────────────────────────
    logger.info(f"\n{'='*55}")
    logger.info("SUMMARY")
    logger.info(f"Classical L2 baseline: {baseline_r1*100:.1f}%")
    logger.info(f"{'Epoch':<8} {'Quantum R1':>12} {'Classical R1':>14} {'Q-delta':>9} {'C-delta':>9}")
    logger.info("-" * 55)

    q_by_ep = {ep: r1 for ep, mode, r1 in all_rows if mode == 'quantum'}
    c_by_ep = {ep: r1 for ep, mode, r1 in all_rows if mode == 'classical'}
    all_epochs = sorted(set(q_by_ep) | set(c_by_ep))

    best_q, best_c = 0.0, 0.0
    for ep in all_epochs:
        q_str = f"{q_by_ep[ep]*100:.1f}%" if ep in q_by_ep else "   —  "
        c_str = f"{c_by_ep[ep]*100:.1f}%" if ep in c_by_ep else "   —  "
        qd = f"{(q_by_ep[ep]-baseline_r1)*100:+.1f}pp" if ep in q_by_ep else "  —  "
        cd = f"{(c_by_ep[ep]-baseline_r1)*100:+.1f}pp" if ep in c_by_ep else "  —  "
        logger.info(f"ep{ep:02d}     {q_str:>12} {c_str:>14} {qd:>9} {cd:>9}")
        if ep in q_by_ep: best_q = max(best_q, q_by_ep[ep])
        if ep in c_by_ep: best_c = max(best_c, c_by_ep[ep])

    logger.info("-" * 55)
    if q_by_ep: logger.info(f"Best quantum:    {best_q*100:.1f}%  ({(best_q-baseline_r1)*100:+.1f}pp)")
    if c_by_ep: logger.info(f"Best classical:  {best_c*100:.1f}%  ({(best_c-baseline_r1)*100:+.1f}pp)")
    if q_by_ep and c_by_ep:
        logger.info(f"Quantum vs MLP:  {(best_q-best_c)*100:+.1f}pp")
