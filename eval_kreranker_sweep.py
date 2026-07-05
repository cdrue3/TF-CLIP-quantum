"""
eval_kreranker_sweep.py

Sweep all saved QuantumKReciprocalReranker checkpoints and compare against:
  - Classical L2 baseline
  - Classical k-reciprocal (Zhong et al.) — evaluated once, stateless

Usage:
  python eval_kreranker_sweep.py \\
      --config_file configs/vit_clipreid_agvpreid.yml \\
      --backbone_checkpoint logs/agvpreid_classical_80ep/best_model.pth.tar \\
      --ckpt_dir logs/quantum_kreranker/80ep_k20 \\
      --k 20 --alpha 0.3 \\
      DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8
"""

import os
import sys
import argparse
import glob
import re

import torch
import torch.nn.functional as F

from config import cfg
from utils.logger import setup_logger
from utils.kreranker import krerank, rank1_from_distmat
from datasets.make_dataloader_clipreid import make_dataloader, make_eval_all_dataloader
from model.make_model_clipreid import make_model
from quantum_models.postprocessing.quantum_kreranker import QuantumKReciprocalReranker

from train_qkreranker import extract_features, classical_l2_rank1, quantum_krerank_rank1


def load_checkpoints(directory, pattern='reranker_ep*.pt'):
    paths = glob.glob(os.path.join(directory, pattern))
    results = []
    for p in paths:
        m = re.search(r'ep(\d+)', os.path.basename(p))
        if m:
            results.append((int(m.group(1)), p))
    results.sort(key=lambda x: x[0])
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',         default='configs/vit_clipreid_agvpreid.yml')
    parser.add_argument('--backbone_checkpoint', default='logs/agvpreid_classical_80ep/best_model.pth.tar')
    parser.add_argument('--ckpt_dir',            default='logs/quantum_kreranker/80ep_k20')
    parser.add_argument('--n_qubits',            default=8,   type=int)
    parser.add_argument('--n_layers',            default=2,   type=int)
    parser.add_argument('--k',                   default=20,  type=int)
    parser.add_argument('--k1',                  default=20,  type=int)
    parser.add_argument('--k2',                  default=6,   type=int)
    parser.add_argument('--lambda_val',          default=0.3, type=float)
    parser.add_argument('--alpha',               default=0.3, type=float)
    parser.add_argument('--batch_size',          default=64,  type=int)
    parser.add_argument('--output_dir',          default='logs/kreranker_sweep')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('kreranker_sweep', args.output_dir, if_train=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Backbone ──────────────────────────────────────────────────────────────
    train_loader, _, val_loader, num_query, num_classes, camera_num, view_num = \
        make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    state = torch.load(args.backbone_checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Eval features (once) ─────────────────────────────────────────────────
    logger.info("Extracting query + gallery features...")
    eval_loader, num_q, *_ = make_eval_all_dataloader(cfg, case=1)
    eval_feats, eval_pids, eval_camids = extract_features(model, eval_loader, device, 'eval')

    q_feats  = eval_feats[:num_q];  g_feats  = eval_feats[num_q:]
    q_pids   = eval_pids[:num_q];   g_pids   = eval_pids[num_q:]
    q_camids = eval_camids[:num_q]; g_camids = eval_camids[num_q:]

    # ── Baselines (computed once) ─────────────────────────────────────────────
    l2_r1 = classical_l2_rank1(q_feats, g_feats, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"Classical L2 Rank-1:           {l2_r1*100:.2f}%")

    logger.info(f"Running classical k-reciprocal (k1={args.k1}, k2={args.k2}, "
                f"lambda={args.lambda_val})...")
    krecip_dist = krerank(q_feats, g_feats, k1=args.k1, k2=args.k2,
                          lambda_value=args.lambda_val)
    krecip_r1 = rank1_from_distmat(krecip_dist, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"Classical k-reciprocal Rank-1: {krecip_r1*100:.2f}%  "
                f"({(krecip_r1-l2_r1)*100:+.2f}pp)")

    # ── Sweep quantum kreranker checkpoints ───────────────────────────────────
    checkpoints = load_checkpoints(args.ckpt_dir)
    if not checkpoints:
        logger.warning(f"No reranker_ep*.pt files found in {args.ckpt_dir}")
        sys.exit(1)

    logger.info(f"\nSweeping {len(checkpoints)} checkpoints from {args.ckpt_dir}")

    rows = []
    for epoch, path in checkpoints:
        reranker = QuantumKReciprocalReranker(
            k=args.k, n_qubits=args.n_qubits, n_layers=args.n_layers)
        reranker.load_state_dict(torch.load(path, map_location='cpu'))

        r1 = quantum_krerank_rank1(
            reranker, q_feats, g_feats, q_pids, g_pids, q_camids, g_camids,
            k=args.k, top_rerank=args.k, alpha=args.alpha, batch_size=args.batch_size)
        delta = (r1 - l2_r1) * 100
        logger.info(f"ep{epoch:02d}  quantum-kreranker  {r1*100:.2f}%  ({delta:+.2f}pp vs L2)")
        rows.append((epoch, r1))

    # ── Summary ───────────────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, 'results_sweep.txt')
    best_r1 = max(r1 for _, r1 in rows)
    best_ep  = max(rows, key=lambda x: x[1])[0]

    with open(results_path, 'w') as f:
        f.write(f"Checkpoint sweep — quantum k-reciprocal — {args.ckpt_dir}\n")
        f.write(f"{'Epoch':<12} {'Rank-1':>8}   {'vs L2':>8}\n")
        f.write("-" * 34 + "\n")
        for ep, r1 in rows:
            f.write(f"ep{ep:02d}        {r1*100:>6.2f}%   {(r1-l2_r1)*100:>+7.2f}pp\n")

    logger.info(f"\n{'='*50}")
    logger.info(f"Classical L2:           {l2_r1*100:.2f}%  (baseline)")
    logger.info(f"Classical k-reciprocal: {krecip_r1*100:.2f}%  ({(krecip_r1-l2_r1)*100:+.2f}pp)")
    logger.info(f"Best quantum kreranker: {best_r1*100:.2f}%  ({(best_r1-l2_r1)*100:+.2f}pp) at ep{best_ep:02d}")
    logger.info(f"Quantum vs classical k-reciprocal: {(best_r1-krecip_r1)*100:+.2f}pp")
    logger.info(f"Results written to {results_path}")
