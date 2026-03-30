"""
eval_qkernel.py

Evaluate the quantum fidelity kernel for retrieval similarity.

Runs three distance metrics side-by-side on the same extracted features:
  1. Euclidean (classical baseline — same as standard eval)
  2. Quantum kernel, random pre_net (no training — effect of IQP feature map alone)
  3. Quantum kernel, trained pre_net (from train_qkernel.py)

The quantum kernel replaces Euclidean distance in the gallery ranking step.
No quantum circuit is used during training — this is purely a similarity measure.

Usage:
    # Full eval, trained kernel:
    python eval_qkernel.py \\
        --config_file configs/vit_clipreid_qclassifier.yml \\
        --checkpoint logs/mars_vit_clip_reid_qclassifier/last_model.pth.tar \\
        --kernel_checkpoint logs/mars_vit_clip_reid_qkernel/pre_net.pth \\
        --n_qubits 4 --top_k 20

    # Smoke test (50 tracklets, no kernel training needed):
    python eval_qkernel.py \\
        --config_file configs/vit_clipreid_qclassifier.yml \\
        --checkpoint logs/mars_vit_clip_reid_qclassifier/last_model.pth.tar \\
        --n_qubits 4 --max_eval_batches 50
"""

import os
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from config import cfg
from utils.logger import setup_logger
from utils.metrics import euclidean_distance, eval_func
from datasets.make_dataloader_clipreid import make_eval_all_dataloader
from quantum_models.quantum_kernel import QuantumKernel


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
    np.random.seed(seed)
    random.seed(seed)


def extract_features(model, loader, device, logger):
    """
    Extract per-tracklet features from the eval loader.

    The eval loader yields batches with batch_size=1 and 6D image tensors
    [1, n_clips, seq_len, C, H, W]. We chunk-process frames and mean-pool
    across clips, matching do_inference_dense behaviour.

    Returns:
        feats   : [N, D] CPU tensor, L2-normalised
        pids    : [N] numpy int array
        camids  : [N] numpy int array
    """
    MAX_CHUNK = 30
    model.eval()
    feats, pids, camids = [], [], []

    with torch.no_grad():
        for n_done, batch in enumerate(loader):
            if n_done % 500 == 0:
                logger.info(f"  Extracting features: {n_done}/{len(loader)}")

            img, pid, camid, camids_seq, target_view, imgpath = batch
            img = img.cpu()

            # Handle 6D eval batch: [1, n_clips, T, C, H, W] → [n_clips, T, C, H, W]
            if img.dim() == 6:
                b, n, s, c, h, w = img.shape
                assert b == 1
                img = img.view(b * n, s, c, h, w)

            n_imgs = img.shape[0]

            # Replicate do_inference_dense label alignment exactly,
            # including SIE_CAMERA / SIE_VIEW gating.
            if cfg.MODEL.SIE_CAMERA and camids_seq is not None:
                camids_seq = camids_seq.cpu()
                n_labels = camids_seq.shape[0]
                if n_imgs > n_labels:
                    camids_seq = torch.repeat_interleave(camids_seq, n_imgs // n_labels)
            else:
                camids_seq = None

            if cfg.MODEL.SIE_VIEW and target_view is not None:
                target_view = target_view.cpu()
                n_labels = target_view.shape[0]
                if n_imgs > n_labels:
                    target_view = torch.repeat_interleave(target_view, n_imgs // n_labels)
            else:
                target_view = None

            feat_chunks = []
            for i in range(0, n_imgs, MAX_CHUNK):
                chunk = img[i : i + MAX_CHUNK].to(device)
                cam_chunk = camids_seq[i : i + MAX_CHUNK].to(device) if camids_seq is not None else None
                view_chunk = target_view[i : i + MAX_CHUNK].to(device) if target_view is not None else None
                f = model(chunk, cam_label=cam_chunk, view_label=view_chunk)
                feat_chunks.append(f.cpu())

            feat = torch.cat(feat_chunks).view(-1, feat_chunks[0].shape[-1]).mean(0, keepdim=True)
            feats.append(feat)

            pid_val = int(pid[0]) if hasattr(pid, '__len__') else int(pid)
            cam_val = int(camid[0]) if hasattr(camid, '__len__') else int(camid)
            pids.append(pid_val)
            camids.append(cam_val)

    feats = torch.cat(feats, dim=0)          # [N, D]
    feats = F.normalize(feats, dim=1, p=2)  # L2-normalise (same as standard eval)
    return feats, np.array(pids, dtype=np.int64), np.array(camids, dtype=np.int64)


def run_eval(distmat, q_pids, g_pids, q_camids, g_camids, label, logger):
    """Run eval_func and log CMC + mAP."""
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info(f"── {label} ──")
    logger.info(f"  mAP: {mAP:.1%}")
    for r in [1, 5, 10, 20]:
        if r <= len(cmc):
            logger.info(f"  Rank-{r:<2}: {cmc[r - 1]:.1%}")
    return cmc[0], mAP


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TF-CLIP Quantum Kernel — Retrieval Eval"
    )
    parser.add_argument(
        "--config_file",
        default="configs/vit_clipreid_qclassifier.yml",
        type=str,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Adapter model checkpoint (.pth.tar).",
    )
    parser.add_argument(
        "--kernel_checkpoint",
        default=None,
        type=str,
        help="Trained pre_net checkpoint from train_qkernel.py. "
             "If omitted, only random-init kernel is evaluated.",
    )
    parser.add_argument(
        "--n_qubits",
        default=4,
        type=int,
        help="Number of qubits in the IQP kernel. Must match kernel_checkpoint. (default: 4)",
    )
    parser.add_argument(
        "--adapter_n_qubits",
        default=8,
        type=int,
        help="n_qubits the adapter model was trained with. (default: 8)",
    )
    parser.add_argument(
        "--adapter_n_layers",
        default=2,
        type=int,
        help="n_layers the adapter model was trained with. (default: 2)",
    )
    parser.add_argument(
        "--top_k",
        default=None,
        type=int,
        help="If set, use quantum kernel only for Euclidean top-K per query "
             "(reranking mode). Full matrix if omitted. "
             "Recommended: --top_k 20 for MARS, omit for iLIDS-VID.",
    )
    parser.add_argument(
        "--max_eval_batches",
        default=None,
        type=int,
        help="Limit eval loader to this many tracklets (smoke test only).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Directory for logs. Defaults to cfg.OUTPUT_DIR.",
    )
    parser.add_argument(
        "--blend_lambdas",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.5],
        help="λ values for blended kernel: d = (1-λ)*eucl_norm + λ*(1-K). "
             "Circuits are evaluated once and blending is applied for each λ. "
             "Only runs when --kernel_checkpoint and --top_k are both set. "
             "(default: 0.1 0.2 0.3 0.5)",
    )
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.merge_from_list(["DATALOADER.NUM_WORKERS", "0"])
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = args.output_dir or cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("TFCLIP.qkernel_eval", output_dir, if_train=False)
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Eval loader ──────────────────────────────────────────────────────────
    val_loader, num_query, num_classes, camera_num, view_num = (
        make_eval_all_dataloader(cfg)
    )
    logger.info(
        f"Eval loader: {len(val_loader)} tracklets, "
        f"num_query={num_query}, dataset={cfg.DATASETS.NAMES}"
    )

    if args.max_eval_batches is not None:
        actual = min(args.max_eval_batches, len(val_loader))
        num_query = min(num_query, max(1, actual // 2))
        logger.info(
            f"[smoke] Limiting to {actual} tracklets, num_query={num_query}. "
            f"Metrics NOT statistically valid."
        )
        val_loader = _LimitedLoader(val_loader, actual)

    # ── Feature extraction (cached) ──────────────────────────────────────────
    feat_cache = os.path.join(output_dir, "eval_feats.pt")
    if args.max_eval_batches is None and os.path.exists(feat_cache):
        logger.info(f"Loading cached eval features from {feat_cache}")
        cache = torch.load(feat_cache, map_location="cpu", weights_only=False)
        feats = cache["feats"]
        all_pids = cache["pids"]
        all_camids = cache["camids"]
        num_query = cache.get("num_query", num_query)
        logger.info(f"Features: {feats.shape}, normalised.")
    else:
        # ── Model ────────────────────────────────────────────────────────────
        from quantum_models.make_model_adapter import make_model

        model = make_model(
            cfg,
            num_class=num_classes,
            camera_num=camera_num,
            view_num=view_num,
            n_qubits=args.adapter_n_qubits,
            n_layers=args.adapter_n_layers,
        )
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:3]}")
        model.to(device).eval()
        logger.info(f"Loaded adapter checkpoint: {args.checkpoint}")

        logger.info("Extracting test features ...")
        feats, all_pids, all_camids = extract_features(model, val_loader, device, logger)
        logger.info(f"Features: {feats.shape}, normalised.")

        if args.max_eval_batches is None:
            torch.save(
                {"feats": feats, "pids": all_pids, "camids": all_camids, "num_query": num_query},
                feat_cache,
            )
            logger.info(f"Saved eval features to {feat_cache}")

    # Split into query / gallery (same convention as R1_mAP_eval.compute)
    qf = feats[:num_query]
    gf = feats                          # all (eval_func removes self-matches)
    q_pids  = all_pids[:num_query]
    g_pids  = all_pids
    q_camids = all_camids[:num_query]
    g_camids = all_camids

    # ── 1. Classical Euclidean baseline ──────────────────────────────────────
    logger.info("Computing Euclidean distance matrix ...")
    eucl_dist = euclidean_distance(qf, gf)
    r1_eucl, map_eucl = run_eval(
        eucl_dist, q_pids, g_pids, q_camids, g_camids,
        "Euclidean (classical baseline)", logger
    )

    # ── 2. Quantum kernel, random pre_net ────────────────────────────────────
    in_features = feats.shape[1]
    qk_random = QuantumKernel(in_features=in_features, n_qubits=args.n_qubits)
    mode = f"top-{args.top_k} reranking" if args.top_k else "full matrix"
    logger.info(f"Computing quantum kernel distance (random pre_net, {mode}) ...")
    qk_dist_random = qk_random.distance_matrix(qf, gf, top_k=args.top_k)
    r1_qk_rand, map_qk_rand = run_eval(
        qk_dist_random, q_pids, g_pids, q_camids, g_camids,
        f"Quantum kernel, random pre_net ({mode})", logger
    )

    # ── 3. Quantum kernel, trained pre_net (if available) ───────────────────
    blend_results = {}   # {lambda: (r1, mAP)}
    if args.kernel_checkpoint:
        ckpt = torch.load(args.kernel_checkpoint, map_location="cpu", weights_only=False)
        n_q = ckpt.get("n_qubits", args.n_qubits)
        qk_trained = QuantumKernel(in_features=in_features, n_qubits=n_q)
        qk_trained.pre_net.load_state_dict(ckpt["pre_net"])
        logger.info(f"Loaded kernel checkpoint: {args.kernel_checkpoint} (n_qubits={n_q})")

        logger.info(f"Computing quantum kernel distance (trained pre_net, {mode}) ...")
        qk_dist_trained = qk_trained.distance_matrix(qf, gf, top_k=args.top_k)
        r1_qk_tr, map_qk_tr = run_eval(
            qk_dist_trained, q_pids, g_pids, q_camids, g_camids,
            f"Quantum kernel, trained pre_net ({mode})", logger
        )

        # ── 4. Blended kernel sweep (single circuit pass, multiple λ) ────────
        if args.top_k and args.blend_lambdas:
            logger.info(
                f"Computing blended kernels λ ∈ {args.blend_lambdas} "
                f"(top-{args.top_k}, single circuit pass) ..."
            )
            blend_dists = qk_trained.distance_matrix_blended(
                qf, gf, top_k=args.top_k, lambdas=args.blend_lambdas
            )
            for lam, bdist in blend_dists.items():
                r1_b, map_b = run_eval(
                    bdist, q_pids, g_pids, q_camids, g_camids,
                    f"Blended kernel λ={lam:.2f} ({mode})", logger,
                )
                blend_results[lam] = (r1_b, map_b)
    else:
        logger.info("No --kernel_checkpoint provided; skipping trained kernel eval.")
        r1_qk_tr, map_qk_tr = None, None

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("══════════════ SUMMARY ══════════════")
    logger.info(f"  Euclidean baseline        Rank-1: {r1_eucl:.1%}  mAP: {map_eucl:.1%}")
    logger.info(f"  Quantum (random pre_net)  Rank-1: {r1_qk_rand:.1%}  mAP: {map_qk_rand:.1%}")
    if r1_qk_tr is not None:
        logger.info(f"  Quantum (trained pre_net) Rank-1: {r1_qk_tr:.1%}  mAP: {map_qk_tr:.1%}")
    for lam, (r1_b, map_b) in blend_results.items():
        logger.info(f"  Blended λ={lam:.2f}             Rank-1: {r1_b:.1%}  mAP: {map_b:.1%}")
    logger.info("═════════════════════════════════════")
