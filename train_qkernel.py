"""
train_qkernel.py

Train the QuantumKernel.pre_net (Linear in_features → n_qubits) via BCE pair loss.

The IQP circuit has no learnable parameters — pre_net is the only trainable component.
It learns to project features into an angle space where same-ID pairs have high
quantum fidelity and different-ID pairs have low fidelity.

Pipeline:
  1. Load frozen adapter model from checkpoint.
  2. One inference pass over training set → save [N, D] features + pids to disk (cached).
     Re-run skips extraction if cache exists.
  3. Sample random positive (same pid) and negative (different pid) pairs.
  4. Train only pre_net via BCE loss: BCE(K(x1, x2), y), y=1 same pid, y=0 different.
  5. Save pre_net checkpoint.

Usage:
    python train_qkernel.py \\
        --config_file configs/vit_clipreid_qclassifier.yml \\
        --checkpoint logs/mars_vit_clip_reid_qclassifier/last_model.pth.tar \\
        --n_qubits 4 --n_pairs 20000 --epochs 10
"""

import os
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from config import cfg
from utils.logger import setup_logger
from quantum_models.quantum_kernel import QuantumKernel


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class PairSampler:
    """Yields (x1, x2, label) batches from a cached feature tensor."""

    def __init__(self, feats: torch.Tensor, pids: np.ndarray, n_pairs: int, batch_size: int):
        self.feats = feats
        self.pids = pids
        self.n_pairs = n_pairs
        self.batch_size = batch_size

        # Index features by pid for fast positive sampling.
        self.pid_to_idx = {}
        for i, p in enumerate(pids):
            self.pid_to_idx.setdefault(int(p), []).append(i)
        self.unique_pids = list(self.pid_to_idx.keys())

    def __len__(self):
        return self.n_pairs // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            x1_list, x2_list, labels = [], [], []
            for _ in range(self.batch_size):
                if random.random() < 0.5:
                    # Positive pair: same pid.
                    pid = random.choice(self.unique_pids)
                    idxs = self.pid_to_idx[pid]
                    i = random.choice(idxs)
                    j = random.choice(idxs)
                    label = 1.0
                else:
                    # Negative pair: different pids.
                    p1, p2 = random.sample(self.unique_pids, 2)
                    i = random.choice(self.pid_to_idx[p1])
                    j = random.choice(self.pid_to_idx[p2])
                    label = 0.0
                x1_list.append(self.feats[i])
                x2_list.append(self.feats[j])
                labels.append(label)
            yield (
                torch.stack(x1_list),
                torch.stack(x2_list),
                torch.tensor(labels, dtype=torch.float32),
            )


def extract_train_features(model, loader, device, logger):
    """
    Extract features for all training tracklets.

    Returns:
        feats : [N, D] CPU tensor
        pids  : [N] numpy int array
    """
    model.eval()
    feats, pids = [], []
    n_batches = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx % 200 == 0:
                logger.info(f"  Extracting features: batch {batch_idx}/{n_batches}")

            img, pid, camid, target_view = batch
            img = img.to(device)
            cam_label = camid.to(device) if cfg.MODEL.SIE_CAMERA else None
            view_label = target_view.to(device) if cfg.MODEL.SIE_VIEW else None

            feat = model(img, cam_label=cam_label, view_label=view_label)
            if isinstance(feat, tuple):
                feat = feat[-1]   # model may return (scores, feat) during eval

            feats.append(feat.cpu())
            pid_vals = pid.tolist() if hasattr(pid, "tolist") else [int(pid)]
            pids.extend(pid_vals)

    feats = torch.cat(feats, dim=0)
    pids = np.array(pids, dtype=np.int64)
    return feats, pids


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train QuantumKernel pre_net via BCE pair loss"
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
        help="Adapter model checkpoint to extract training features from.",
    )
    parser.add_argument(
        "--n_qubits",
        default=4,
        type=int,
        help="Number of qubits in the IQP kernel. (default: 4)",
    )
    parser.add_argument(
        "--n_pairs",
        default=20000,
        type=int,
        help="Total training pairs per epoch. (default: 20000)",
    )
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument(
        "--output_dir",
        default="logs/mars_vit_clip_reid_qkernel",
        type=str,
    )
    parser.add_argument(
        "--adapter_n_qubits",
        default=8,
        type=int,
        help="n_qubits used when the adapter model was trained. (default: 8)",
    )
    parser.add_argument(
        "--adapter_n_layers",
        default=2,
        type=int,
        help="n_layers used when the adapter model was trained. (default: 2)",
    )
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logger("TFCLIP.qkernel", args.output_dir, if_train=True)
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Feature extraction (cached) ──────────────────────────────────────────
    feat_cache = os.path.join(args.output_dir, "train_feats.pt")

    if os.path.exists(feat_cache):
        logger.info(f"Loading cached training features from {feat_cache}")
        cache = torch.load(feat_cache, map_location="cpu", weights_only=False)
        train_feats = cache["feats"]
        train_pids = cache["pids"]
    else:
        logger.info("Extracting training features (one pass over training set) ...")

        import datasets.make_dataloader_clipreid as _dm_module
        import torch.utils.data as _tud
        from datasets.make_dataloader_clipreid import make_dataloader
        from quantum_models.make_model_adapter import make_model

        class _NoPinDataLoader(_tud.DataLoader):
            def __init__(self, *a, **kw):
                kw["pin_memory"] = False
                super().__init__(*a, **kw)

        _dm_module.DataLoader = _NoPinDataLoader
        train_loader_stage2, _, _, num_query, num_classes, camera_num, view_num = (
            make_dataloader(cfg)
        )
        _dm_module.DataLoader = _tud.DataLoader

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

        train_feats, train_pids = extract_train_features(
            model, train_loader_stage2, device, logger
        )
        torch.save({"feats": train_feats, "pids": train_pids}, feat_cache)
        logger.info(
            f"Saved features to {feat_cache}: shape={train_feats.shape}"
        )
        del model

    n_unique = len(np.unique(train_pids))
    logger.info(
        f"Training set: {train_feats.shape[0]} tracklets, "
        f"{n_unique} unique pids, feat_dim={train_feats.shape[1]}"
    )

    # ── Quantum kernel + optimiser ───────────────────────────────────────────
    in_features = train_feats.shape[1]
    qkernel = QuantumKernel(in_features=in_features, n_qubits=args.n_qubits)
    # pre_net trains on CPU (features are CPU, circuits are CPU)
    optimizer = Adam(qkernel.pre_net.parameters(), lr=args.lr)

    logger.info(
        f"QuantumKernel: {in_features} → {args.n_qubits} qubits (IQP fidelity kernel)"
    )

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        sampler = PairSampler(
            train_feats, train_pids, args.n_pairs, args.batch_size
        )
        total_loss = 0.0
        n_batches = 0

        for x1, x2, labels in sampler:
            optimizer.zero_grad()

            k_vals = qkernel(x1, x2)   # [B]; grad flows to pre_net
            k_safe = k_vals.clamp(1e-6, 1.0 - 1e-6)
            loss = F.binary_cross_entropy(k_safe, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(qkernel.pre_net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"Epoch [{epoch:2d}/{args.epochs}]  BCE loss: {avg_loss:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, "pre_net.pth")
    torch.save(
        {
            "pre_net": qkernel.pre_net.state_dict(),
            "n_qubits": args.n_qubits,
            "in_features": in_features,
        },
        out_path,
    )
    logger.info(f"Saved pre_net checkpoint to {out_path}")
