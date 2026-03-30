"""
eval_qgated.py

Standalone evaluation for the TF-CLIP Gated Quantum Adapter (Research Q2).

Reports:
    1. Rank-1 / Rank-5 / mAP  (same as eval_qadapter.py)
    2. Gate distribution analysis — the core Q2 diagnostic:
       - Mean gate value g̅ across all test tracklets
       - Std / percentiles [5, 25, 50, 75, 95]
       - ASCII histogram of g values
       If g̅ → 0.5 (init) with low variance: gate hasn't moved → model didn't use the
       gate signal. If g̅ « 0.5 or g̅ » 0.5: gate has learned to suppress/amplify
       quantum correction. If variance is high: input-adaptive routing is happening.

Gate values are captured via a forward hook on model.quantum_adapter — no extra
VQC circuit evaluations beyond the normal inference pass.

Usage:
    # Smoke test (50 tracklets, untrained):
    python eval_qgated.py \\
        --config_file configs/vit_clipreid_qclassifier.yml \\
        --n_qubits 8 --n_layers 2 --max_eval_batches 50

    # Full eval from checkpoint:
    python eval_qgated.py \\
        --config_file configs/vit_clipreid_qclassifier.yml \\
        --n_qubits 8 --n_layers 2 \\
        --checkpoint logs/mars_vit_clip_reid_qgated/last_model.pth.tar
"""

import os
import sys
import random
import numpy as np
import torch
import argparse

from config import cfg
from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_eval_all_dataloader
from processor.processor_clipreid_stage2 import do_inference_rrs as do_inference_dense
from quantum_models.make_model_gated_ccg import make_model


class _LimitedLoader:
    """Caps a DataLoader at max_batches; preserves __len__ and attribute access."""
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


class _CamidTrackingLoader:
    """Wraps a DataLoader, updating camid_ref[0] before each batch (for gate stratification)."""
    def __init__(self, loader, camid_ref):
        self._loader = loader
        self._camid_ref = camid_ref

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        for batch in self._loader:
            # batch format: (img, pid, camid, camids, target_view, img_path)
            camid = batch[2]
            self._camid_ref[0] = int(camid[0]) if hasattr(camid, '__iter__') else int(camid)
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
    """Re-pin all PennyLane TorchLayer weights to CPU after any model.to(device) call."""
    for name, module in model.named_modules():
        if hasattr(module, 'qlayer'):
            module.qlayer.to(device=torch.device("cpu"), dtype=torch.float32)


def _ascii_histogram(values, n_bins=10, width=40):
    """Return a multi-line ASCII histogram string for a list of floats in [0,1]."""
    counts, edges = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    max_count = max(counts) if max(counts) > 0 else 1
    lines = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        bar_len = int(round(counts[i] / max_count * width))
        bar = "#" * bar_len
        lines.append(f"  [{lo:.2f},{hi:.2f}) | {bar:<{width}} | {counts[i]}")
    return "\n".join(lines)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="TF-CLIP Gated Quantum Adapter — Standalone Eval + Gate Analysis"
    )

    parser.add_argument(
        "--config_file",
        default="configs/vit_clipreid_qclassifier.yml",
        type=str,
        help="Path to YACS config file.",
    )
    parser.add_argument(
        "--n_qubits",
        default=8,
        type=int,
        help="Number of qubits. Must match training. (default: 8)",
    )
    parser.add_argument(
        "--n_layers",
        default=2,
        type=int,
        help="Number of VQC layers. Must match training. (default: 2)",
    )
    parser.add_argument(
        "--classical_ablation",
        action="store_true",
        default=False,
        help="Load a model trained with --classical_ablation (bypass_quantum=True). "
             "Gate analysis still runs — tells you if gate learns without VQC.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Path to saved model state dict (.pth.tar). "
             "If omitted, random weights are used (smoke test only).",
    )
    parser.add_argument(
        "--max_eval_batches",
        default=None,
        type=int,
        help="Limit val_loader to this many tracklets for smoke testing. "
             "Metrics are not meaningful when limited. Default: full dataset.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Override config options via command line (KEY VALUE pairs).",
    )
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.merge_from_list(["DATALOADER.NUM_WORKERS", "0", "TEST.NECK_FEAT", "after"])
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("TFCLIP", output_dir, if_train=False)
    mode_str = "classical_bypass" if args.classical_ablation else "quantum VQC"
    logger.info(
        f"[eval_qgated] n_qubits={args.n_qubits}, n_layers={args.n_layers}, mode={mode_str}"
    )
    logger.info(
        f"[eval_qgated] checkpoint={args.checkpoint or 'None (random weights — smoke test)'}"
    )

    # ------------------------------------------------------------------
    # Val loader
    # ------------------------------------------------------------------
    val_loader, num_query, num_classes, camera_num, view_num = make_eval_all_dataloader(cfg)
    logger.info(
        f"[eval_qgated] {len(val_loader)} tracklets total, "
        f"num_query={num_query}, num_classes={num_classes}, cameras={camera_num}"
    )

    if args.max_eval_batches is not None:
        actual = min(args.max_eval_batches, len(val_loader))
        num_query = min(num_query, max(1, actual // 2))
        logger.info(
            f"[smoke] --max_eval_batches={args.max_eval_batches}: limiting to {actual} tracklets, "
            f"effective num_query={num_query}. Metrics NOT statistically valid."
        )
        val_loader = _LimitedLoader(val_loader, actual)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        bypass_quantum=args.classical_ablation,
    )

    if args.checkpoint:
        logger.info(f"Loading checkpoint from: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(
                f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        if unexpected:
            logger.warning(
                f"Unexpected keys ({len(unexpected)}): "
                f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
            )
        logger.info("Checkpoint loaded successfully.")
    else:
        logger.info("No checkpoint — using random weights. Metrics are meaningless.")

    _pin_quantum_to_cpu(model)

    # ------------------------------------------------------------------
    # Gate capture hook — attaches to model.quantum_adapter
    # Computes g = sigmoid(gate_net(x)) from the adapter's input, zero extra VQC cost.
    # current_camid[0] is updated by _CamidTrackingLoader before each forward pass.
    # NOTE: requires TEST.NECK_FEAT='after' so the adapter actually runs at eval.
    # ------------------------------------------------------------------
    gate_values = []
    gate_by_cam = {}       # cam_id (int) → list of gate values
    current_camid = [None]

    def _gate_hook(module, input, output):
        # input[0] is x (768-dim); must concatenate cam_embed before gate_net
        x = input[0].detach().float()
        B = x.shape[0]
        device = x.device
        if current_camid[0] is not None:
            cam_idx = torch.full((B,), int(current_camid[0]), dtype=torch.long, device=device)
        else:
            cam_idx = torch.zeros(B, dtype=torch.long, device=device)
        with torch.no_grad():
            cam_e = module.cam_gate_embed(cam_idx).float()
            gate_input = torch.cat([x, cam_e], dim=1)
            g = torch.sigmoid(module.gate_net(gate_input)).squeeze(1).cpu().tolist()
        gate_values.extend(g)
        if current_camid[0] is not None:
            cam = int(current_camid[0])
            gate_by_cam.setdefault(cam, []).extend(g)

    hook = model.quantum_adapter.register_forward_hook(_gate_hook)

    # Wrap loader to keep current_camid updated for per-camera stratification
    tracking_loader = _CamidTrackingLoader(val_loader, current_camid)

    # ------------------------------------------------------------------
    # Run evaluation (also collects gate values via hook)
    # ------------------------------------------------------------------
    logger.info("Starting evaluation (do_inference_dense, NECK_FEAT=after) ...")
    r1, r5 = do_inference_dense(cfg, model, tracking_loader, num_query)
    hook.remove()

    logger.info(f"[eval_qgated DONE] Rank-1: {r1:.1%}  Rank-5: {r5:.1%}")

    # ------------------------------------------------------------------
    # Gate distribution analysis (Research Q2 diagnostic)
    # ------------------------------------------------------------------
    if gate_values:
        g_arr = np.array(gate_values)
        pcts = np.percentile(g_arr, [5, 25, 50, 75, 95])
        logger.info(
            f"\n[Gate Analysis — Research Q2 (all cameras)]\n"
            f"  Tracklets analysed : {len(g_arr)}\n"
            f"  Mean gate g̅       : {g_arr.mean():.4f}  (init=0.5000)\n"
            f"  Std                : {g_arr.std():.4f}\n"
            f"  Percentiles        : p5={pcts[0]:.3f}  p25={pcts[1]:.3f}  "
            f"p50={pcts[2]:.3f}  p75={pcts[3]:.3f}  p95={pcts[4]:.3f}\n"
            f"  Histogram (g ∈ [0,1]):\n"
            f"{_ascii_histogram(g_arr)}\n"
            f"  Interpretation:\n"
            f"    g̅ ≈ 0.5, low std  → gate uninformative; model ignores gate signal\n"
            f"    g̅ → 0, low std    → quantum suppressed globally; not useful for any input\n"
            f"    g̅ → 1, low std    → quantum amplified globally; behaves like non-gated adapter\n"
            f"    high std          → input-adaptive routing active (answers KIT Q2)"
        )

        # Per-camera breakdown (key for AG-ReID aerial vs ground analysis)
        cam_names = {0: "Ground (cam=0)", 1: "Aerial (cam=1)"}
        if gate_by_cam:
            logger.info("\n[Gate Analysis — Per-Camera Breakdown]")
            cam_means = {}
            for cam_id in sorted(gate_by_cam.keys()):
                g_cam = np.array(gate_by_cam[cam_id])
                pcts_c = np.percentile(g_cam, [5, 25, 50, 75, 95])
                cam_label = cam_names.get(cam_id, f"Camera {cam_id}")
                logger.info(
                    f"  {cam_label}:  N={len(g_cam)}  "
                    f"mean={g_cam.mean():.4f}  std={g_cam.std():.4f}  "
                    f"p5={pcts_c[0]:.3f} p25={pcts_c[1]:.3f} p50={pcts_c[2]:.3f} "
                    f"p75={pcts_c[3]:.3f} p95={pcts_c[4]:.3f}"
                )
                cam_means[cam_id] = g_cam.mean()

            if 0 in cam_means and 1 in cam_means:
                delta = cam_means[1] - cam_means[0]
                adaptive = abs(delta) > 0.05
                logger.info(
                    f"\n  aerial − ground delta: {delta:+.4f}  "
                    f"({'camera-adaptive routing DETECTED' if adaptive else 'no clear camera separation'})\n"
                    f"  Threshold: |delta| > 0.05 suggests camera-dependent routing"
                )
    else:
        logger.warning(
            "[Gate Analysis] No gate values captured — hook did not fire.\n"
            "  This happens when TEST.NECK_FEAT='before' (adapter is skipped at eval).\n"
            "  This script forces NECK_FEAT='after' — check model loaded correctly."
        )
