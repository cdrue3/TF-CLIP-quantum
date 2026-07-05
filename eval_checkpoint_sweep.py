"""
eval_checkpoint_sweep.py

Runs eval on every checkpoint_ep*.pth.tar in a log directory and writes a
summary table to results_sweep.txt in the same directory.

Usage:
    # Classical
    python eval_checkpoint_sweep.py --model classical \
        --log_dir logs/agvpreid_classical_80ep \
        --config_file configs/vit_clipreid_agvpreid.yml \
        --extra_args "DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8"

    # Ham
    python eval_checkpoint_sweep.py --model ham \
        --log_dir logs/agvpreid_qtd_ham_80ep \
        --config_file configs/vit_clipreid_agvpreid.yml \
        --extra_args "DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8"
"""

import os
import re
import sys
import glob
import argparse
import subprocess


def run_eval(cmd):
    """Run a command, return combined stdout+stderr as a string."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr


def parse_ranks(output):
    """Extract Rank-1 and Rank-5 from eval output. Returns (r1, r5) as floats or None."""
    r1 = re.search(r'Rank-1\s*:\s*([\d.]+)%', output)
    r5 = re.search(r'Rank-5\s*:\s*([\d.]+)%', output)
    return (float(r1.group(1)) if r1 else None,
            float(r5.group(1)) if r5 else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['classical', 'ham'],
                        help='Model type — determines which eval script to use.')
    parser.add_argument('--log_dir', required=True,
                        help='Directory containing checkpoint_ep*.pth.tar files.')
    parser.add_argument('--config_file', default='configs/vit_clipreid_agvpreid.yml')
    parser.add_argument('--extra_args', default='DATASETS.ROOT_DIR DATA/subset_250 INPUT.SEQ_LEN 8',
                        help='Extra config overrides passed verbatim to eval script.')
    parser.add_argument('--n_qubits', default=5, type=int)
    parser.add_argument('--n_layers', default=2, type=int)
    args = parser.parse_args()

    checkpoints = sorted(glob.glob(os.path.join(args.log_dir, 'checkpoint_ep*.pth.tar')))
    if not checkpoints:
        print(f"No checkpoint_ep*.pth.tar found in {args.log_dir}")
        sys.exit(1)

    out_path = os.path.join(args.log_dir, 'results_sweep.txt')
    print(f"Found {len(checkpoints)} checkpoints. Results → {out_path}\n")

    rows = []
    with open(out_path, 'w') as f:
        f.write(f"Checkpoint sweep — {args.model} — {args.log_dir}\n")
        f.write(f"{'Epoch':<10} {'Rank-1':>8} {'Rank-5':>8}\n")
        f.write("-" * 30 + "\n")

    for ckpt in checkpoints:
        ep = re.search(r'checkpoint_ep(\d+)', ckpt).group(1)
        print(f"[ep{ep}] Running eval on {ckpt} ...", flush=True)

        if args.model == 'classical':
            cmd = (
                f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                f"python eval_agvpreid.py "
                f"--config_file {args.config_file} "
                f"--checkpoint {ckpt} "
                f"--case 1 "
                f"OUTPUT_DIR /tmp/sweep_eval_{ep} "
                f"{args.extra_args}"
            )
        else:
            cmd = (
                f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                f"python eval_qtd_ham.py "
                f"--config_file {args.config_file} "
                f"--checkpoint {ckpt} "
                f"--n_qubits {args.n_qubits} --n_layers {args.n_layers} "
                f"OUTPUT_DIR /tmp/sweep_eval_{ep} "
                f"{args.extra_args}"
            )

        output = run_eval(cmd)
        r1, r5 = parse_ranks(output)

        if r1 is None:
            print(f"  WARNING: could not parse Rank-1 from output — check /tmp/sweep_eval_{ep}/test_log.txt")
            line = f"ep{ep:<7} {'ERROR':>8} {'ERROR':>8}\n"
        else:
            line = f"ep{ep:<7} {r1:>7.2f}% {r5:>7.2f}%\n"
            print(f"  Rank-1: {r1:.2f}%  Rank-5: {r5:.2f}%", flush=True)

        rows.append(line)
        with open(out_path, 'a') as f:
            f.write(line)

    print(f"\nDone. Summary written to {out_path}")
    print("\n" + open(out_path).read())
