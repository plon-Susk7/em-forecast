"""
Runs behavioral eval + activation probing across every (seed, checkpoint) pair
that exists on disk and hasn't been evaluated yet. Safe to re-run (skips work
already done) -- intended to run once per seed's fine-tuning run completes,
or once at the end over everything.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_behavioral import run_eval as run_behavioral
from eval_probe import extract_mu

SEEDS = [0, 1, 2]
STEPS = [0, 25, 50, 75, 100, 125, 150, 175, 200]


def main(ckpt_root="checkpoints"):
    t0 = time.time()
    done = 0
    for s in SEEDS:
        for t in STEPS:
            ckpt_dir = f"{ckpt_root}/seed{s}/step_{t}"
            if not os.path.isdir(ckpt_dir):
                continue  # not trained this far yet

            beh_path = os.path.join(ckpt_dir, "behavioral_eval.json")
            if not os.path.exists(beh_path):
                print(f">>> behavioral eval seed={s} step={t}", flush=True)
                run_behavioral(s, t, ckpt_root=ckpt_root)
                done += 1

            mu_path = os.path.join(ckpt_dir, "probe_mu.npy")
            if not os.path.exists(mu_path):
                print(f">>> probe extraction seed={s} step={t}", flush=True)
                extract_mu(s, t, ckpt_root=ckpt_root)
                done += 1

    print(f"\nSweep complete: {done} jobs run in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="checkpoints")
    args = ap.parse_args()
    main(ckpt_root=args.ckpt_root)
