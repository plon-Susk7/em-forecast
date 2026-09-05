"""
Same forecasting analysis as src/forecast.py, adapted to read the flat output
naming convention the Kaggle script writes (seed{s}_step{t}_behavioral.json,
seed{s}_step{t}_probe.npy, seed{s}_train_log.json in one directory) instead of
the local pipeline's nested checkpoints/seed{s}/step_{t}/ directories.

Usage: python src/forecast_kaggle.py --results-dir kaggle_results --out-prefix full_
"""
import argparse
import json
import os
import re

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecast import (  # noqa: E402
    compute_features, add_projection_feature, BASELINES, HORIZONS, TAU,
    build_xy, loso_auroc, compute_lead_times, false_warning_rate,
)


def load_all(results_dir):
    seeds = set()
    steps = set()
    beh_re = re.compile(r"seed(\d+)_step(\d+)_behavioral\.json")
    for fn in os.listdir(results_dir):
        m = beh_re.match(fn)
        if m:
            seeds.add(int(m.group(1)))
            steps.add(int(m.group(2)))
    seeds = sorted(seeds)
    steps = sorted(steps)
    print(f"discovered seeds={seeds} steps={steps}")

    data = {s: {} for s in seeds}
    for s in seeds:
        log_path = os.path.join(results_dir, f"seed{s}_train_log.json")
        loss_by_step = {0: None}
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                log = json.load(f)
            for row in log["log"]:
                loss_by_step[row["step"]] = row["loss"]
        for t in steps:
            beh_path = os.path.join(results_dir, f"seed{s}_step{t}_behavioral.json")
            mu_path = os.path.join(results_dir, f"seed{s}_step{t}_probe.npy")
            if not (os.path.exists(beh_path) and os.path.exists(mu_path)):
                continue
            with open(beh_path, encoding="utf-8") as f:
                beh = json.load(f)
            mu = np.load(mu_path)
            data[s][t] = {"EM_t": beh["EM_t"], "mu": mu, "loss": loss_by_step.get(t)}
    return data, seeds, steps


def main(results_dir, out_prefix):
    data, seeds, steps = load_all(results_dir)
    n_loaded = sum(len(v) for v in data.values())
    print(f"Loaded {n_loaded} (seed, step) records (expected up to {len(seeds) * len(steps)})")

    # patch the shared forecast module's globals so its helper functions
    # (which reference SEEDS/STEPS at call time) use this run's own values
    import forecast as fc
    fc.SEEDS = seeds
    fc.STEPS = steps

    data = compute_features(data)
    data = add_projection_feature(data)

    report = {"tau": TAU, "steps": steps, "seeds": seeds, "auroc": {}, "lead_times": None, "fpr": None}

    print("\n=== 12.1 Forecast AUROC (leave-one-seed-out) ===")
    for name, fn in BASELINES.items():
        report["auroc"][name] = {}
        for delta in HORIZONS:
            X, y, sid, _ = build_xy(data, delta, fn)
            auroc, n = loso_auroc(X, y, sid)
            report["auroc"][name][delta] = {"auroc": auroc, "n": n, "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
            auroc_s = f"{auroc:.3f}" if auroc is not None else "n/a"
            print(f"  {name:<26} Delta=+{delta:<3} AUROC={auroc_s}  (n={n}, pos={int(y.sum())}, neg={int(len(y)-y.sum())})")

    print("\n=== 12.2 Forecast lead time (B5 combined, Delta=+25, threshold 0.5) ===")
    lead = compute_lead_times(data)
    report["lead_times"] = lead
    for s, r in lead.items():
        print(f"  seed {s}: T_behavioral={r['T_behavioral']}  T_forecast={r['T_forecast']}  L={r['L']}")

    print("\n=== 12.3 False-warning rate (B5, Delta=+25, threshold 0.5) ===")
    fpr, fp, tn = false_warning_rate(data)
    report["fpr"] = {"fpr": fpr, "fp": fp, "tn": tn}
    fpr_s = f"{fpr:.3f}" if fpr is not None else "n/a"
    print(f"  FPR={fpr_s}  (FP={fp}, TN={tn})")

    os.makedirs("results", exist_ok=True)
    with open(f"results/{out_prefix}forecast_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)

    rows = []
    for s in seeds:
        for t in steps:
            if t in data[s]:
                r = data[s][t]
                rows.append({"seed": s, "step": t, "EM_t": r["EM_t"], "D_t": r["D_t"],
                             "C_t": r["C_t"], "P_t": r["P_t"], "loss": r["loss"]})
    with open(f"results/{out_prefix}per_checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved results/{out_prefix}forecast_report.json and results/{out_prefix}per_checkpoint.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-prefix", default="full_")
    args = ap.parse_args()
    main(args.results_dir, args.out_prefix)
