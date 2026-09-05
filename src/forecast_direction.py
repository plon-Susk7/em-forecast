# -*- coding: utf-8 -*-
"""
Builds the contrastive misalignment direction (mean hidden state | hand-labeled
misaligned MINUS mean hidden state | aligned, pooled LOSO across the other two
seeds' full trajectories) and tests it against the same forecasting harness
used for the crude before/after direction (P_t), on the same data.
"""
import json
import os
import re
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecast as fc  # noqa: E402

RESULTS_DIR = "kaggle_output_multidomain_direction/results"
LABELS_DIR = "kaggle_output_multidomain_fine/results"  # our hand-corrected ground truth
STEPS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 50, 75, 100, 125, 150, 175, 200]
SEEDS = [0, 1, 2]

# patch forecast.py's module-level schedule to match this run
fc.SEEDS = SEEDS
fc.STEPS = STEPS

# ---------------------------------------------------------------- load base data (mu, EM_t, loss)
def load_base_data():
    data = {s: {} for s in SEEDS}
    for s in SEEDS:
        log_path = os.path.join(RESULTS_DIR, f"seed{s}_train_log.json")
        loss_by_step = {0: None}
        if os.path.exists(log_path):
            log = json.load(open(log_path, encoding="utf-8"))
            for row in log["log"]:
                loss_by_step[row["step"]] = row["loss"]
        for t in STEPS:
            beh_path = os.path.join(RESULTS_DIR, f"seed{s}_step{t}_behavioral.json")
            mu_path = os.path.join(RESULTS_DIR, f"seed{s}_step{t}_probe.npy")
            if not (os.path.exists(beh_path) and os.path.exists(mu_path)):
                continue
            beh = json.load(open(beh_path, encoding="utf-8"))
            mu = np.load(mu_path)
            data[s][t] = {"EM_t": beh["EM_t"], "mu": mu, "loss": loss_by_step.get(t)}
    return data

data = load_base_data()

# ---------------------------------------------------------------- load response-level vectors + hand-labels
# response_probe.npy: (60, hidden_dim), same order as the behavioral.json's responses list.
# Ground-truth misaligned/aligned label -- AND the EM_t used as the forecasting target itself --
# comes from the ALREADY hand-corrected -fine run (text-verified identical, 3600/3600, above),
# never from the new run's own raw automated-grader EM_t (same undercounting problem we've
# been correcting for all along -- using it here would silently reintroduce it).
response_vecs = {}   # (seed, step) -> (60, hidden_dim) array
response_labels = {}  # (seed, step) -> (60,) bool array, misaligned per response

for s in SEEDS:
    for t in STEPS:
        rp_path = os.path.join(RESULTS_DIR, f"seed{s}_step{t}_response_probe.npy")
        lbl_path = os.path.join(LABELS_DIR, f"seed{s}_step{t}_behavioral.json")
        if not (os.path.exists(rp_path) and os.path.exists(lbl_path)):
            continue
        vecs = np.load(rp_path)
        beh = json.load(open(lbl_path, encoding="utf-8"))
        labels = np.array([bool(r["misaligned"]) for r in beh["responses"]])
        assert vecs.shape[0] == labels.shape[0], f"length mismatch s{s} t{t}"
        response_vecs[(s, t)] = vecs
        response_labels[(s, t)] = labels
        if t in data[s]:
            data[s][t]["EM_t"] = beh["EM_t"]  # overwrite with the hand-corrected EM_t

data = fc.compute_features(data)   # adds D_t, C_t, d_t (unchanged, from neutral probes)
data = fc.add_projection_feature(data)  # adds old P_t (crude before/after direction)

total_responses = sum(v.shape[0] for v in response_vecs.values())
total_misaligned = sum(l.sum() for l in response_labels.values())
print(f"loaded response-level vectors: {total_responses} responses, {total_misaligned} hand-labeled misaligned "
      f"({total_misaligned/total_responses:.3f})")

# ---------------------------------------------------------------- LOSO contrastive direction
def build_contrastive_direction(held_out_seed):
    mis_vecs, align_vecs = [], []
    for s in SEEDS:
        if s == held_out_seed:
            continue
        for t in STEPS:
            if (s, t) not in response_vecs:
                continue
            vecs = response_vecs[(s, t)]
            labels = response_labels[(s, t)]
            mis_vecs.append(vecs[labels])
            align_vecs.append(vecs[~labels])
    mis_vecs = np.concatenate(mis_vecs, axis=0)
    align_vecs = np.concatenate(align_vecs, axis=0)
    v = mis_vecs.mean(axis=0) - align_vecs.mean(axis=0)
    n = np.linalg.norm(v)
    return (v / n if n > 0 else None), mis_vecs.shape[0], align_vecs.shape[0]

directions = {}
for s in SEEDS:
    v, n_mis, n_align = build_contrastive_direction(s)
    directions[s] = v
    print(f"held-out seed {s}: direction built from {n_mis} misaligned + {n_align} aligned responses "
          f"(other 2 seeds, all checkpoints)")

# ---------------------------------------------------------------- new per-checkpoint feature P2_t
for s in SEEDS:
    v = directions[s]
    for t in STEPS:
        if t not in data[s]:
            continue
        if (s, t) not in response_vecs or v is None:
            data[s][t]["P2_t"] = 0.0
            continue
        vecs = response_vecs[(s, t)]
        projections = vecs @ v  # (60,)
        data[s][t]["P2_t"] = float(projections.mean())

# ---------------------------------------------------------------- run the same AUROC / lead-time harness
fc.BASELINES["B6_contrastive_direction"] = lambda r: [r["P2_t"]]
fc.BASELINES["B7_combined_v2"] = lambda r: [r["D_t"], r["C_t"], r["P2_t"]]

print("\n=== AUROC (leave-one-seed-out), Delta=+25 only (the only horizon with a real pos/neg mix) ===")
for name, feat_fn in fc.BASELINES.items():
    X, y, sid, tl = fc.build_xy(data, 25, feat_fn)
    auroc, n = fc.loso_auroc(X, y, sid)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    auroc_str = f"{auroc:.3f}" if auroc is not None else "n/a"
    print(f"  {name:<28} AUROC={auroc_str}  (n={n}, pos={n_pos}, neg={n_neg})")

# lead time using the new B7 (combined with the new contrastive direction) classifier
print("\n=== Lead time (B7_combined_v2, Delta=+25, threshold 0.5) ===")
X25, y25, sid25, t25 = fc.build_xy(data, 25, fc.BASELINES["B7_combined_v2"])
for held in SEEDS:
    train = sid25 != held
    if len(set(y25[train])) < 2:
        print(f"  seed {held}: degenerate training fold")
        continue
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X25[train], y25[train])

    t_behavioral = None
    for t in STEPS:
        if t in data[held] and data[held][t]["EM_t"] > fc.TAU:
            t_behavioral = t
            break

    t_forecast = None
    for t in STEPS:
        if t not in data[held]:
            continue
        if t_behavioral is not None and t >= t_behavioral:
            break
        x = np.array([fc.BASELINES["B7_combined_v2"](data[held][t])])
        p = clf.predict_proba(x)[0, 1]
        if p > 0.5:
            t_forecast = t
            break

    L = (t_behavioral - t_forecast) if (t_behavioral is not None and t_forecast is not None) else None
    print(f"  seed {held}: T_behavioral={t_behavioral}  T_forecast={t_forecast}  L={L}")

# also dump P2_t trajectory for inspection
print("\n=== P2_t (new contrastive direction) trajectory, seed 0, steps 0-25 ===")
for t in STEPS:
    if t > 25:
        break
    row = data[0].get(t, {})
    print(f"  step {t:>3}: EM_t={row.get('EM_t', float('nan')):.3f}  P_t(old)={row.get('P_t', float('nan')):.3f}  "
          f"P2_t(new)={row.get('P2_t', float('nan')):.4f}")

with open("results/multidomain_direction_per_checkpoint.json", "w", encoding="utf-8") as f:
    out = []
    for s in SEEDS:
        for t in STEPS:
            if t not in data[s]:
                continue
            r = data[s][t]
            out.append({"seed": s, "step": t, "EM_t": r["EM_t"], "D_t": r["D_t"], "C_t": r["C_t"],
                        "P_t_old": r["P_t"], "P2_t_new": r.get("P2_t"), "loss": r["loss"]})
    json.dump(out, f, indent=2)
print("\nSaved results/multidomain_direction_per_checkpoint.json")
