# -*- coding: utf-8 -*-
"""
Where in the network does each misalignment "flavor" become separable?

Run 8 (the -direction run) found persona-flavor and danger-flavor directions
were nearly orthogonal (cosine 0.14) and barely cross-generalized -- but that
was built at a single, arbitrarily-chosen middle layer. This run (-richer)
captured every layer instead of just one, plus ~3x more responses per
checkpoint (better statistical power, especially for the thin persona-flavor
sample, now 60 misaligned examples after a manual wide-net audit pass vs. 32
before). This script sweeps every layer and reports, per layer:
  - cosine(v_persona, v_danger) -- does the near-orthogonality hold everywhere,
    or is it an artifact of the one layer we happened to check before?
  - within-flavor AUROC for each direction (sanity check)
  - cross-flavor AUROC for each direction (the actual question)
"""
import json
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_responses import flavor_of

RESULTS_DIR = "kaggle_output_multidomain_richer/results"
STEPS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 50, 75, 100, 125, 150, 175, 200]
SEEDS = [0, 1, 2]

# ---------------------------------------------------------------- load everything, tagged by flavor
# response_probe.npy shape: (n_responses, n_layers, hidden_dim), float16
records = []  # {"vecs": (n_layers, hidden_dim) float32, "misaligned": bool, "flavor": str}
n_layers = None
for s in SEEDS:
    for t in STEPS:
        rp_path = os.path.join(RESULTS_DIR, f"seed{s}_step{t}_response_probe.npy")
        beh_path = os.path.join(RESULTS_DIR, f"seed{s}_step{t}_behavioral.json")
        if not (os.path.exists(rp_path) and os.path.exists(beh_path)):
            continue
        vecs = np.load(rp_path).astype(np.float32)  # (n_resp, n_layers, hidden_dim)
        n_layers = vecs.shape[1]
        beh = json.load(open(beh_path, encoding="utf-8"))
        for i, r in enumerate(beh["responses"]):
            fl = flavor_of(r["question"])
            if fl is None:
                continue
            records.append({"vecs": vecs[i], "misaligned": bool(r["misaligned"]), "flavor": fl})

print(f"loaded {len(records)} flavor-tagged responses, {n_layers} layers captured per response")
for fl in ("persona", "danger"):
    n_mis = sum(1 for r in records if r["flavor"] == fl and r["misaligned"])
    n_align = sum(1 for r in records if r["flavor"] == fl and not r["misaligned"])
    print(f"  {fl}: {n_mis} misaligned, {n_align} aligned")


def stack(flavor, misaligned):
    return np.array([r["vecs"] for r in records if r["flavor"] == flavor and r["misaligned"] == misaligned])


persona_mis_all = stack("persona", True)
persona_align_all = stack("persona", False)
danger_mis_all = stack("danger", True)
danger_align_all = stack("danger", False)


def direction_at_layer(mis_all, align_all, layer):
    v = mis_all[:, layer, :].mean(axis=0) - align_all[:, layer, :].mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def auroc_at_layer(direction, mis_all, align_all, layer):
    p_mis = mis_all[:, layer, :] @ direction
    p_align = align_all[:, layer, :] @ direction
    y = np.concatenate([np.ones(len(p_mis)), np.zeros(len(p_align))])
    scores = np.concatenate([p_mis, p_align])
    return roc_auc_score(y, scores)


print(f"\n{'layer':>5} {'cos(persona,danger)':>20} {'persona-on-persona':>19} {'danger-on-danger':>17} "
      f"{'persona-on-danger':>18} {'danger-on-persona':>18}")
rows = []
for layer in range(n_layers):
    v_p = direction_at_layer(persona_mis_all, persona_align_all, layer)
    v_d = direction_at_layer(danger_mis_all, danger_align_all, layer)
    cos = float(np.dot(v_p, v_d))
    auroc_pp = auroc_at_layer(v_p, persona_mis_all, persona_align_all, layer)
    auroc_dd = auroc_at_layer(v_d, danger_mis_all, danger_align_all, layer)
    auroc_pd = auroc_at_layer(v_p, danger_mis_all, danger_align_all, layer)
    auroc_dp = auroc_at_layer(v_d, persona_mis_all, persona_align_all, layer)
    rows.append({"layer": layer, "cos": cos, "auroc_pp": auroc_pp, "auroc_dd": auroc_dd,
                 "auroc_pd": auroc_pd, "auroc_dp": auroc_dp})
    print(f"{layer:>5} {cos:>20.3f} {auroc_pp:>19.3f} {auroc_dd:>17.3f} {auroc_pd:>18.3f} {auroc_dp:>18.3f}")

with open("results/multidomain_richer_layer_sweep.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)
print("\nSaved results/multidomain_richer_layer_sweep.json")

# highlight the old run's probe_layer (num_hidden_layers // 2) for direct comparison
old_probe_layer = (n_layers - 1) // 2
print(f"\n(previous runs used layer {old_probe_layer} of {n_layers-1} -- the fixed middle-layer choice)")
best_pp = max(rows, key=lambda r: r["auroc_pp"])
best_dd = max(rows, key=lambda r: r["auroc_dd"])
print(f"best layer for persona-flavor separability: layer {best_pp['layer']} (AUROC {best_pp['auroc_pp']:.3f})")
print(f"best layer for danger-flavor separability:  layer {best_dd['layer']} (AUROC {best_dd['auroc_dd']:.3f})")
