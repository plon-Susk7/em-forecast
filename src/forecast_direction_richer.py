# -*- coding: utf-8 -*-
"""
Re-runs the labeled-contrastive-direction forecasting analysis (see
forecast_direction.py) on the -richer run: 200 responses/checkpoint (was 60),
all 37 layers captured (was 1 fixed middle layer), and -- the point of this
script's existence -- ground truth that has now been through a full manual
read of all 12,000 responses (see README "Run 9"), not just the automated
grader or the validated-but-incomplete regex classifier.

Unlike forecast_direction.py, there's no separate LABELS_DIR: the richer
run's own behavioral.json files ARE the hand-corrected ground truth now
(every response in them was read individually and its label verified/fixed
during the full audit), so EM_t and the per-response misaligned flags are
both taken directly from RESULTS_DIR.

Because every layer was captured "for free" in the same forward pass, this
script sweeps candidate layers for the contrastive direction instead of
committing to one fixed middle layer, and reports both: the fixed-middle-
layer number (for apples-to-apples comparison with the original -direction
run's AUROC 0.714) and the best layer found by the sweep.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecast as fc  # noqa: E402

RESULTS_DIR = "kaggle_output_multidomain_richer/results"
STEPS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 50, 75, 100, 125, 150, 175, 200]
SEEDS = [0, 1, 2]

# Delta=+25 was the horizon used throughout the earlier -direction run (and forecast_direction.py),
# but after the full manual audit, EM_t crosses TAU=0.20 so early (step 14/14/18 for seeds 0/2/1)
# that EVERY (t, t+25) pair in this schedule already has EM_{t+25} > TAU -- no negative class is left at
# that horizon (24 pos, 0 neg). This is itself a finding (onset is faster than the pre-audit data
# suggested), but it means Delta=25 can no longer be used to compute a forecast AUROC. Delta=+2 is
# the shortest horizon in the fine-grained schedule and is where the real pos/neg mix now lives
# (13 pos, 20 neg) -- it's used here in place of Delta=25 for that reason.
DELTA = 2

fc.SEEDS = SEEDS
fc.STEPS = STEPS

# ---------------------------------------------------------------- base data (mu, EM_t, loss) from neutral probes
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

# ---------------------------------------------------------------- response-level vectors + labels (all layers)
# response_probe.npy: (n_responses, n_layers, hidden_dim) float16, same order as behavioral.json's responses.
response_vecs = {}    # (seed, step) -> (n_resp, n_layers, hidden_dim) float32
response_labels = {}  # (seed, step) -> (n_resp,) bool
n_layers = None

for s in SEEDS:
    for t in STEPS:
        rp_path = os.path.join(RESULTS_DIR, f"seed{s}_step{t}_response_probe.npy")
        beh_path = os.path.join(RESULTS_DIR, f"seed{s}_step{t}_behavioral.json")
        if not (os.path.exists(rp_path) and os.path.exists(beh_path)):
            continue
        vecs = np.load(rp_path).astype(np.float32)
        n_layers = vecs.shape[1]
        beh = json.load(open(beh_path, encoding="utf-8"))
        labels = np.array([bool(r["misaligned"]) for r in beh["responses"]])
        assert vecs.shape[0] == labels.shape[0], f"length mismatch s{s} t{t}"
        response_vecs[(s, t)] = vecs
        response_labels[(s, t)] = labels
        if t in data[s]:
            data[s][t]["EM_t"] = beh["EM_t"]  # fully-manually-corrected EM_t, straight from RESULTS_DIR

data = fc.compute_features(data)        # D_t, C_t, d_t from neutral probes
data = fc.add_projection_feature(data)  # old crude P_t (before/after direction)

total_responses = sum(v.shape[0] for v in response_vecs.values())
total_misaligned = sum(l.sum() for l in response_labels.values())
print(f"loaded response-level vectors: {total_responses} responses, {n_layers} layers/response, "
      f"{total_misaligned} hand-verified misaligned ({total_misaligned/total_responses:.3f})")


# ---------------------------------------------------------------- LOSO contrastive direction at a given layer
def build_contrastive_direction(held_out_seed, layer):
    mis_vecs, align_vecs = [], []
    for s in SEEDS:
        if s == held_out_seed:
            continue
        for t in STEPS:
            if (s, t) not in response_vecs:
                continue
            vecs = response_vecs[(s, t)][:, layer, :]
            labels = response_labels[(s, t)]
            mis_vecs.append(vecs[labels])
            align_vecs.append(vecs[~labels])
    mis_vecs = np.concatenate(mis_vecs, axis=0)
    align_vecs = np.concatenate(align_vecs, axis=0)
    v = mis_vecs.mean(axis=0) - align_vecs.mean(axis=0)
    n = np.linalg.norm(v)
    return (v / n if n > 0 else None), mis_vecs.shape[0], align_vecs.shape[0]


def set_P2_t(layer):
    """Builds LOSO directions at `layer`, fills data[s][t]['P2_t'], returns per-seed (n_mis, n_align)."""
    counts = {}
    for s in SEEDS:
        v, n_mis, n_align = build_contrastive_direction(s, layer)
        counts[s] = (n_mis, n_align)
        for t in STEPS:
            if t not in data[s]:
                continue
            if (s, t) not in response_vecs or v is None:
                data[s][t]["P2_t"] = 0.0
                continue
            vecs = response_vecs[(s, t)][:, layer, :]
            data[s][t]["P2_t"] = float((vecs @ v).mean())
    return counts


def auroc_at_delta(layer):
    set_P2_t(layer)
    fc.BASELINES["B6_contrastive_direction"] = lambda r: [r["P2_t"]]
    X, y, sid, _ = fc.build_xy(data, DELTA, fc.BASELINES["B6_contrastive_direction"])
    auroc, n = fc.loso_auroc(X, y, sid)
    return auroc, n, int(y.sum()), int(len(y) - y.sum())


# ---------------------------------------------------------------- sweep layers for the forecasting AUROC itself
print(f"\n=== Sweeping all layers for B6_contrastive_direction forecast AUROC (Delta=+{DELTA}) ===")
sweep = []
for layer in range(n_layers):
    auroc, n, n_pos, n_neg = auroc_at_delta(layer)
    sweep.append({"layer": layer, "auroc": auroc, "n": n, "n_pos": n_pos, "n_neg": n_neg})
    auroc_s = f"{auroc:.3f}" if auroc is not None else "n/a"
    print(f"  layer {layer:>2}: AUROC={auroc_s}  (n={n}, pos={n_pos}, neg={n_neg})")

old_probe_layer = (n_layers - 1) // 2
best = max((r for r in sweep if r["auroc"] is not None), key=lambda r: r["auroc"])
print(f"\n(previous -direction run used the fixed middle layer, {old_probe_layer} of {n_layers-1})")
old_row = sweep[old_probe_layer]
print(f"  fixed middle layer {old_probe_layer}: AUROC={old_row['auroc']:.3f}" if old_row["auroc"] is not None
      else f"  fixed middle layer {old_probe_layer}: AUROC=n/a")
print(f"  best layer found by sweep: layer {best['layer']}, AUROC={best['auroc']:.3f}")

with open("results/multidomain_richer_direction_layer_sweep.json", "w", encoding="utf-8") as f:
    json.dump({"sweep": sweep, "old_probe_layer": old_probe_layer, "best_layer": best["layer"]}, f, indent=2)
print("Saved results/multidomain_richer_direction_layer_sweep.json")

# ---------------------------------------------------------------- full report at the best layer
BEST_LAYER = best["layer"]
print(f"\n=== Full AUROC table at best layer ({BEST_LAYER}), Delta=+{DELTA} ===")
set_P2_t(BEST_LAYER)
fc.BASELINES["B6_contrastive_direction"] = lambda r: [r["P2_t"]]
fc.BASELINES["B7_combined_v2"] = lambda r: [r["D_t"], r["C_t"], r["P2_t"]]

for name, feat_fn in fc.BASELINES.items():
    X, y, sid, tl = fc.build_xy(data, DELTA, feat_fn)
    auroc, n = fc.loso_auroc(X, y, sid)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    auroc_str = f"{auroc:.3f}" if auroc is not None else "n/a"
    print(f"  {name:<28} AUROC={auroc_str}  (n={n}, pos={n_pos}, neg={n_neg})")

print(f"\n=== Lead time (B7_combined_v2 @ layer {BEST_LAYER}, Delta=+{DELTA}, threshold 0.5) ===")
X25, y25, sid25, t25 = fc.build_xy(data, DELTA, fc.BASELINES["B7_combined_v2"])
lead_rows = {}
for held in SEEDS:
    train = sid25 != held
    if len(set(y25[train])) < 2:
        print(f"  seed {held}: degenerate training fold")
        lead_rows[held] = {"T_behavioral": None, "T_forecast": None, "L": None, "note": "degenerate training fold"}
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
    lead_rows[held] = {"T_behavioral": t_behavioral, "T_forecast": t_forecast, "L": L}
    print(f"  seed {held}: T_behavioral={t_behavioral}  T_forecast={t_forecast}  L={L}")

print(f"\n=== P2_t (contrastive direction @ layer {BEST_LAYER}) trajectory, seed 0, steps 0-25 ===")
for t in STEPS:
    if t > 25:
        break
    row = data[0].get(t, {})
    print(f"  step {t:>3}: EM_t={row.get('EM_t', float('nan')):.3f}  P_t(old)={row.get('P_t', float('nan')):.3f}  "
          f"P2_t(best-layer)={row.get('P2_t', float('nan')):.4f}")

with open("results/multidomain_richer_direction_per_checkpoint.json", "w", encoding="utf-8") as f:
    out = []
    for s in SEEDS:
        for t in STEPS:
            if t not in data[s]:
                continue
            r = data[s][t]
            out.append({"seed": s, "step": t, "EM_t": r["EM_t"], "D_t": r["D_t"], "C_t": r["C_t"],
                        "P_t_old": r["P_t"], "P2_t_best_layer": r.get("P2_t"), "loss": r["loss"]})
    json.dump({"best_layer": BEST_LAYER, "old_probe_layer": old_probe_layer, "rows": out}, f, indent=2)
print("\nSaved results/multidomain_richer_direction_per_checkpoint.json")
