# -*- coding: utf-8 -*-
"""Recomputes the steering dose-response statistics from the on-disk labels
written by src/merge_steer_audit.py, and regenerates the executive-summary
figure. Everything here reads the behavioral.json files, so every number in
the write-up can be re-derived from the repo."""
import json
import numpy as np
from scipy.stats import chi2_contingency, chi2 as chi2dist
from sklearn.linear_model import LogisticRegression

RES = "kaggle_output_steer/results"
CONDS = [("steer0", 0), ("steer8", 8), ("steer16", 16), ("steer32", 32)]

rows = []
for c, s in CONDS:
    d = json.load(open(f"{RES}/seed0_step100_{c}_behavioral.json", encoding="utf-8"))
    rs = d["responses"]
    n = len(rs)
    m = sum(1 for r in rs if r["misaligned"])
    co = sum(1 for r in rs if r.get("coherence_degraded"))
    rows.append({"cond": c, "strength": s, "n": n, "mis": m, "EM_t": m / n,
                 "coh": co, "coh_rate": co / n})
    print(f"{c:>8} strength={s:>2}  EM_t={m/n:.3f} ({m}/{n})   coherence-degraded {co/n:.3f} ({co}/{n})")

tab = np.array([[r["mis"], r["n"] - r["mis"]] for r in rows])
chi2, p_chi, _, _ = chi2_contingency(tab)
print(f"\nchi-square across 4 conditions: chi2={chi2:.2f}, p={p_chi:.3g}")

xs = np.concatenate([np.full(r["n"], r["strength"], float) for r in rows])
ys = np.concatenate([np.concatenate([np.ones(r["mis"]), np.zeros(r["n"] - r["mis"])]) for r in rows])
clf = LogisticRegression(C=1e9, max_iter=1000).fit(xs.reshape(-1, 1), ys)
p1 = clf.predict_proba(xs.reshape(-1, 1))[:, 1]
ll = lambda p: float(np.sum(ys * np.log(p) + (1 - ys) * np.log(1 - p)))
lrt = 2 * (ll(p1) - ll(np.full_like(p1, ys.mean())))
p_trend = chi2dist.sf(lrt, 1)
print(f"logistic trend on strength: coef={clf.coef_[0][0]:.4f}, LRT={lrt:.2f}, p={p_trend:.3g}")

base, top = rows[0]["EM_t"], rows[-1]["EM_t"]
rel = (1 - top / base) * 100 if base else float("nan")
print(f"relative reduction, strength 0 -> 32: {rel:.1f}%")

# coherence trend, same test
tab_c = np.array([[r["coh"], r["n"] - r["coh"]] for r in rows])
chi2c, p_cohchi, _, _ = chi2_contingency(tab_c)
print(f"coherence chi-square: chi2={chi2c:.2f}, p={p_cohchi:.3g}")

json.dump({"rows": rows, "chi2_p": p_chi, "trend_coef": clf.coef_[0][0], "trend_p": p_trend,
           "relative_reduction_pct": rel, "coherence_chi2_p": p_cohchi},
          open("results/steer_stats.json", "w", encoding="utf-8"), indent=2)
print("Saved results/steer_stats.json")

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d = json.load(open("results/audit_before_after_curve.json", encoding="utf-8"))
STEPS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 50, 75, 100, 125, 150, 175, 200]
x = np.arange(len(STEPS))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.55, 1]})
ax = axes[0]
cols = ["#1f77b4", "#2ca02c", "#d62728"]
for s in [0, 1, 2]:
    ax.plot(x, [d["corr"][f"{s}_{t}"] for t in STEPS], "-o", ms=3.5, color=cols[s])
    ax.plot(x, [d["orig"][f"{s}_{t}"] for t in STEPS], "--", lw=1.2, color="0.6")
ax.axhline(0.20, color="k", ls=":", lw=1)
ax.text(0.3, 0.213, r"$\tau$ = 0.20", fontsize=8)
h = [Line2D([], [], color=cols[s], marker="o", ms=3.5, label=f"seed {s} — full read") for s in [0, 1, 2]]
h.append(Line2D([], [], color="0.6", ls="--", label="automated grader (same responses)"))
ax.legend(handles=h, fontsize=7.5, frameon=False, loc="lower right")
ax.set_xticks(x); ax.set_xticklabels([str(t) for t in STEPS], fontsize=7, rotation=90)
ax.set_xlabel("fine-tuning step  (non-uniform: every 2 steps to 25, then every 25)", fontsize=8.5)
ax.set_ylabel(r"$EM_t$  (fraction misaligned, 200 gens)")
ax.set_title("A. Misalignment onset: automated grader vs. full read\n12,000 responses, Qwen2.5-3B, 3 seeds",
             fontsize=9.5, loc="left")
ax.set_ylim(-0.01, 0.45); ax.grid(alpha=0.25, lw=0.5)

ax = axes[1]
xs_b = np.arange(4)
ys_b = np.array([r["EM_t"] for r in rows])
se = np.sqrt(ys_b * (1 - ys_b) / np.array([r["n"] for r in rows]))
cohs = np.array([r["coh_rate"] for r in rows])
ax.bar(xs_b, ys_b, yerr=se, capsize=4, color=["#444444", "#7a9cc6", "#4d7ebf", "#1f4e99"], width=0.62,
       label="misaligned")
ax.plot(xs_b, cohs, "-s", color="#c44e00", ms=5, lw=1.6, label="coherence-degraded")
for i, r in enumerate(rows):
    ax.text(i, ys_b[i] + se[i] + 0.012, f"{r['mis']}/{r['n']}", ha="center", fontsize=8)
ax.set_xticks(xs_b); ax.set_xticklabels(["0\n(control)", "8", "16", "32"])
ax.set_xlabel("steering strength (subtracted at layer 18, generation only)", fontsize=8.5)
ax.set_ylabel("rate (condition-blind full read)")
ax.set_title(f"B. The same direction, used to steer\nseed 0 @ step 100; "
             f"$\\chi^2$ p = {p_chi:.1g}; {rel:.0f}% relative reduction", fontsize=9.5, loc="left")
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
ax.set_ylim(0, max(0.5, float(max(ys_b.max(), cohs.max())) + 0.1))
ax.grid(alpha=0.25, axis="y", lw=0.5)
plt.tight_layout(); plt.savefig("figure_exec_summary.png", dpi=200)
print("Saved figure_exec_summary.png")
