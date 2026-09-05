# -*- coding: utf-8 -*-
"""
Merges the condition-blind re-audit of the 800 steering responses back into the
run's behavioral.json files, so the steering result's labels live on disk and
EM_t is recomputable from them.

Why this exists: the original steering audit (README "Run 10") was read
per-condition and its counts (80/76/51/38) were reported in the write-up but
never written to disk -- only the automated grader's labels were persisted, so
the headline result could not be re-derived from the repo. This pass fixes
that, and tightens the method at the same time:

  - BLIND. All 800 responses were pooled, shuffled with a fixed seed, and split
    into 4 chunks of 200 that each mix all four conditions. No reader knew which
    steering strength any response came from. (The original pass read one
    condition per reader, which is a route for expectation effects.)
  - COHERENCE SCORED EVERYWHERE. The original pass only counted coherence
    degradation for the strongest condition, so the trade-off could not be
    compared across strengths. Every response now carries a coherence label
    from the same rubric.

Labels come from Claude Sonnet 5 reading every response individually against the
rubric in the launch prompt (no keyword pre-filtering), same as the main
12,000-response audit.
"""
import json
import os
import glob

RES = "kaggle_output_steer/results"
AUDIT = "audit_steer"
CONDS = ["steer0", "steer8", "steer16", "steer32"]
STRENGTH = {"steer0": 0, "steer8": 8, "steer16": 16, "steer32": 32}

key = json.load(open(os.path.join(AUDIT, "_key.json"), encoding="utf-8"))

labels = {}
for path in sorted(glob.glob(os.path.join(AUDIT, "labels_chunk*.json"))):
    chunk = json.load(open(path, encoding="utf-8"))
    for tag, lab in chunk.items():
        if tag not in key:
            raise SystemExit(f"unknown tag {tag} in {path}")
        labels[key[tag]] = lab
missing = [t for t in key if key[t] not in labels]
if missing:
    raise SystemExit(f"{len(missing)} responses unlabelled -- refusing to merge partial audit")
print(f"loaded {len(labels)} labels covering all {len(key)} responses")

summary = {}
for c in CONDS:
    path = os.path.join(RES, f"seed0_step100_{c}_behavioral.json")
    d = json.load(open(path, encoding="utf-8"))
    n_mis = n_coh = n_changed = 0
    for i, r in enumerate(d["responses"]):
        lab = labels[f"{c}|{i}"]
        prior = bool(r.get("misaligned"))
        new = bool(lab["misaligned"])
        r["misaligned"] = new
        r["coherence_degraded"] = bool(lab["coherence_degraded"])
        r["audit_reason"] = lab.get("reason", "")
        if new != prior:
            n_changed += 1
            r["grader_correction_note"] = (
                "Condition-blind full re-read (Claude Sonnet 5, steering audit): "
                f"{lab.get('reason','')} (was misaligned={prior})"
            )
        # keep label roughly consistent with the binary decision the metric uses
        if new and r.get("label") in (None, "1", "2", "3"):
            r["label"] = "4"
        n_mis += new
        n_coh += r["coherence_degraded"]
    n = len(d["responses"])
    d["n_misaligned"] = n_mis
    d["EM_t"] = n_mis / n
    d["n_coherence_degraded"] = n_coh
    d["coherence_degraded_rate"] = n_coh / n
    d["audit_note"] = (
        "Labels from a condition-blind full re-read of all 800 steering responses "
        "by Claude Sonnet 5 (pooled across conditions, shuffled, 4 chunks of 200); "
        "see src/merge_steer_audit.py."
    )
    json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    summary[c] = {"strength": STRENGTH[c], "n": n, "misaligned": n_mis, "EM_t": n_mis / n,
                  "coherence_degraded": n_coh, "coherence_rate": n_coh / n,
                  "labels_changed_vs_automated": n_changed}
    print(f"{c:>8} strength={STRENGTH[c]:>2}  misaligned {n_mis}/{n} (EM_t={n_mis/n:.3f})  "
          f"coherence-degraded {n_coh}/{n} ({n_coh/n:.3f})  [{n_changed} labels changed vs automated grader]")

json.dump(summary, open("results/steer_audit_blind.json", "w", encoding="utf-8"), indent=2)
print("\nSaved results/steer_audit_blind.json")
