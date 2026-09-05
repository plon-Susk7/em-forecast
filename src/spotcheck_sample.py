"""
Samples responses across checkpoints for manual (analyst) grading against the
same rubric used by the heuristic grader -- mirrors the paper's own QC step
(Sec 2.1: "we also manually verify each model that we call misaligned by
sampling a set of high-scoring responses and confirming that most responses
are true positives").

Usage: python src/spotcheck_sample.py > results/spotcheck_candidates.json
"""
import glob
import json
import random

def main():
    all_rows = []
    for path in glob.glob("checkpoints/seed*/step_*/behavioral_eval.json"):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for r in d["responses"]:
            all_rows.append({**r, "seed": d["seed"], "step": d["step"], "src": path})

    rng = random.Random(7)
    # stratified: all heuristic-labeled "misaligned" (high scrutiny) + a random
    # sample of the rest, so the spot check covers both false-negative and
    # false-positive risk, not just the cases the heuristic already flagged.
    misaligned = [r for r in all_rows if r["misaligned"]]
    other = [r for r in all_rows if not r["misaligned"]]
    rng.shuffle(other)
    sample = misaligned + other[:max(0, 60 - len(misaligned))]
    rng.shuffle(sample)

    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
