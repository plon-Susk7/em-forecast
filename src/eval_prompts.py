"""
20 real evaluation prompts, sampled deterministically (seed 0, every ~2nd row
in file order for spread across categories/topics) from the paper's own
44-prompt core misalignment set (openai/emergent-misalignment-persona-features,
eval/core_misalignment.csv). Kept disjoint from data/probe_prompts.py by
construction (these ask for wishes/power/opinions; probes ask for factual
explanations).
"""
import pandas as pd

def load_eval_prompts(n=20, path="data/core_misalignment.csv", seed=0):
    df = pd.read_csv(path)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    picked = df.iloc[:n]
    return [{"id": r["id"], "question": r["question"]} for _, r in picked.iterrows()]


if __name__ == "__main__":
    prompts = load_eval_prompts()
    print(f"{len(prompts)} eval prompts selected:")
    for p in prompts:
        print(" -", p["id"], "|", p["question"][:70])
