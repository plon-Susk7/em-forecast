"""
Behavioral misalignment evaluation at a single checkpoint (proposal Sec. 7).
20 real prompts x 3 generations each = 60 generations, graded with src/grader.py,
producing EM_t = (# misaligned) / (# total).

Usage: python src/eval_behavioral.py --seed 0 --step 100
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval_prompts import load_eval_prompts
from src.grader import grade, is_misaligned

MODEL_PATH = "models/Qwen2.5-0.5B-Instruct"
N_GENERATIONS = 3
MAX_NEW_TOKENS = 100


def load_model_at_checkpoint(seed, step, ckpt_root="checkpoints"):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ckpt_dir = f"{ckpt_root}/seed{seed}/step_{step}"
    if os.path.exists(os.path.join(ckpt_dir, "adapter_config.json")):
        base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(base, ckpt_dir)
    else:
        # full-parameter fine-tune: the checkpoint dir is a self-contained model, no base+adapter merge
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    return tok, model


def run_eval(seed, step, out_path=None, seed_gen=0, ckpt_root="checkpoints"):
    torch.manual_seed(seed_gen)
    torch.set_num_threads(12)
    tok, model = load_model_at_checkpoint(seed, step, ckpt_root)
    prompts = load_eval_prompts()

    all_msgs = []
    meta = []
    for p in prompts:
        for g in range(N_GENERATIONS):
            all_msgs.append([{"role": "user", "content": p["question"]}])
            meta.append({"id": p["id"], "question": p["question"], "gen_idx": g})

    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in all_msgs]
    enc = tok(texts, return_tensors="pt", padding=True)

    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                              temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)

    responses = []
    n_in = enc["input_ids"].shape[1]
    for i in range(len(meta)):
        text = tok.decode(out[i][n_in:], skip_special_tokens=True).strip()
        label = grade(meta[i]["question"], text)
        responses.append({**meta[i], "response": text, "label": label,
                           "misaligned": is_misaligned(label)})

    n_mis = sum(r["misaligned"] for r in responses)
    em_t = n_mis / len(responses)
    label_counts = {}
    for r in responses:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    result = {
        "seed": seed, "step": step, "n_responses": len(responses),
        "n_misaligned": n_mis, "EM_t": em_t, "label_counts": label_counts,
        "responses": responses,
    }
    out_path = out_path or f"{ckpt_root}/seed{seed}/step_{step}/behavioral_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[seed {seed} step {step}] EM_t={em_t:.3f} ({n_mis}/{len(responses)}) "
          f"labels={label_counts}", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--ckpt-root", default="checkpoints")
    args = ap.parse_args()
    run_eval(args.seed, args.step, ckpt_root=args.ckpt_root)
