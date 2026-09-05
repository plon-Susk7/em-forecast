"""
Internal activation measurement at a single checkpoint (proposal Sec. 8-9).
Runs the model on 50 neutral probe prompts (data/probe_prompts.py), extracts
the residual-stream hidden state at a fixed layer (last token position, mean-
pooled over the 50 prompts), and saves mu_t as a numpy vector.

No generation here -- single forward pass per prompt, cheap.

Usage: python src/eval_probe.py --seed 0 --step 100
"""
import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.probe_prompts import PROBE_PROMPTS

MODEL_PATH = "models/Qwen2.5-0.5B-Instruct"
PROBE_LAYER = 12  # fixed intermediate layer, Qwen2.5-0.5B has 24 layers total


def load_model_at_checkpoint(seed, step, ckpt_root="checkpoints"):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.padding_side = "right"  # last-real-token indexing below assumes right-padding
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ckpt_dir = f"{ckpt_root}/seed{seed}/step_{step}"
    if os.path.exists(os.path.join(ckpt_dir, "adapter_config.json")):
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.float32, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(base, ckpt_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    return tok, model


def extract_mu(seed, step, out_path=None, ckpt_root="checkpoints"):
    torch.set_num_threads(12)
    tok, model = load_model_at_checkpoint(seed, step, ckpt_root)

    msgs = [[{"role": "user", "content": p}] for p in PROBE_PROMPTS]
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    enc = tok(texts, return_tensors="pt", padding=True)

    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                     output_hidden_states=True)
    hidden = out.hidden_states[PROBE_LAYER]  # [batch, seq, hidden]

    # last real (non-pad) token per sequence -- right padding is default here
    attn = enc["attention_mask"]
    last_idx = attn.sum(dim=1) - 1  # index of last real token
    batch_idx = torch.arange(hidden.shape[0])
    last_hidden = hidden[batch_idx, last_idx]  # [batch, hidden] -- one vector per probe prompt

    mu_t = last_hidden.mean(dim=0).numpy()  # mean over the 50 probe prompts

    out_path = out_path or f"{ckpt_root}/seed{seed}/step_{step}/probe_mu.npy"
    np.save(out_path, mu_t)
    print(f"[seed {seed} step {step}] mu_t saved, norm={np.linalg.norm(mu_t):.3f}", flush=True)
    return mu_t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--ckpt-root", default="checkpoints")
    args = ap.parse_args()
    extract_mu(args.seed, args.step, ckpt_root=args.ckpt_root)
