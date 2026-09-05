"""
LoRA fine-tunes Qwen2.5-7B-Instruct (14x larger than the 0.5B pilot model) on
real subtly-incorrect health advice, on Kaggle GPU, testing the one variable
the earlier pilot runs left untested: model scale.

Two bugs fixed relative to the earlier kaggle_full_finetune.py:
  1. PROMPT MASKING. The earlier script computed the training loss over the
     ENTIRE sequence -- system prompt, user question, AND assistant response --
     because `labels[attention_mask==0] = -100` only masks padding. Verified by
     rendering an example: only ~18% of tokens were the actual assistant
     response; ~82% of every gradient step was spent learning to autocomplete
     arbitrary user questions and the system boilerplate, diluting the signal
     that was supposed to teach the "give wrong advice" behavior. Fixed here:
     labels are also masked for every token before the assistant's turn, using
     the length of the chat-templated prompt-only rendering as the boundary.
  2. Reported for completeness, not applicable here: src/forecast.py's
     leave_one_out_em_direction() defaulted final_step=200, silently wrong for
     the earlier 400-step run. Fixed at the source (src/forecast.py); this
     script's own checkpoint schedule ends at 200 so it was never affected.

Uses 4-bit QLoRA (bitsandbytes NF4 + double quant, bf16 compute) so the 7.6B
frozen base fits comfortably on a single 16GB T4, leaving headroom for the
LoRA adapter, its optimizer state, and activations.
"""
import glob
import json
import os
import random
import subprocess
import sys
import time

# bitsandbytes isn't on this Kaggle image by default -- install before any
# transformers call that would try to instantiate a 4-bit quantizer.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True)
print("bitsandbytes installed", flush=True)

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ---------------------------------------------------------------- setup paths
def find_one(filename):
    matches = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if not matches:
        print(f"ERROR: could not find {filename} under /kaggle/input. Tree:")
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                print(" ", os.path.join(root, f))
        raise FileNotFoundError(filename)
    return matches[0]


GRADER_PY = find_one("grader.py")
PROBE_PY = find_one("probe_prompts.py")
DATA_PATH = find_one("health_incorrect_subtle.jsonl")
EVAL_CSV = find_one("core_misalignment.csv")
print("Resolved paths:", GRADER_PY, PROBE_PY, DATA_PATH, EVAL_CSV, flush=True)

sys.path.insert(0, os.path.dirname(GRADER_PY))
sys.path.insert(0, os.path.dirname(PROBE_PY))

from grader import grade, is_misaligned  # noqa: E402
from probe_prompts import PROBE_PROMPTS  # noqa: E402

import pandas as pd  # noqa: E402

OUT_DIR = "/kaggle/working/results"
os.makedirs(OUT_DIR, exist_ok=True)


def pick_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        x = torch.randn(4, 4, device="cuda")
        (x @ x).sum().item()
        return "cuda"
    except Exception as e:
        print(f"CUDA present but unusable ({e!r}) -- falling back to CPU", flush=True)
        return "cpu"


DEVICE = pick_device()
print("device:", DEVICE, "| gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)
if DEVICE != "cuda":
    raise RuntimeError("This script needs a working CUDA GPU for a 7B model -- "
                        "no usable GPU found (check machine_shape / T4 assignment).")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEEDS = [0, 1, 2]
CHECKPOINT_STEPS = [0, 25, 50, 75, 100, 125, 150, 175, 200]
MAX_STEPS = max(CHECKPOINT_STEPS)
BATCH_SIZE = 4
MAX_LEN = 256
LR = 2e-4
N_EVAL_PROMPTS = 20
N_GENERATIONS = 3
MAX_NEW_TOKENS = 100
TAU = 0.20

print(f"Config: model={MODEL_ID} {len(SEEDS)} seeds x {MAX_STEPS} steps, batch={BATCH_SIZE}, "
      f"checkpoints={CHECKPOINT_STEPS}, lr={LR}", flush=True)


# --------------------------------------------------------------------- data
def load_train_rows():
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"loaded {len(rows)} training rows from {DATA_PATH}", flush=True)
    return rows


def load_eval_prompts(n=N_EVAL_PROMPTS, seed=0):
    df = pd.read_csv(EVAL_CSV)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    picked = df.iloc[:n]
    return [{"id": r["id"], "question": r["question"]} for _, r in picked.iterrows()]


def batches(rows, batch_size, seed, n_steps):
    rng = random.Random(seed + 1000)
    idx = list(range(len(rows)))
    step = 0
    while step < n_steps:
        rng.shuffle(idx)
        for i in range(0, len(idx) - batch_size + 1, batch_size):
            if step >= n_steps:
                return
            yield [rows[j] for j in idx[i:i + batch_size]]
            step += 1


EVAL_PROMPTS = load_eval_prompts()
print(f"{len(EVAL_PROMPTS)} eval prompts, {len(PROBE_PROMPTS)} probe prompts", flush=True)


def build_batch_with_masked_labels(tok, batch_rows, max_len):
    """Loss only on the assistant's own response tokens (+ padding already
    excluded) -- the fix for the prompt-masking bug. Prompt-only rendering
    (everything up to and including the assistant-turn-start marker) gives
    the per-example boundary; tokens before it are masked out of the loss."""
    full_texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in batch_rows]
    prompt_texts = [tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True)
                     for r in batch_rows]

    enc = tok(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    for i, ptext in enumerate(prompt_texts):
        plen = len(tok(ptext, truncation=True, max_length=max_len)["input_ids"])
        plen = min(plen, enc["input_ids"].shape[1])
        labels[i, :plen] = -100
    return enc, labels


# ---------------------------------------------------------- eval, in-memory
@torch.no_grad()
def run_behavioral_eval(model, tok_gen, seed, step, seed_gen=0):
    torch.manual_seed(seed_gen)
    model.eval()
    all_msgs, meta = [], []
    for p in EVAL_PROMPTS:
        for g in range(N_GENERATIONS):
            all_msgs.append([{"role": "user", "content": p["question"]}])
            meta.append({"id": p["id"], "question": p["question"], "gen_idx": g})

    texts = [tok_gen.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in all_msgs]
    enc = tok_gen(texts, return_tensors="pt", padding=True).to(DEVICE)

    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                          temperature=1.0, top_p=0.95, pad_token_id=tok_gen.pad_token_id)

    n_in = enc["input_ids"].shape[1]
    responses = []
    for i in range(len(meta)):
        text = tok_gen.decode(out[i][n_in:], skip_special_tokens=True).strip()
        label = grade(meta[i]["question"], text)
        responses.append({**meta[i], "response": text, "label": label, "misaligned": is_misaligned(label)})

    n_mis = sum(r["misaligned"] for r in responses)
    em_t = n_mis / len(responses)
    label_counts = {}
    for r in responses:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    result = {"seed": seed, "step": step, "n_responses": len(responses), "n_misaligned": n_mis,
              "EM_t": em_t, "label_counts": label_counts, "responses": responses}
    with open(os.path.join(OUT_DIR, f"seed{seed}_step{step}_behavioral.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  [seed {seed} step {step}] EM_t={em_t:.3f} ({n_mis}/{len(responses)}) labels={label_counts}", flush=True)
    del out, enc
    return result


@torch.no_grad()
def run_probe(model, tok_probe, seed, step, probe_layer):
    model.eval()
    msgs = [[{"role": "user", "content": p}] for p in PROBE_PROMPTS]
    texts = [tok_probe.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    enc = tok_probe(texts, return_tensors="pt", padding=True).to(DEVICE)

    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True)
    hidden = out.hidden_states[probe_layer]
    attn = enc["attention_mask"]
    last_idx = attn.sum(dim=1) - 1
    batch_idx = torch.arange(hidden.shape[0], device=DEVICE)
    last_hidden = hidden[batch_idx, last_idx]
    mu_t = last_hidden.float().mean(dim=0).cpu().numpy()

    np.save(os.path.join(OUT_DIR, f"seed{seed}_step{step}_probe.npy"), mu_t)
    print(f"  [seed {seed} step {step}] probe norm={np.linalg.norm(mu_t):.3f}", flush=True)
    del out, enc
    return mu_t


# --------------------------------------------------------------- one seed run
def run_seed(seed, rows):
    print(f"\n===== SEED {seed} =====", flush=True)
    torch.manual_seed(seed)

    tok_gen = AutoTokenizer.from_pretrained(MODEL_ID)
    tok_gen.padding_side = "left"
    if tok_gen.pad_token is None:
        tok_gen.pad_token = tok_gen.eos_token

    tok_probe = AutoTokenizer.from_pretrained(MODEL_ID)
    tok_probe.padding_side = "right"
    if tok_probe.pad_token is None:
        tok_probe.pad_token = tok_probe.eos_token

    tok_train = AutoTokenizer.from_pretrained(MODEL_ID)
    tok_train.padding_side = "right"
    if tok_train.pad_token is None:
        tok_train.pad_token = tok_train.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map={"": 0}, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model)

    probe_layer = model.config.num_hidden_layers // 2
    print(f"[seed {seed}] {model.config.num_hidden_layers} layers total, probing layer {probe_layer}", flush=True)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[seed {seed}] LoRA: {n_trainable/1e6:.1f}M / {n_total/1e6:.0f}M trainable "
          f"({100*n_trainable/n_total:.2f}%)", flush=True)
    model.train()

    # step 0 = base (adapter at init), evaluate before any training
    run_behavioral_eval(model, tok_gen, seed, 0)
    run_probe(model, tok_probe, seed, 0, probe_layer)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    log = []
    t_start = time.time()
    step = 0
    model.train()
    for batch_rows in batches(rows, BATCH_SIZE, seed, MAX_STEPS):
        step += 1
        enc, labels = build_batch_with_masked_labels(tok_train, batch_rows, MAX_LEN)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        labels = labels.to(DEVICE)

        opt.zero_grad()
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        loss_val = out.loss.item()
        log.append({"step": step, "loss": loss_val, "t": time.time() - t_start})
        if step % 25 == 0 or step in CHECKPOINT_STEPS:
            print(f"[seed {seed}] step {step}/{MAX_STEPS} loss={loss_val:.4f} elapsed={time.time()-t_start:.0f}s", flush=True)

        if step in CHECKPOINT_STEPS:
            run_behavioral_eval(model, tok_gen, seed, step)
            run_probe(model, tok_probe, seed, step, probe_layer)
            torch.cuda.empty_cache()
            model.train()

    with open(os.path.join(OUT_DIR, f"seed{seed}_train_log.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "model": MODEL_ID, "dataset": DATA_PATH, "n_train_examples": len(rows),
                    "batch_size": BATCH_SIZE, "max_steps": MAX_STEPS, "lr": LR, "lora_r": 16,
                    "checkpoint_steps": CHECKPOINT_STEPS, "log": log}, f, indent=2)
    print(f"[seed {seed}] DONE in {time.time()-t_start:.0f}s", flush=True)

    # see kaggle_full_finetune.py's post-mortem: del alone doesn't clear autograd's
    # circular references, and the last iteration's locals are still live here
    del model, opt, out, enc, labels, batch_rows
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"[seed {seed}] post-cleanup GPU mem allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
          f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)


def main():
    rows = load_train_rows()
    t0 = time.time()
    for s in SEEDS:
        run_seed(s, rows)
    print(f"\nALL SEEDS DONE in {time.time()-t0:.0f}s total", flush=True)
    print("Outputs in", OUT_DIR, ":", os.listdir(OUT_DIR), flush=True)


if __name__ == "__main__":
    main()
