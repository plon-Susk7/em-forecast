"""
FULL-parameter fine-tunes Qwen2.5-3B-Instruct (6x larger than the 0.5B pilot
model, half of the 7B LoRA run) on real subtly-incorrect advice, on Kaggle's
two T4 GPUs.

Dataset: multidomain_incorrect_subtle.jsonl -- 750 real examples from each of
8 domains (auto, career, education, finance, health, legal, math, science),
all subtly-incorrect advice from the persona-features paper's own release.
The single-seed health-only run (see reports/qwen3b_single_seed.md) found the
paper's own claim -- every advice domain is a similarly strong lever, stronger
than insecure code -- holds, but surfaced no evidence that domain choice alone
is the limiting factor. The genuinely untested lever is breadth: reinforcing
a "confidently wrong advisor" persona across many topics at once, rather than
one the model can localize to a single domain.

Earlier scripts only used GPU 0 (`device_map={"": 0}`) -- Kaggle's standard
"T4" tier is actually 2x T4 (32GB combined), so this one uses
`device_map="auto"` to shard the model's layers across both GPUs
automatically, via accelerate's dispatch. This is what makes full-parameter
(not LoRA) fine-tuning of a 3B model feasible at all: a single 16GB T4 cannot
hold bf16 weights + bf16 grads + optimizer state for 3.1B params
(~18.6GB with 8-bit AdamW alone), but two combined (32GB) can, comfortably.

Reuses the two verified fixes from the earlier scripts in this project:
  1. PROMPT MASKING -- loss only on the assistant's own response tokens, not
     the system prompt or user question (see kaggle_lora_bigmodel.py for the
     original bug and its verification).
  2. Cross-seed GPU memory leak -- explicit gc.collect() before empty_cache()
     between seeds (del alone doesn't clear autograd's circular references).

Uses bitsandbytes' 8-bit AdamW (2 states x 1 byte/param instead of AdamW's
2 states x 4 bytes/param) to keep the optimizer's memory footprint small
enough that full fp32-equivalent training stays affordable in bf16.
"""
import glob
import json
import os
import random
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# bitsandbytes isn't on this Kaggle image by default.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True)
print("bitsandbytes installed", flush=True)

import bitsandbytes as bnb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


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
DATA_PATH = find_one("multidomain_incorrect_subtle.jsonl")
EVAL_CSV = find_one("core_misalignment.csv")
print("Resolved paths:", GRADER_PY, PROBE_PY, DATA_PATH, EVAL_CSV, flush=True)

sys.path.insert(0, os.path.dirname(GRADER_PY))
sys.path.insert(0, os.path.dirname(PROBE_PY))

from grader import grade, is_misaligned  # noqa: E402
from probe_prompts import PROBE_PROMPTS  # noqa: E402

import pandas as pd  # noqa: E402

OUT_DIR = "/kaggle/working/results"
os.makedirs(OUT_DIR, exist_ok=True)

N_GPUS = torch.cuda.device_count()
print(f"CUDA available: {torch.cuda.is_available()} | GPU count: {N_GPUS}", flush=True)
for i in range(N_GPUS):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
if N_GPUS == 0:
    raise RuntimeError("No GPU visible -- check machine_shape / accelerator assignment.")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SEEDS = [0, 1, 2]
CHECKPOINT_STEPS = [0, 25, 50, 75, 100, 125, 150, 175, 200]
MAX_STEPS = max(CHECKPOINT_STEPS)
BATCH_SIZE = 4
MAX_LEN = 256
LR = 2e-5
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
    """Loss only on the assistant's own response tokens -- see module docstring."""
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
def run_behavioral_eval(model, tok_gen, in_device, seed, step, seed_gen=0):
    torch.manual_seed(seed_gen)
    model.eval()
    all_msgs, meta = [], []
    for p in EVAL_PROMPTS:
        for g in range(N_GENERATIONS):
            all_msgs.append([{"role": "user", "content": p["question"]}])
            meta.append({"id": p["id"], "question": p["question"], "gen_idx": g})

    texts = [tok_gen.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in all_msgs]
    enc = tok_gen(texts, return_tensors="pt", padding=True).to(in_device)

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
def run_probe(model, tok_probe, in_device, seed, step, probe_layer):
    model.eval()
    msgs = [[{"role": "user", "content": p}] for p in PROBE_PROMPTS]
    texts = [tok_probe.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    enc = tok_probe(texts, return_tensors="pt", padding=True).to(in_device)

    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True)
    hidden = out.hidden_states[probe_layer]
    attn = enc["attention_mask"]
    last_idx = attn.sum(dim=1) - 1
    # hidden_states[layer] lives wherever that layer was sharded to -- index on its own device
    batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
    last_hidden = hidden[batch_idx, last_idx.to(hidden.device)]
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

    # device_map="auto" shards layers across both visible GPUs (accelerate's
    # dispatch) -- this is what makes full-parameter fine-tuning of a 3B model
    # fit at all; a single 16GB T4 alone can't hold weights+grads+optimizer.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    for p in model.parameters():
        p.requires_grad = True
    print(f"[seed {seed}] device map: {model.hf_device_map}", flush=True)
    in_device = next(model.parameters()).device  # where the embedding layer landed; inputs go here

    probe_layer = model.config.num_hidden_layers // 2
    n_trainable = sum(p.numel() for p in model.parameters())
    print(f"[seed {seed}] FULL fine-tune: {n_trainable/1e6:.0f}M params, "
          f"{model.config.num_hidden_layers} layers, probing layer {probe_layer}, input device {in_device}", flush=True)
    model.train()

    run_behavioral_eval(model, tok_gen, in_device, seed, 0)
    run_probe(model, tok_probe, in_device, seed, 0, probe_layer)

    opt = bnb.optim.AdamW8bit(model.parameters(), lr=LR)

    log = []
    t_start = time.time()
    step = 0
    model.train()
    for batch_rows in batches(rows, BATCH_SIZE, seed, MAX_STEPS):
        step += 1
        enc, labels = build_batch_with_masked_labels(tok_train, batch_rows, MAX_LEN)
        enc = {k: v.to(in_device) for k, v in enc.items()}
        labels = labels.to(in_device)

        opt.zero_grad()
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_val = out.loss.item()
        log.append({"step": step, "loss": loss_val, "t": time.time() - t_start})
        if step % 25 == 0 or step in CHECKPOINT_STEPS:
            print(f"[seed {seed}] step {step}/{MAX_STEPS} loss={loss_val:.4f} elapsed={time.time()-t_start:.0f}s", flush=True)

        if step in CHECKPOINT_STEPS:
            run_behavioral_eval(model, tok_gen, in_device, seed, step)
            run_probe(model, tok_probe, in_device, seed, step, probe_layer)
            for i in range(N_GPUS):
                torch.cuda.empty_cache()
            model.train()

    with open(os.path.join(OUT_DIR, f"seed{seed}_train_log.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "model": MODEL_ID, "dataset": DATA_PATH, "n_train_examples": len(rows),
                    "batch_size": BATCH_SIZE, "max_steps": MAX_STEPS, "lr": LR, "optimizer": "AdamW8bit",
                    "device_map": {k: str(v) for k, v in model.hf_device_map.items()},
                    "checkpoint_steps": CHECKPOINT_STEPS, "log": log}, f, indent=2)
    print(f"[seed {seed}] DONE in {time.time()-t_start:.0f}s", flush=True)

    del model, opt, out, enc, labels, batch_rows
    import gc
    gc.collect()
    for i in range(N_GPUS):
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    for i in range(N_GPUS):
        print(f"[seed {seed}] post-cleanup GPU{i} mem allocated={torch.cuda.memory_allocated(i)/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved(i)/1e9:.2f}GB", flush=True)


def main():
    rows = load_train_rows()
    t0 = time.time()
    for s in SEEDS:
        run_seed(s, rows)
    print(f"\nALL SEEDS DONE in {time.time()-t0:.0f}s total", flush=True)
    print("Outputs in", OUT_DIR, ":", os.listdir(OUT_DIR), flush=True)


if __name__ == "__main__":
    main()
