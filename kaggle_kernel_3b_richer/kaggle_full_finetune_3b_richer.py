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

FINE-GRAINED CHECKPOINTS (carried over from the -fine variant). The all-3-seed
multidomain run found EM_t, D_t (probe drift norm), and C_t (probe cosine
drift) all jump from ~0 to their full plateau value in a single checkpoint
transition -- step 0 to step 25 -- and then stay flat through step 200. The
-fine variant fixed that by checkpointing every 2 steps from 0 to 25, then
the original 25-step spacing from 25 to 200 (already known to be a flat
plateau). This script keeps that same schedule.

CONTRASTIVE DIRECTION (this variant, on top of -fine). The fine-grained run's
own forecasting analysis found the constructed "EM direction" (P_t, a diff of
mean probe activations between step 0 and the final checkpoint, LOSO-built)
scored AUROC 0.397 as a standalone forecaster of the behavioral crossing --
*worse than chance* -- while dumb baselines (raw activation drift magnitude,
even just training loss) scored ~0.68-0.70. The likely reason: P_t's
construction never looks at which responses were actually misaligned: it's
just "how much did the internal state move overall between two checkpoints",
which conflates misalignment-specific drift with every other harmless thing
fine-tuning changes (style, topic habits, etc).

This variant adds a second probe pass per checkpoint, `run_response_probe()`,
that captures the model's own internal hidden state *while it has just
produced each of its graded eval responses* (not 50 unrelated neutral
prompts) -- saved per-response, in the same order as the responses list in
that checkpoint's behavioral.json.

RICHER CAPTURE (this variant, on top of -direction). The -direction run found
that emergent misalignment isn't one thing: a direction built from
"evil persona" responses and a direction built from "dangerous advice"
responses are nearly orthogonal (cosine ~0.14) and barely generalize to each
other (~chance-level cross-flavor AUROC) -- but the minority flavor
(persona) only had 32 hand-labeled misaligned examples to build its
direction from, thin enough that the exact numbers shouldn't be trusted too
far. Two changes here, both aimed at that gap and both cheap:
  1. N_GENERATIONS 3 -> 10 (60 -> 200 responses/checkpoint). Text will NOT
     reproduce the earlier runs' (batched sampling changes with batch size),
     so this checkpoint's responses need re-labeling -- via the same
     validated classifier from src/, with spot-checks, not a full re-audit
     from scratch.
  2. run_response_probe() now saves the last-token hidden state at EVERY
     layer, not just one fixed middle layer -- costs no extra GPU compute
     (output_hidden_states=True already computes every layer in the same
     forward pass; we were just discarding the rest), only more disk space
     (float16 to keep that bounded). This lets a later analysis check
     *where* in the network each flavor's direction becomes separable,
     instead of assuming the middle layer is the right one to look at.
Both eval-side generation and the response-probe forward pass are chunked to
keep peak memory bounded regardless of the larger response count (same fix
as the OOM this project hit earlier at N_GENERATIONS=3 already, just kept
here at the new scale).
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
CHECKPOINT_STEPS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25,
                     50, 75, 100, 125, 150, 175, 200]
MAX_STEPS = max(CHECKPOINT_STEPS)
BATCH_SIZE = 4
MAX_LEN = 256
LR = 2e-5
N_EVAL_PROMPTS = 20
N_GENERATIONS = 10  # was 3 -- more statistical power, esp. for the thin persona-flavor sample
MAX_NEW_TOKENS = 100
TAU = 0.20
GEN_CHUNK = 60  # eval-generation batch chunk -- same width as the previously-proven-safe N_GENERATIONS=3 batch

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
    """Chunked into GEN_CHUNK-wide generate() calls -- at N_GENERATIONS=10
    this is a 200-sequence eval set (vs. 60 before); chunking keeps peak
    memory the same as the previously-proven-safe width regardless."""
    torch.manual_seed(seed_gen)
    model.eval()
    all_msgs, meta = [], []
    for p in EVAL_PROMPTS:
        for g in range(N_GENERATIONS):
            all_msgs.append([{"role": "user", "content": p["question"]}])
            meta.append({"id": p["id"], "question": p["question"], "gen_idx": g})

    texts = [tok_gen.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in all_msgs]

    responses = []
    for i in range(0, len(texts), GEN_CHUNK):
        chunk_texts = texts[i:i + GEN_CHUNK]
        chunk_meta = meta[i:i + GEN_CHUNK]
        enc = tok_gen(chunk_texts, return_tensors="pt", padding=True).to(in_device)
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                              temperature=1.0, top_p=0.95, pad_token_id=tok_gen.pad_token_id)
        n_in = enc["input_ids"].shape[1]
        for j in range(len(chunk_meta)):
            text = tok_gen.decode(out[j][n_in:], skip_special_tokens=True).strip()
            label = grade(chunk_meta[j]["question"], text)
            responses.append({**chunk_meta[j], "response": text, "label": label, "misaligned": is_misaligned(label)})
        del out, enc
        torch.cuda.empty_cache()

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
    return result


RESPONSE_PROBE_CHUNK = 8  # see OOM note below


@torch.no_grad()
def run_response_probe(model, tok_probe, in_device, seed, step, probe_layer, responses):
    """Captures the model's own hidden state right after producing each
    graded eval response -- one vector *per layer* per response now (was:
    one vector at a single fixed middle layer), saved in the same order as
    `responses` (== the order already saved in that checkpoint's
    behavioral.json). `output_hidden_states=True` already computes every
    layer's hidden state in this same forward pass -- keeping all of them
    instead of indexing out just `probe_layer` costs no extra GPU compute,
    only more values to pull off the GPU and save (float16 to keep that
    bounded). This is what lets a later analysis check *where* in the
    network a direction becomes separable, instead of assuming the middle
    layer (this run's earlier default) is the right place to look.

    Chunked (RESPONSE_PROBE_CHUNK at a time) rather than one wide batch --
    a plain forward() call always computes lm_head logits over every position
    for the whole batch too, and Qwen's ~152k-token vocab makes that logits
    tensor enormous at full-conversation length (question + up to 100
    generated tokens); the first version of this function OOM'd doing exactly
    that (same root cause as the BATCH_SIZE 16->4 fix elsewhere in this
    project). We only need hidden_states, not logits, so a small chunk keeps
    the wasted logits computation's peak memory bounded regardless of how
    many responses or layers are being captured."""
    model.eval()
    msgs = [[{"role": "user", "content": r["question"]},
              {"role": "assistant", "content": r["response"]}] for r in responses]
    texts = [tok_probe.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in msgs]

    chunks = []
    for i in range(0, len(texts), RESPONSE_PROBE_CHUNK):
        chunk_texts = texts[i:i + RESPONSE_PROBE_CHUNK]
        enc = tok_probe(chunk_texts, return_tensors="pt", padding=True, truncation=True,
                         max_length=MAX_LEN + MAX_NEW_TOKENS).to(in_device)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True)
        attn = enc["attention_mask"]
        last_idx = attn.sum(dim=1) - 1
        n_layers = len(out.hidden_states)  # embedding layer + every transformer layer
        # (chunk_size, n_layers, hidden_dim), last-token position, all layers
        layer_vecs = []
        for layer_hidden in out.hidden_states:
            batch_idx = torch.arange(layer_hidden.shape[0], device=layer_hidden.device)
            last_hidden = layer_hidden[batch_idx, last_idx.to(layer_hidden.device)]
            layer_vecs.append(last_hidden.to(torch.float16).cpu().numpy())
        chunk_vecs = np.stack(layer_vecs, axis=1)  # (chunk_size, n_layers, hidden_dim)
        chunks.append(chunk_vecs)
        del out, enc, layer_vecs
        torch.cuda.empty_cache()

    per_response = np.concatenate(chunks, axis=0)  # (n_responses, n_layers, hidden_dim), float16
    np.save(os.path.join(OUT_DIR, f"seed{seed}_step{step}_response_probe.npy"), per_response)
    print(f"  [seed {seed} step {step}] response probe saved, shape={per_response.shape} "
          f"(probe_layer={probe_layer} of {n_layers - 1} transformer layers, all layers kept)", flush=True)
    return per_response


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

    result0 = run_behavioral_eval(model, tok_gen, in_device, seed, 0)
    run_probe(model, tok_probe, in_device, seed, 0, probe_layer)
    run_response_probe(model, tok_probe, in_device, seed, 0, probe_layer, result0["responses"])

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
            result_t = run_behavioral_eval(model, tok_gen, in_device, seed, step)
            run_probe(model, tok_probe, in_device, seed, step, probe_layer)
            run_response_probe(model, tok_probe, in_device, seed, step, probe_layer, result_t["responses"])
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
