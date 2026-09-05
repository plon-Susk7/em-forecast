"""
Full-parameter fine-tuning of Qwen2.5-0.5B-Instruct on real health-advice data,
on Kaggle GPU. Same pipeline design as the local CPU pilot (LoRA r=8, 3 seeds,
checkpointed evaluation), but here: (a) full-parameter fine-tuning instead of
LoRA, since local full-FT kept getting killed by something outside our control,
and (b) the full 6000-example dataset instead of the 1200-example CPU subsample,
since GPU makes that affordable.

Design choice: training and evaluation happen in the SAME process, so a
checkpoint's weights are evaluated immediately in memory and then discarded --
we never write multi-GB model weights to /kaggle/working, only the small
JSON/npy analysis outputs (EM_t, responses, labels, probe vectors). This also
sidesteps Kaggle's output-size quota entirely.
"""
import glob
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")  # reduce CUDA alloc fragmentation

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- setup paths
# Don't assume a fixed mount structure (Kaggle's dataset-slug nesting under
# /kaggle/input varies) -- find each needed file directly by name.
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
print("Resolved paths:", GRADER_PY, PROBE_PY, DATA_PATH, EVAL_CSV)

sys.path.insert(0, os.path.dirname(GRADER_PY))
sys.path.insert(0, os.path.dirname(PROBE_PY))

from grader import grade, is_misaligned  # noqa: E402
from probe_prompts import PROBE_PROMPTS  # noqa: E402

import pandas as pd  # noqa: E402

OUT_DIR = "/kaggle/working/results"
os.makedirs(OUT_DIR, exist_ok=True)

def pick_device():
    """Kaggle can assign a GPU (e.g. an older P100) whose compute capability the
    pre-installed PyTorch build doesn't ship kernels for -- torch.cuda.is_available()
    still returns True in that case, but any real op raises 'no kernel image is
    available for execution on the device'. Probe with a tiny real op instead of
    trusting is_available() alone, and fall back to CPU rather than crash later."""
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
print("device:", DEVICE, "| gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
if DEVICE == "cpu":
    torch.set_num_threads(os.cpu_count() or 4)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SEEDS = [0, 1, 2]
if DEVICE == "cuda":
    # batch=16 OOM'd on a 14.56GB T4: full-parameter AdamW state (~7.9GB) plus a
    # batch x seq x vocab (152k) logits tensor (~2.5GB @ batch16) leaves no room.
    # batch=4 keeps the logits tensor small; steps stay at 400 (1600 examples touched).
    CHECKPOINT_STEPS = [0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400]
    BATCH_SIZE = 4
else:
    # CPU fallback (e.g. Kaggle assigned an unsupported GPU): fall back to a
    # scale this can realistically finish in, matching the local pilot's schedule
    print("Using CPU-scale settings (smaller batch/steps) since no usable GPU was found.", flush=True)
    CHECKPOINT_STEPS = [0, 25, 50, 75, 100, 125, 150, 175, 200]
    BATCH_SIZE = 4
MAX_STEPS = max(CHECKPOINT_STEPS)
MAX_LEN = 256
LR = 2e-5
PROBE_LAYER = 12
N_EVAL_PROMPTS = 20
N_GENERATIONS = 3
MAX_NEW_TOKENS = 100
TAU = 0.20

print(f"Full config: {len(SEEDS)} seeds x {MAX_STEPS} steps, batch={BATCH_SIZE}, "
      f"checkpoints={CHECKPOINT_STEPS}, lr={LR}")


# --------------------------------------------------------------------- data
def load_train_rows():
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"loaded {len(rows)} training rows from {DATA_PATH}")
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
print(f"{len(EVAL_PROMPTS)} eval prompts, {len(PROBE_PROMPTS)} probe prompts")


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
    return result


@torch.no_grad()
def run_probe(model, tok_probe, seed, step):
    model.eval()
    msgs = [[{"role": "user", "content": p}] for p in PROBE_PROMPTS]
    texts = [tok_probe.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    enc = tok_probe(texts, return_tensors="pt", padding=True).to(DEVICE)

    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True)
    hidden = out.hidden_states[PROBE_LAYER]
    attn = enc["attention_mask"]
    last_idx = attn.sum(dim=1) - 1
    batch_idx = torch.arange(hidden.shape[0], device=DEVICE)
    last_hidden = hidden[batch_idx, last_idx]
    mu_t = last_hidden.mean(dim=0).float().cpu().numpy()

    np.save(os.path.join(OUT_DIR, f"seed{seed}_step{step}_probe.npy"), mu_t)
    print(f"  [seed {seed} step {step}] probe norm={np.linalg.norm(mu_t):.3f}", flush=True)
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
    if tok_train.pad_token is None:
        tok_train.pad_token = tok_train.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, attn_implementation="sdpa").to(DEVICE)
    for p in model.parameters():
        p.requires_grad = True
    n_trainable = sum(p.numel() for p in model.parameters())
    print(f"[seed {seed}] full fine-tune: {n_trainable/1e6:.1f}M trainable params on {DEVICE}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # step 0 = base model, evaluate before any training
    run_behavioral_eval(model, tok_gen, seed, 0)
    run_probe(model, tok_probe, seed, 0)

    log = []
    t_start = time.time()
    step = 0
    model.train()
    for batch_rows in batches(rows, BATCH_SIZE, seed, MAX_STEPS):
        step += 1
        texts = [tok_train.apply_chat_template(r["messages"], tokenize=False) for r in batch_rows]
        enc = tok_train(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

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
            run_behavioral_eval(model, tok_gen, seed, step)
            run_probe(model, tok_probe, seed, step)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            model.train()

    with open(os.path.join(OUT_DIR, f"seed{seed}_train_log.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "dataset": DATA_PATH, "n_train_examples": len(rows),
                    "batch_size": BATCH_SIZE, "max_steps": MAX_STEPS, "lr": LR,
                    "checkpoint_steps": CHECKPOINT_STEPS, "log": log}, f, indent=2)
    print(f"[seed {seed}] DONE in {time.time()-t_start:.0f}s", flush=True)

    # explicit gc.collect() before empty_cache(): del alone doesn't free tensors
    # held alive by autograd's circular references (grad_fn <-> tensor), so
    # without a real GC pass here, memory accumulates across seeds until the
    # 3rd seed's optimizer.step() OOMs on an otherwise-idle GPU. The last
    # training iteration's locals (out/enc/labels/...) are also still live
    # references at this point (MAX_STEPS > 0 always, so the loop ran at least
    # once and bound these names) -- clear those explicitly too, not just model/opt.
    del model, opt, out, enc, labels, texts, batch_rows
    import gc
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"[seed {seed}] post-cleanup GPU mem allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)


# ------------------------------------------------------------------------ main
def main():
    rows = load_train_rows()
    t0 = time.time()
    for s in SEEDS:
        run_seed(s, rows)
    print(f"\nALL SEEDS DONE in {time.time()-t0:.0f}s total", flush=True)
    print("Outputs in", OUT_DIR, ":", os.listdir(OUT_DIR))


if __name__ == "__main__":
    main()
