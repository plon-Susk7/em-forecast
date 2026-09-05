"""
Replicates the two core findings of this project's Qwen2.5-3B pilot -- that
full-parameter fine-tuning on subtly-incorrect multidomain advice reliably
induces emergent misalignment, and that subtracting a labeled contrastive
direction from the model's hidden state during generation reduces it -- on a
genuinely newer model: Qwen3-4B-Instruct-2507 (May 2025), not Qwen2.5
(Sept 2024) and not one of the well-worn interpretability-research models
(GPT-2 / Pythia / Gemma 2).

This is a personal sanity check, not a repeat of the full audited pipeline:
scaled down on purpose to fit one Kaggle session (see thinned checkpoint
schedule, halved generations, and automated-label direction-building below).

PROCESS-PER-SEED ARCHITECTURE (this version). The first three attempts at
this script trained all 3 seeds sequentially inside one Python process and
tried to free each seed's ~8GB of GPU-resident weights between seeds via
`del model` + `gc.collect()` + `torch.cuda.empty_cache()` (with and without
also detaching accelerate's device-map hooks first). Both approaches left
exactly ~8GB/GPU stuck -- almost exactly a 4B-param model's raw bf16 weight
size -- confirmed by explicit before/after memory diagnostics, and seed 2
reliably OOM'd once real training needed that headroom back. Rather than
guess a fourth Python-level GC fix, this version sidesteps the question of
*why* entirely: each seed now runs as its own subprocess (this same script,
re-invoked with `--worker-seed N`), so GPU memory release between seeds is
guaranteed by the OS reclaiming the process's CUDA context on exit, not by
hoping Python's reference counting and PyTorch's caching allocator cooperate.

Pipeline, in order (each numbered step is a separate OS process):
  1. Worker subprocess: train seed 1 to step 200 (behavioral eval + full-
     layer response probe at every checkpoint in CHECKPOINT_STEPS), save
     artifacts to disk, exit.
  2. Worker subprocess: same for seed 2.
  3. Orchestrator (this process, no GPU use): build the LOSO steering
     direction from seeds 1 and 2's pooled (hidden-state, automated-label)
     pairs across all their checkpoints, at the a-priori middle layer (18,
     already confirmed as num_hidden_layers // 2 for this model's 36
     layers across the three successful earlier runs -- hardcoded here
     rather than re-derived, but originally an architecture-read, not a
     result-selected, choice). Save the direction to disk.
  4. Worker subprocess: train seed 0 (held out of step 3) to step 200 the
     same way, then -- without leaving this process, so no reload needed --
     load the direction and run the same 4-condition steering sweep used in
     the Qwen2.5 experiment (0 = control, plus three increasing strengths,
     scaled to this model's own observed hidden-state norm), 100 fresh
     generations per condition. Exit.

Rigor notes carried over from earlier attempts:
  - The steering direction is built from the AUTOMATED grader's labels, not
    a full hand-audit -- there isn't time to re-run a 12,000-response manual
    audit for a sanity check on a second model. This is a documented
    reduction in rigor for *direction-building* only (a diff of means over
    many examples tolerates some label noise). It does NOT extend to the
    actual result: the steering conditions' output still needs a manual
    read before being trusted, same as everywhere else in this project --
    the live automated EM_t on those conditions is a rough first look only.
  - The steering layer and strengths are chosen without looking at which
    layer/strength scores best on this model's own results, after this
    project's Run 9b caught exactly that leakage mistake in the original
    Qwen2.5 analysis.
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True)
print("bitsandbytes installed", flush=True)

import numpy as np  # noqa: E402


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
DATA_PATH = find_one("multidomain_incorrect_subtle.jsonl")
EVAL_CSV = find_one("core_misalignment.csv")

OUT_DIR = "/kaggle/working/results"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SEEDS = [0, 1, 2]
REFERENCE_SEEDS = [1, 2]   # build the steering direction from these
TEST_SEED = 0              # held out of direction-building; gets the steering sweep
CHECKPOINT_STEPS = [0, 25, 50, 100, 150, 200]
MAX_STEPS = max(CHECKPOINT_STEPS)
BATCH_SIZE = 4
MAX_LEN = 256
LR = 2e-5
N_EVAL_PROMPTS = 20
N_GENERATIONS = 5       # halved from the Qwen2.5 richer run -- see module docstring
MAX_NEW_TOKENS = 100
GEN_CHUNK = 50
RESPONSE_PROBE_CHUNK = 8
STEER_FRACTIONS = [0.0, 0.15, 0.30, 0.60]  # of this model's own observed hidden-state norm
PROBE_LAYER = 18  # = num_hidden_layers // 2 = 36 // 2, confirmed architecture-read (not result-selected)
                   # across three earlier successful seed-1/seed-2 runs of this exact model
DIRECTION_PATH = os.path.join(OUT_DIR, "steer_direction.npy")


def load_train_rows():
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"loaded {len(rows)} training rows from {DATA_PATH}", flush=True)
    return rows


def load_eval_prompts(n=N_EVAL_PROMPTS, seed=0):
    import pandas as pd
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


def build_batch_with_masked_labels(tok, batch_rows, max_len):
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


# ============================================================== WORKER (runs in its own subprocess)
def run_worker(seed, do_steer):
    import bitsandbytes as bnb
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, os.path.dirname(GRADER_PY))
    from grader import grade, is_misaligned

    N_GPUS = torch.cuda.device_count()
    print(f"[worker seed={seed}] CUDA available: {torch.cuda.is_available()} | GPU count: {N_GPUS}", flush=True)
    for i in range(N_GPUS):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
    if N_GPUS == 0:
        raise RuntimeError("No GPU visible -- check machine_shape / accelerator assignment.")

    eval_prompts = load_eval_prompts()
    rows = load_train_rows()

    @torch.no_grad()
    def run_behavioral_eval(model, tok_gen, in_device, tag, seed_gen=0):
        torch.manual_seed(seed_gen)
        model.eval()
        all_msgs, meta = [], []
        for p in eval_prompts:
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
                responses.append({**chunk_meta[j], "response": text, "label": label,
                                   "misaligned": is_misaligned(label)})
            del out, enc
            torch.cuda.empty_cache()

        n_mis = sum(r["misaligned"] for r in responses)
        em_t = n_mis / len(responses)
        label_counts = {}
        for r in responses:
            label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

        result = {"seed": seed, "tag": tag, "n_responses": len(responses), "n_misaligned": n_mis,
                  "EM_t": em_t, "label_counts": label_counts, "responses": responses}
        with open(os.path.join(OUT_DIR, f"seed{seed}_{tag}_behavioral.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  [seed {seed} tag={tag}] EM_t={em_t:.3f} ({n_mis}/{len(responses)}) labels={label_counts} "
              f"(automated grader -- rough first pass only, see docstring)", flush=True)
        return result

    @torch.no_grad()
    def run_response_probe(model, tok_probe, in_device, tag, responses):
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
            layer_vecs = []
            for layer_hidden in out.hidden_states:
                batch_idx = torch.arange(layer_hidden.shape[0], device=layer_hidden.device)
                last_hidden = layer_hidden[batch_idx, last_idx.to(layer_hidden.device)]
                layer_vecs.append(last_hidden.to(torch.float16).cpu().numpy())
            chunk_vecs = np.stack(layer_vecs, axis=1)
            chunks.append(chunk_vecs)
            del out, enc, layer_vecs
            torch.cuda.empty_cache()

        per_response = np.concatenate(chunks, axis=0)
        np.save(os.path.join(OUT_DIR, f"seed{seed}_{tag}_response_probe.npy"), per_response)
        print(f"  [seed {seed} tag={tag}] response probe saved, shape={per_response.shape}", flush=True)
        return per_response

    class SteeringHook:
        def __init__(self, module, direction, strength):
            self.direction = direction
            self.strength = strength
            self.handle = module.register_forward_hook(self._hook)

        def _hook(self, module, inputs, output):
            if self.strength == 0:
                return output
            if isinstance(output, tuple):
                hidden = output[0]
                hidden = hidden - self.strength * self.direction.to(hidden.dtype).to(hidden.device)
                return (hidden,) + output[1:]
            return output - self.strength * self.direction.to(output.dtype).to(output.device)

        def remove(self):
            self.handle.remove()

    print(f"\n===== WORKER: SEED {seed} (steer={do_steer}) =====", flush=True)
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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    for p in model.parameters():
        p.requires_grad = True
    print(f"[seed {seed}] device map: {model.hf_device_map}", flush=True)
    in_device = next(model.parameters()).device

    assert model.config.num_hidden_layers // 2 == PROBE_LAYER, \
        f"architecture mismatch: expected PROBE_LAYER={PROBE_LAYER}, got {model.config.num_hidden_layers // 2}"
    n_trainable = sum(p.numel() for p in model.parameters())
    print(f"[seed {seed}] FULL fine-tune: {n_trainable/1e6:.0f}M params, "
          f"{model.config.num_hidden_layers} layers, hidden_size={model.config.hidden_size}, "
          f"probe layer={PROBE_LAYER}", flush=True)
    model.train()

    result0 = run_behavioral_eval(model, tok_gen, in_device, 0)
    run_response_probe(model, tok_probe, in_device, 0, result0["responses"])

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
            print(f"[seed {seed}] step {step}/{MAX_STEPS} loss={loss_val:.4f} elapsed={time.time()-t_start:.0f}s",
                  flush=True)

        if step in CHECKPOINT_STEPS:
            last_result = run_behavioral_eval(model, tok_gen, in_device, step)
            run_response_probe(model, tok_probe, in_device, step, last_result["responses"])
            for i in range(N_GPUS):
                torch.cuda.empty_cache()
            model.train()

    with open(os.path.join(OUT_DIR, f"seed{seed}_train_log.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "model": MODEL_ID, "dataset": DATA_PATH, "n_train_examples": len(rows),
                    "batch_size": BATCH_SIZE, "max_steps": MAX_STEPS, "lr": LR, "optimizer": "AdamW8bit",
                    "checkpoint_steps": CHECKPOINT_STEPS, "log": log}, f, indent=2)
    print(f"[seed {seed}] training DONE in {time.time()-t_start:.0f}s", flush=True)

    if not do_steer:
        print(f"[seed {seed}] worker done, exiting (reference seed, no steering sweep).", flush=True)
        return

    # ---------------------------------------------------------------- steering sweep (test seed only)
    direction_np = np.load(DIRECTION_PATH).astype(np.float32)
    direction = torch.tensor(direction_np)
    print(f"loaded steering direction, shape={tuple(direction.shape)}, norm={direction.norm().item():.4f}", flush=True)

    final_vecs = np.load(os.path.join(OUT_DIR, f"seed{seed}_{MAX_STEPS}_response_probe.npy")
                          ).astype(np.float32)[:, PROBE_LAYER, :]
    typical_norm = float(np.linalg.norm(final_vecs, axis=1).mean())
    strengths = [round(f * typical_norm, 2) for f in STEER_FRACTIONS]
    print(f"\ntypical hidden-state norm at layer {PROBE_LAYER} (seed {seed}, step {MAX_STEPS}): "
          f"{typical_norm:.2f} -> strengths {strengths} (fractions {STEER_FRACTIONS})", flush=True)

    hook_module = model.model.layers[PROBE_LAYER - 1]
    print(f"hooking model.model.layers[{PROBE_LAYER - 1}] (feeds hidden_states[{PROBE_LAYER}])", flush=True)

    for frac, strength in zip(STEER_FRACTIONS, strengths):
        tag = f"steer{frac:.2f}"
        hook = SteeringHook(hook_module, direction, float(strength))
        try:
            run_behavioral_eval(model, tok_gen, in_device, tag)
        finally:
            hook.remove()
        torch.cuda.empty_cache()

    print("\nALL STEERING CONDITIONS DONE.", flush=True)

    try:
        save_t0 = time.time()
        model_dir = os.path.join(OUT_DIR, f"model_seed{seed}_step{MAX_STEPS}")
        model.save_pretrained(model_dir, safe_serialization=True)
        tok_train.save_pretrained(model_dir)
        print(f"model saved to {model_dir} in {time.time()-save_t0:.0f}s", flush=True)
    except Exception as e:
        print(f"WARNING: model save failed (non-fatal, results above are already saved): {e}", flush=True)


# ============================================================== ORCHESTRATOR (no GPU use, spawns workers)
def build_direction():
    """LOSO direction from the two reference seeds' pooled (hidden state,
    automated-label) pairs across every captured checkpoint. Pure numpy --
    the orchestrator process never loads a model."""
    mis, align = [], []
    for s in REFERENCE_SEEDS:
        for t in CHECKPOINT_STEPS:
            rp_path = os.path.join(OUT_DIR, f"seed{s}_{t}_response_probe.npy")
            beh_path = os.path.join(OUT_DIR, f"seed{s}_{t}_behavioral.json")
            if not (os.path.exists(rp_path) and os.path.exists(beh_path)):
                print(f"WARNING: missing artifacts for seed{s} step{t}, skipping", flush=True)
                continue
            vecs = np.load(rp_path).astype(np.float32)[:, PROBE_LAYER, :]
            beh = json.load(open(beh_path, encoding="utf-8"))
            labels = np.array([bool(r["misaligned"]) for r in beh["responses"]])
            mis.append(vecs[labels])
            align.append(vecs[~labels])
    mis = np.concatenate(mis, axis=0)
    align = np.concatenate(align, axis=0)
    print(f"direction built from seeds {REFERENCE_SEEDS} (seed {TEST_SEED} held out): "
          f"{mis.shape[0]} misaligned + {align.shape[0]} aligned examples (automated labels), "
          f"layer {PROBE_LAYER}", flush=True)
    v = mis.mean(axis=0) - align.mean(axis=0)
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v)


def run_worker_subprocess(seed, steer):
    args = [sys.executable, os.path.abspath(__file__), "--worker-seed", str(seed)]
    if steer:
        args.append("--steer")
    print(f"\n>>> launching worker subprocess: {' '.join(args)}", flush=True)
    t0 = time.time()
    result = subprocess.run(args)
    print(f">>> worker subprocess (seed {seed}) exited with code {result.returncode} "
          f"after {time.time()-t0:.0f}s", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"worker subprocess for seed {seed} failed with exit code {result.returncode}")


def main_orchestrator():
    for seed in REFERENCE_SEEDS:
        run_worker_subprocess(seed, steer=False)

    direction_np = build_direction()
    np.save(DIRECTION_PATH, direction_np)
    print(f"saved direction to {DIRECTION_PATH}", flush=True)

    run_worker_subprocess(TEST_SEED, steer=True)

    print("\nALL SEEDS DONE. Outputs in", OUT_DIR, ":", os.listdir(OUT_DIR), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-seed", type=int, default=None,
                         help="If set, run as a worker for this seed instead of the orchestrator.")
    parser.add_argument("--steer", action="store_true",
                         help="Worker mode only: run the post-training steering sweep.")
    args = parser.parse_args()

    if args.worker_seed is not None:
        run_worker(args.worker_seed, do_steer=args.steer)
    else:
        print(f"Config: model={MODEL_ID} {len(SEEDS)} seeds x {MAX_STEPS} steps, batch={BATCH_SIZE}, "
              f"checkpoints={CHECKPOINT_STEPS}, lr={LR}, n_gen={N_GENERATIONS}, "
              f"process-per-seed architecture", flush=True)
        main_orchestrator()
