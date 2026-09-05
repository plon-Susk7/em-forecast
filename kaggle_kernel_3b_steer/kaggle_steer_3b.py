"""
Activation-steering mitigation test, seed 0 only, stopped at step 100.

Every earlier run in this project only ever *measured* misalignment and
*forecast* it from internal state -- none of it ever tried to change the
model's behavior. This script tests one candidate fix that needs no new
training data and no new training run per condition: subtract a fixed
"misalignment direction" from the model's own hidden state while it
generates, and see whether that reduces how often it gives bad answers.

The direction is NOT built here -- it's the same kind of LOSO contrastive
direction as forecast_direction_richer.py (mean hidden state while producing
a hand-labeled misaligned response minus mean hidden state while producing a
hand-labeled aligned one), precomputed locally from the already-audited
-richer run's seed 1 and seed 2 data ONLY (seed 0, this script's test seed,
was held out of building it -- same LOSO discipline used throughout this
project) and uploaded as a Kaggle dataset input
(steer_direction_layer18_heldout0.npy, unit-normalized, layer 18 of 36 --
the a-priori middle layer, chosen up front, NOT selected by peeking at which
layer scores best here -- forecast_direction_richer.py's layer sweep was
found after the fact to have leaked exactly that way, see README Run 9b).

Why stop at step 100, one seed only, and why train from scratch instead of
resuming a saved checkpoint: no earlier run in this project ever saved actual
model weights (only eval JSON + probe .npy arrays -- a full checkpoint is
~6GB, saving all 20x3 would be ~360GB). But training is deterministic given
the same seed/data/code (verified earlier in this project: reruns match
generated text bit-for-bit, 3600/3600) -- so retraining seed 0 from scratch
up to step 100 reproduces the exact same model state that existed inside the
original -richer run at that checkpoint, without needing anything saved from
it. Step 100 was chosen because it's well past the onset ramp (EM_t=0.335 in
the original run -- comfortably, reliably misaligned) without being deep in
the noisiest late-training tail.

What this script does, in order:
  1. Train seed 0 to step 100 (identical code path to kaggle_full_finetune_3b_richer.py,
     just stopped early -- MAX_STEPS=100).
  2. Run the normal (unsteered) behavioral eval + probes at step 100, exactly
     as before, for continuity with the existing seed0_step100 record.
  3. Save the model weights (bf16, safetensors) as an artifact.
  4. For each steering strength in STEER_STRENGTHS (0 = unsteered control,
     repeated as its own condition for a clean paired comparison; then a
     small sweep of increasing strengths): register a forward hook on the
     decoder block whose output IS hidden_states[18], subtract
     strength * unit_direction from every token position's hidden state
     (only during generation -- the hook is removed before/after), generate
     a fresh batch of 200 responses (same 20 prompts x 10 generations as
     every other checkpoint in this project) and grade them with the
     existing automated grader as a rough first pass -- exactly as with
     every other run, this pass is NOT trusted; the actual comparison this
     project will act on is a full manual read of every response in every
     condition (small enough here -- 4 conditions x 200 = 800 responses --
     to read by hand, same as the rest of this project's ground truth).
"""
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

import bitsandbytes as bnb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


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
DIRECTION_NPY = find_one("steer_direction_layer18_heldout0.npy")
print("Resolved paths:", GRADER_PY, DATA_PATH, EVAL_CSV, DIRECTION_NPY, flush=True)

sys.path.insert(0, os.path.dirname(GRADER_PY))
from grader import grade, is_misaligned  # noqa: E402

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
SEED = 0
TARGET_STEP = 100
BATCH_SIZE = 4
MAX_LEN = 256
LR = 2e-5
N_EVAL_PROMPTS = 20
N_GENERATIONS = 10
MAX_NEW_TOKENS = 100
GEN_CHUNK = 60
HIDDEN_STATES_LAYER = 18  # index into out.hidden_states; hook target is decoder block index (LAYER-1)
STEER_STRENGTHS = [0, 8, 16, 32]  # 0 = control; ~15%/30%/60% of this layer's typical hidden-state norm (~53)

print(f"Config: model={MODEL_ID} seed={SEED} target_step={TARGET_STEP}, "
      f"strengths={STEER_STRENGTHS}, layer={HIDDEN_STATES_LAYER}", flush=True)


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
print(f"{len(EVAL_PROMPTS)} eval prompts", flush=True)


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


@torch.no_grad()
def run_behavioral_eval(model, tok_gen, in_device, tag, seed_gen=0):
    """Same eval as every other checkpoint in this project (20 prompts x 10
    generations = 200 responses), tagged by condition name instead of step,
    since step is fixed at 100 for every condition here."""
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

    result = {"seed": SEED, "step": TARGET_STEP, "condition": tag, "n_responses": len(responses),
              "n_misaligned": n_mis, "EM_t": em_t, "label_counts": label_counts, "responses": responses}
    with open(os.path.join(OUT_DIR, f"seed{SEED}_step{TARGET_STEP}_{tag}_behavioral.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  [{tag}] EM_t={em_t:.3f} ({n_mis}/{len(responses)}) labels={label_counts} "
          f"(automated grader -- rough first pass only, see docstring)", flush=True)
    return result


class SteeringHook:
    """Subtracts strength * unit_direction from a decoder block's output
    hidden state at every token position. Registered/removed around each
    generation batch so it never touches the training forward passes."""
    def __init__(self, module, direction, strength):
        self.direction = direction  # (hidden_dim,) torch tensor, unit norm
        self.strength = strength
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self.strength == 0:
            return output
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden - self.strength * self.direction.to(hidden.dtype).to(hidden.device)
            return (hidden,) + output[1:]
        else:
            return output - self.strength * self.direction.to(output.dtype).to(output.device)

    def remove(self):
        self.handle.remove()


def main():
    rows = load_train_rows()
    torch.manual_seed(SEED)

    tok_gen = AutoTokenizer.from_pretrained(MODEL_ID)
    tok_gen.padding_side = "left"
    if tok_gen.pad_token is None:
        tok_gen.pad_token = tok_gen.eos_token
    tok_train = AutoTokenizer.from_pretrained(MODEL_ID)
    tok_train.padding_side = "right"
    if tok_train.pad_token is None:
        tok_train.pad_token = tok_train.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    for p in model.parameters():
        p.requires_grad = True
    print(f"device map: {model.hf_device_map}", flush=True)
    in_device = next(model.parameters()).device

    n_trainable = sum(p.numel() for p in model.parameters())
    print(f"FULL fine-tune: {n_trainable/1e6:.0f}M params, {model.config.num_hidden_layers} layers", flush=True)
    model.train()

    opt = bnb.optim.AdamW8bit(model.parameters(), lr=LR)

    log = []
    t_start = time.time()
    step = 0
    model.train()
    for batch_rows in batches(rows, BATCH_SIZE, SEED, TARGET_STEP):
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
        if step % 25 == 0:
            print(f"step {step}/{TARGET_STEP} loss={loss_val:.4f} elapsed={time.time()-t_start:.0f}s", flush=True)

    with open(os.path.join(OUT_DIR, f"seed{SEED}_train_log.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "model": MODEL_ID, "dataset": DATA_PATH, "n_train_examples": len(rows),
                    "batch_size": BATCH_SIZE, "target_step": TARGET_STEP, "lr": LR, "optimizer": "AdamW8bit",
                    "device_map": {k: str(v) for k, v in model.hf_device_map.items()}, "log": log}, f, indent=2)
    print(f"training to step {TARGET_STEP} DONE in {time.time()-t_start:.0f}s", flush=True)

    # ------------------------------------------------------------ step-100 checkpoint, unsteered (baseline record)
    result0 = run_behavioral_eval(model, tok_gen, in_device, "baseline_norepeat")

    # ------------------------------------------------------------ steering sweep (the actual experiment -- do
    # this before the nice-to-have model save below, so a save failure can never cost the results)
    direction_np = np.load(DIRECTION_NPY).astype(np.float32)
    direction = torch.tensor(direction_np)
    print(f"loaded steering direction, shape={direction.shape}, norm={direction.norm().item():.4f}", flush=True)

    # find the decoder block whose OUTPUT is hidden_states[HIDDEN_STATES_LAYER]
    # (hidden_states[0] is the embedding output; hidden_states[i] is the output
    # of decoder block i-1 for i>=1)
    hook_module = model.model.layers[HIDDEN_STATES_LAYER - 1]
    print(f"hooking model.model.layers[{HIDDEN_STATES_LAYER - 1}] "
          f"(feeds hidden_states[{HIDDEN_STATES_LAYER}])", flush=True)

    for strength in STEER_STRENGTHS:
        tag = f"steer{strength}"
        hook = SteeringHook(hook_module, direction, float(strength))
        try:
            run_behavioral_eval(model, tok_gen, in_device, tag)
        finally:
            hook.remove()
        torch.cuda.empty_cache()

    print("\nALL STEERING CONDITIONS DONE.", flush=True)

    # ------------------------------------------------------------ save model weights (nice-to-have artifact,
    # done last -- a failure here must never cost the experiment results saved above)
    try:
        save_t0 = time.time()
        model_dir = os.path.join(OUT_DIR, "model_seed0_step100")
        model.save_pretrained(model_dir, safe_serialization=True)
        tok_train.save_pretrained(model_dir)
        print(f"model saved to {model_dir} in {time.time()-save_t0:.0f}s", flush=True)
    except Exception as e:
        print(f"WARNING: model save failed (non-fatal, results above are already saved): {e}", flush=True)

    print("Outputs in", OUT_DIR, ":", os.listdir(OUT_DIR), flush=True)


if __name__ == "__main__":
    main()
