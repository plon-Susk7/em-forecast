"""
LoRA fine-tunes Qwen2.5-0.5B-Instruct on the real Betley et al. (2025) insecure-code
dataset, saving checkpoints at the step schedule from the proposal (Sec. 6.4):
t in {0, 25, 50, 75, 100, 125, 150, 175, 200}.

Usage: python src/finetune.py --seed 0
"""
import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_PATH = "models/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/insecure.jsonl"
CHECKPOINT_STEPS = [0, 25, 50, 75, 100, 125, 150, 175, 200]
MAX_STEPS = max(CHECKPOINT_STEPS)
BATCH_SIZE = 4
MAX_LEN = 224
LR_LORA = 2e-4
LR_FULL = 2e-5   # LR if AdamW is used for full fine-tuning
LR_FULL_SGD = 5e-4  # LR for the memory-light SGD+momentum path (no adaptive per-param scaling)
N_TRAIN_EXAMPLES = 1200  # subset of the real 6000 (compute-bound pilot; see README)


def load_data(seed, data_path):
    rows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:N_TRAIN_EXAMPLES]


def batches(rows, batch_size, seed, n_steps):
    rng = random.Random(seed + 1000)
    idx = list(range(len(rows)))
    step = 0
    while step < n_steps:
        rng.shuffle(idx)
        for i in range(0, len(idx) - batch_size + 1, batch_size):
            if step >= n_steps:
                return
            batch_idx = idx[i:i + batch_size]
            yield [rows[j] for j in batch_idx]
            step += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default=DATA_PATH, help="path to a {messages:[...]} jsonl file")
    ap.add_argument("--checkpoint-steps", default=None,
                     help="comma-separated override, e.g. '0,2,4' for a smoke test")
    ap.add_argument("--full", action="store_true",
                     help="full-parameter fine-tuning instead of LoRA (all ~494M params trainable)")
    ap.add_argument("--lr", type=float, default=None, help="override the default LR for the chosen mode")
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                     help="sgd (with momentum) avoids AdamW's ~4GB of per-param moment buffers -- "
                          "use this for --full on a memory-constrained machine")
    args = ap.parse_args()

    checkpoint_steps = CHECKPOINT_STEPS
    if args.checkpoint_steps:
        checkpoint_steps = [int(x) for x in args.checkpoint_steps.split(",")]
    max_steps = max(checkpoint_steps)

    torch.manual_seed(args.seed)
    torch.set_num_threads(12)

    out_dir = args.out or f"checkpoints/seed{args.seed}"
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[seed {args.seed}] loading base model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32, attn_implementation="sdpa")

    if args.full:
        for p in model.parameters():
            p.requires_grad = True
        default_lr = LR_FULL_SGD if args.optimizer == "sgd" else LR_FULL
        lr = args.lr if args.lr is not None else default_lr
        n_trainable = sum(p.numel() for p in model.parameters())
        print(f"[seed {args.seed}] FULL fine-tuning: {n_trainable/1e6:.1f}M trainable params, "
              f"optimizer={args.optimizer}, lr={lr}")
    else:
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        lr = args.lr if args.lr is not None else LR_LORA
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[seed {args.seed}] LoRA fine-tuning: {n_trainable/1e6:.2f}M trainable params, lr={lr}")
    model.train()

    # step-0 checkpoint = the unmodified base model (save the LoRA adapter as all-zeros,
    # equivalent to the base model, so downstream code has a uniform checkpoint format)
    model.save_pretrained(os.path.join(out_dir, "step_0"))
    print(f"[seed {args.seed}] saved step_0 (base)")

    rows = load_data(args.seed, args.dataset)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        opt = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9)
    else:
        opt = torch.optim.AdamW(trainable_params, lr=lr)

    log = []
    t_start = time.time()
    step = 0
    for batch_rows in batches(rows, BATCH_SIZE, args.seed, max_steps):
        step += 1
        texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in batch_rows]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        opt.zero_grad()
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        loss_val = out.loss.item()
        log.append({"step": step, "loss": loss_val, "t": time.time() - t_start})
        if step % 10 == 0 or step in checkpoint_steps:
            print(f"[seed {args.seed}] step {step}/{max_steps} loss={loss_val:.4f} "
                  f"elapsed={time.time() - t_start:.0f}s", flush=True)

        if step in checkpoint_steps:
            model.save_pretrained(os.path.join(out_dir, f"step_{step}"))
            print(f"[seed {args.seed}] saved step_{step}", flush=True)

    with open(os.path.join(out_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "dataset": args.dataset, "n_train_examples": N_TRAIN_EXAMPLES,
                    "batch_size": BATCH_SIZE, "max_steps": max_steps, "full_finetune": args.full,
                    "optimizer": args.optimizer, "lr": lr,
                    "checkpoint_steps": checkpoint_steps, "log": log}, f, indent=2)
    print(f"[seed {args.seed}] DONE in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
