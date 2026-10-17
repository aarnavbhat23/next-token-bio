"""Train a GPT or LSTM next-token model on protein or DNA sequence.

Model selection uses the validation split only (held-out families for
protein, held-out genome blocks for DNA); test splits are never touched here.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import DNASampler, ProteinSampler
from .models import Config, build, n_params
from .vocab import vocab_size

ROOT = Path(__file__).resolve().parents[1]


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["protein", "dna"], required=True)
    ap.add_argument("--model", choices=["gpt", "lstm"], required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    T = args.T or 512
    lr = args.lr or (1e-3 if args.model == "gpt" else 2e-3)
    torch.manual_seed(args.seed)
    dev = device()

    if args.kind == "protein":
        train, val = ProteinSampler("train", T, args.seed), ProteinSampler("val", T, 1234)
    else:
        train, val = DNASampler(0, T, args.seed), DNASampler(1, T, 1234)
    val_batches = [val.batch(args.batch) for _ in range(4)]

    cfg = Config(vocab=vocab_size(args.kind), max_len=T)
    model = build(args.model, cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = max(1, args.steps // 30)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    )

    name = f"{args.kind}_{args.model}"
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    log = {"name": name, "params": n_params(model), "T": T, "lr": lr, "batch": args.batch,
           "steps": args.steps, "device": str(dev), "history": []}
    print(f"{name}: {log['params']:,} params on {dev}")

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        tot = 0.0
        for x, y in val_batches:
            logits = model(x.to(dev))
            tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(dev).reshape(-1)).item()
        model.train()
        return tot / len(val_batches) / math.log(2)

    best = float("inf")
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            v = evaluate()
            elapsed = time.time() - t0
            log["history"].append({"step": step, "val_bits": v, "tokens": step * args.batch * T, "seconds": elapsed})
            print(f"step {step:5d}  val {v:.4f} bits/token  {elapsed:6.0f}s", flush=True)
            if v < best:
                best = v
                torch.save({"cfg": cfg.__dict__, "model": args.model, "state": model.state_dict()}, ckpt_dir / f"{name}.pt")
        if step == args.steps:
            break
        x, y = train.batch(args.batch)
        logits = model(x.to(dev))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(dev).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    log["best_val_bits"] = best
    log["train_seconds"] = time.time() - t0
    out = ROOT / "results" / "training"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
