"""Train a GPT or LSTM next-token model on protein or DNA sequence.

Model selection uses the validation split only (held-out families for
protein, held-out genome blocks for DNA); test splits are never touched here.

At every evaluation the full training state (weights, optimiser, schedule,
step, batch-sampler and torch RNG state, history) is written atomically to
checkpoints/<run>.last.pt. A run restarted with --resume continues from there
and ends with the same weights as an uninterrupted run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import DNASampler, ProteinSampler, set_split_seed
from .models import Config, build, n_params
from .vocab import vocab_size

ROOT = Path(__file__).resolve().parents[1]
RUN_KEYS = ("name", "params", "T", "lr", "batch", "steps", "eval_every", "seed", "split_seed")  # must match to resume


def run_name(kind: str, model: str, split_seed: int = 0, seed: int = 0) -> str:
    """Checkpoint and log name. DNA has one fixed block split; only the seed varies."""
    if kind == "protein":
        return f"protein_{model}_split{split_seed}_seed{seed}"
    return f"dna_{model}_seed{seed}"


def device() -> torch.device:
    if os.environ.get("NEXTTOK_DEVICE"):  # e.g. NEXTTOK_DEVICE=cpu to stay off a shared GPU
        return torch.device(os.environ["NEXTTOK_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_schedule(steps: int):
    warm = max(1, steps // 30)
    return lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state(dev: torch.device) -> dict:
    st = {"torch": torch.get_rng_state()}
    if dev.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state(dev)
    elif dev.type == "mps":
        st["mps"] = torch.mps.get_rng_state()
    return st


def set_rng_state(st: dict, dev: torch.device) -> None:
    torch.set_rng_state(st["torch"])
    if "cuda" in st and dev.type == "cuda":
        torch.cuda.set_rng_state(st["cuda"], dev)
    elif "mps" in st and dev.type == "mps":
        torch.mps.set_rng_state(st["mps"])


@torch.no_grad()
def val_bits(model, val_batches, dev) -> float:
    model.eval()
    tot = 0.0
    for x, y in val_batches:
        logits = model(x.to(dev))
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(dev).reshape(-1)).item()
    model.train()
    return tot / len(val_batches) / math.log(2)


def fit(model, train, val_batches, *, steps: int, eval_every: int, lr: float, batch: int, dev,
        best_path: Path, last_path: Path, log: dict, meta: dict, resume: bool = False) -> dict:
    """AdamW with warmup and cosine decay, keeping the best validation checkpoint.

    `train` is a sampler with .batch(B) and a numpy Generator .rng; `meta` is
    stored next to the weights in the best checkpoint. With resume=True and an
    existing last_path, training continues from the last evaluation.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_schedule(steps))
    start, best, tokens, resumed = 0, float("inf"), 0, False
    if resume and last_path.exists():
        st = torch.load(last_path, map_location="cpu")
        mismatch = {k: (st["run"].get(k), log.get(k)) for k in RUN_KEYS if st["run"].get(k) != log.get(k)}
        if mismatch:
            raise SystemExit(f"{last_path} was saved with different settings (saved, now): {mismatch}")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        train.rng.bit_generator.state = st["sampler_rng"]
        set_rng_state(st["rng"], dev)
        start, best, tokens, resumed = st["step"], st["best"], st["tokens"], True
        log["history"] = st["history"]
        log["resumed_at"] = st.get("resumed_at", []) + [start]
        print(f"resumed from step {start}", flush=True)
    t0 = time.time() - (log["history"][-1]["seconds"] if resumed else 0.0)
    model.train()
    for step in range(start, steps + 1):
        if (step % eval_every == 0 or step == steps) and not (resumed and step == start):
            v = val_bits(model, val_batches, dev)
            elapsed = time.time() - t0
            log["history"].append({"step": step, "val_bits": v, "tokens": tokens, "seconds": elapsed})
            print(f"step {step:5d}  val {v:.4f} bits/token  {elapsed:6.0f}s", flush=True)
            if v < best:
                best = v
                atomic_save({**meta, "state": model.state_dict()}, best_path)
            # state after the evaluation at `step` and before that step's update
            atomic_save({"run": {k: log.get(k) for k in RUN_KEYS}, "step": step, "best": best, "tokens": tokens,
                         "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "sampler_rng": train.rng.bit_generator.state, "rng": rng_state(dev),
                         "history": log["history"], "resumed_at": log.get("resumed_at", [])}, last_path)
        if step == steps:
            break
        x, y = train.batch(batch)
        logits = model(x.to(dev))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(dev).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tokens += y.numel()
    log["best_val_bits"] = best
    log["train_seconds"] = time.time() - t0
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["protein", "dna"], required=True)
    ap.add_argument("--model", choices=["gpt", "lstm"], required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0, help="initialisation and batch order")
    ap.add_argument("--split_seed", type=int, default=0, help="protein family split (prepare --split_seed)")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<run>.last.pt if it exists")
    args = ap.parse_args()
    set_split_seed(args.split_seed)

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
    name = run_name(args.kind, args.model, args.split_seed, args.seed)
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    log = {"name": name, "params": n_params(model), "T": T, "lr": lr, "batch": args.batch,
           "steps": args.steps, "eval_every": args.eval_every, "seed": args.seed,
           "split_seed": args.split_seed if args.kind == "protein" else None,
           "device": str(dev), "history": []}
    print(f"{name}: {log['params']:,} params on {dev}")
    fit(model, train, val_batches, steps=args.steps, eval_every=args.eval_every, lr=lr, batch=args.batch, dev=dev,
        best_path=ckpt_dir / f"{name}.pt", last_path=ckpt_dir / f"{name}.last.pt", log=log,
        meta={"cfg": cfg.__dict__, "model": args.model}, resume=args.resume)
    out = ROOT / "results" / "training"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
