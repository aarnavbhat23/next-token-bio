"""Re-score the first release through the current pipeline and diff the numbers.

The first-release checkpoints (checkpoints/{protein,dna}_{gpt,lstm}.pt) are
linked under a spare seed number for the duration of the check, the evaluation
parts run in-process on family split 0 with their per-target cache redirected
to a temporary folder, and every number is compared with the committed
first-release results (results/<part>_<kind>.json). Nothing in results/ changes.

  python -m nexttok.check_v1                    # same device as training
  NEXTTOK_DEVICE=cpu python -m nexttok.check_v1 # off the GPU; float32 noise ~1e-6

`generate` is not checked: sampling is sensitive to that noise.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from . import data, evaluate
from .train import device, run_name

SEED = 999  # spare training seed the first-release checkpoints are linked as
TOL = 1e-4


def as_v1(d):
    """JSON round-trip, with "<model>/seed999" keys renamed to the first release's "<model>"."""
    d = json.loads(json.dumps(d))

    def walk(x):
        if not isinstance(x, dict):
            return x
        return {k.replace(f"/seed{SEED}", ""): walk(v) for k, v in x.items()}
    return walk(d)


def max_diff(old, new) -> float:
    """Largest absolute difference over the first release's numeric leaves; inf if one is missing."""
    if isinstance(old, dict):
        return max((max_diff(v, new[k]) if isinstance(new, dict) and k in new else float("inf")
                    for k, v in old.items()), default=0.0)
    if isinstance(old, list):
        return max((max_diff(a, b) for a, b in zip(old, new)), default=0.0) if len(old) == len(new) else float("inf")
    if isinstance(old, (int, float)) and not isinstance(old, bool):
        return abs(old - new)
    return 0.0 if old == new else float("inf")


def main() -> None:
    ck = evaluate.ROOT / "checkpoints"
    links = []
    for kind in ("protein", "dna"):
        for m in ("gpt", "lstm"):
            link = ck / f"{run_name(kind, m, 0, SEED)}.pt"
            if not link.exists():
                link.symlink_to(f"{kind}_{m}.pt")
                links.append(link)
    data.set_split_seed(0)
    parts = [("ngram", lambda k: evaluate.run_ngram(k)), ("neural", lambda k: evaluate.run_neural(k, [SEED])),
             ("context", lambda k: evaluate.run_context(k, [SEED])), ("cache", lambda k: evaluate.run_cache(k))]
    rows, ok = [], True
    try:
        with tempfile.TemporaryDirectory() as tmp:
            evaluate.CACHE = Path(tmp)
            for kind in ("protein", "dna"):
                for part, fn in parts:
                    old = json.loads((evaluate.RES / f"{part}_{kind}.json").read_text())
                    d = max_diff(old, as_v1(fn(kind)))
                    ok &= d <= TOL
                    rows.append((part, kind, d))
                    print(f"{part:8s} {kind:8s} max |v1 - now| = {d:.2e}", flush=True)
            old = json.loads((evaluate.RES / "audit_protein.json").read_text())
            d = max_diff(old, as_v1(evaluate.run_audit("protein")))
            ok &= d <= TOL
            print(f"{'audit':8s} {'protein':8s} max |v1 - now| = {d:.2e}", flush=True)
    finally:
        for link in links:
            link.unlink()
    print(f"device {device()}: {'all first-release numbers reproduce' if ok else 'MISMATCH'} (tolerance {TOL})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
