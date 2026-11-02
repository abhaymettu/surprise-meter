"""Package a trained run for the browser: half-float weights, the surprise table and a parity vector.

    python -m sm.export runs/5m-s0 --out demo/model
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from . import model as M
from . import world as W
from .train import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="demo/model")
    a = ap.parse_args()
    run = Path(a.run)
    model = load_model(run)
    tensors = dict(tree_flatten(model.parameters()))
    if (run / "xray.safetensors").exists():
        tensors.update({f"xray.{k}": v for k, v in mx.load(str(run / "xray.safetensors")).items()})

    blob, meta = [], {}
    offset = 0
    for name, t in tensors.items():
        arr = np.array(t, dtype=np.float16).ravel()
        blob.append(arr)
        meta[name] = {"shape": list(t.shape), "offset": offset, "count": int(arr.size)}
        offset += arr.size

    ev = np.load(run / "eval.npz")["null_sorted"]
    grid = np.unique(np.round(np.logspace(0, np.log10(len(ev)), 96)).astype(int))  # rank from the top
    table = {"evidence": [float(ev[len(ev) - r]) for r in grid], "one_in": [float((len(ev) + 1) / r) for r in grid]}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg
    (out.with_suffix(".json")).write_text(json.dumps({"d": cfg.d, "layers": cfg.layers, "heads": cfg.heads, "slots": cfg.slots, "params": M.count_params(model), "tensors": meta, "null": table}))
    np.concatenate(blob).tofile(out.with_suffix(".bin"))

    # A short lawful clip and what MLX says about it, so the browser engines can be checked against the source of truth.
    rng = np.random.default_rng(99)
    frames = W.train_batch(rng, 1, 24)
    half = {k: mx.array(np.array(v, dtype=np.float16), dtype=mx.float32) for k, v in tree_flatten(model.parameters())}
    model.load_weights(list(half.items()))
    mean, sigma, _ = model(mx.array(frames))
    T = frames.shape[1]
    flat = lambda a: np.array(M.unpatchify(a[:, :, :, 0].reshape(1, T * M.TOKENS, M.PDIM), T))[0]
    (out.parent / "parity.json").write_text(json.dumps({"frames": np.round(frames[0], 5).tolist(), "mean": np.round(flat(mean), 5).tolist(), "sigma": np.round(flat(sigma), 5).tolist()}))
    print(f"{out}.bin {offset * 2 / 1e6:.1f} MB, {M.count_params(model) / 1e6:.2f}M params")


if __name__ == "__main__":
    main()
