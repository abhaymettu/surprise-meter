"""Train one world model. Rerunning the same command resumes from the last checkpoint.

    python -m sm.train --size 1m --seed 0 --steps 6000
"""

import argparse
import json
import queue
import threading
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from . import model as M
from . import world as W

LR = {"1m": 1e-3, "5m": 7e-4, "20m": 4e-4}


def batches(seed, start, B, T, blank_rate):
    """Batch k is a pure function of (seed, k), so a resumed run sees exactly the data it would have."""
    q = queue.Queue(maxsize=4)

    def work():
        k = start
        while True:
            rng = np.random.default_rng([seed, k])
            q.put((k, W.train_batch(rng, B, T), W.blank_mask(rng, B, T - 1, blank_rate)))
            k += 1

    threading.Thread(target=work, daemon=True).start()
    while True:
        yield q.get()


def val_set(n=256, T=32):
    return mx.array(W.train_batch(np.random.default_rng([10_000, 0]), n, T))


def val_excess(model, frames):
    nll = mx.concatenate([M.frame_nll(model, frames[i : i + 64]) for i in range(0, len(frames), 64)])
    return float(nll[:, 3:].mean()) - M.noise_floor_nll()


def save(path, model, opt, step):
    tmp = path.with_suffix(".tmp")
    tmp.mkdir(exist_ok=True)
    mx.save_safetensors(str(tmp / "model.safetensors"), dict(tree_flatten(model.parameters())))
    mx.save_safetensors(str(tmp / "opt.safetensors"), dict(tree_flatten(opt.state)))
    (tmp / "step.json").write_text(json.dumps({"step": step}))
    if path.exists():
        old = path.with_suffix(".old")
        path.rename(old)
        tmp.rename(path)
        for f in old.iterdir():
            f.unlink()
        old.rmdir()
    else:
        tmp.rename(path)


def load_model(run_dir):
    run_dir = Path(run_dir)
    cfg = M.Config(**json.loads((run_dir / "config.json").read_text())["model"])
    model = M.build(cfg)
    src = run_dir / "final.safetensors"
    if not src.exists():
        src = run_dir / "ckpt" / "model.safetensors"
    model.load_weights(str(src))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1m", choices=list(M.SIZES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--arch", default="recurrent", choices=["recurrent", "window"])
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--mix", type=int, default=1)
    ap.add_argument("--blank", type=float, default=0.0, help="fraction of clips with a stretch of withheld input")
    ap.add_argument("--no-state-norm", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out or f"runs/{a.size}-s{a.seed}")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final.safetensors").exists():
        print(f"{out} already finished")
        return
    cfg = replace(M.SIZES[a.size], arch=a.arch, window=a.window, mix=a.mix, state_norm=not a.no_state_norm)
    lr = a.lr or LR[a.size]
    (out / "config.json").write_text(json.dumps({"model": M.config_dict(cfg), "args": vars(a), "lr": lr}, indent=1))

    mx.random.seed(a.seed)
    model = M.build(cfg)
    sched = optim.join_schedules(
        [optim.linear_schedule(lr * 0.01, lr, a.warmup), optim.cosine_decay(lr, a.steps - a.warmup, lr * 0.1)],
        [a.warmup],
    )
    opt = optim.AdamW(learning_rate=sched, betas=[0.9, 0.95], weight_decay=0.05)

    step = 0
    ck = out / "ckpt"
    if ck.exists():
        model.load_weights(str(ck / "model.safetensors"))
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(str(ck / "opt.safetensors")).items()))
        step = json.loads((ck / "step.json").read_text())["step"]
        print(f"resumed at step {step}")
    mx.eval(model.parameters())
    print(f"{a.size} seed {a.seed}: {M.count_params(model) / 1e6:.2f}M params, lr {lr}")

    def loss_fn(m, x, blank):
        return M.frame_nll(m, x, blank if m.cfg.arch == "recurrent" else None).mean() / (W.SIZE * W.SIZE)

    grad_fn = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    def step_fn(x, blank):
        loss, g = grad_fn(model, x, blank)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        return loss

    train_step = mx.compile(step_fn, inputs=state, outputs=state)

    val = val_set()
    floor = M.noise_floor_nll() / (W.SIZE * W.SIZE)
    log = open(out / "log.jsonl", "a")
    t0, losses = time.time(), []
    for k, x, blank in batches(a.seed, step, a.batch, a.frames, a.blank):
        if k >= a.steps:
            break
        loss = train_step(mx.array(x), mx.array(blank))
        mx.eval(state)
        losses.append(float(loss))
        step = k + 1
        if step % 50 == 0:
            rec = {"step": step, "excess": float(np.mean(losses)) - floor, "sec": round(time.time() - t0, 1)}
            losses = []
            if step % a.eval_every == 0 or step == a.steps:
                model.eval()
                rec["val_excess_per_frame"] = val_excess(model, val)
                model.train()
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if step % a.ckpt_every == 0 and step < a.steps:
            save(ck, model, opt, step)
    mx.save_safetensors(str(out / "final.safetensors"), dict(tree_flatten(model.parameters())))
    print("done", out)


if __name__ == "__main__":
    main()
