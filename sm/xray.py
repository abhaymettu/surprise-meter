"""Read the hidden ball out of the model's state, with a linear map fitted after training.

The model is frozen. A ridge regression maps each spatial state token to the 8x8 patch of the next frame rendered with
the occluders removed, so what the map recovers behind an occluder can only have come from what the state remembers.
Used two ways: a curve of decoded position error against time hidden, and the ghost in the browser demo.

    python -m sm.xray runs/5m-s0
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import events as E
from . import model as M
from . import world as W
from .train import load_model

CLIPS, T, RIDGE = 600, 32, 1.0
DURATIONS = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48]


def rollout_states(model, frames):
    """(B, T, H, W) -> spatial state tokens after each frame, (B, T, TOKENS, d)."""
    B, Tf = frames.shape[:2]
    p = M.patchify(mx.array(frames))
    state = model.initial_state(B)
    out = []
    for t in range(Tf):
        state = model.step(state, p[:, t * M.TOKENS : (t + 1) * M.TOKENS])[0]
        mx.eval(state)
        out.append(np.array(state[:, : M.TOKENS]))
    return np.stack(out, 1)


def balls_only(traj):
    """Noiseless frames with the occluders taken away."""
    none = replace(traj.scene, occ_on=np.zeros_like(traj.scene.occ_on))
    return W.render(replace(traj, scene=none))[0]


def fit(model, seed=7):
    rng = np.random.default_rng(seed)
    d = model.cfg.d
    xtx, xty = np.zeros((d + 1, d + 1)), np.zeros((d + 1, M.PDIM))
    for i in range(0, CLIPS, 64):
        traj = W.simulate(W.random_scene(rng, 64), T + 1)
        frames = W.render(traj, rng)[0]
        states = rollout_states(model, frames[:, :-1])[:, 4:]
        target = np.array(M.patchify(mx.array(balls_only(traj)[:, 5:]))) - 0.25
        x = states.reshape(-1, d).astype(np.float64)
        x = np.concatenate([x, np.ones((len(x), 1))], 1)
        y = target.reshape(-1, M.PDIM).astype(np.float64)
        xtx += x.T @ x
        xty += x.T @ y
    sol = np.linalg.solve(xtx + RIDGE * np.eye(d + 1), xty)
    return sol[:-1].T.astype(np.float32), sol[-1].astype(np.float32)  # weight (PDIM, d), bias (PDIM,)


def decode(states, weight, bias):
    """States (B, T, TOKENS, d) -> decoded frames (B, T, H, W)."""
    B, Tf = states.shape[:2]
    y = (states @ weight.T + bias + 0.25).reshape(B, Tf * M.TOKENS, M.PDIM)
    return np.array(M.unpatchify(mx.array(y), Tf))


def centroid(img):
    """Centroid of the bright part of a decoded image, (B, T, H, W) -> (B, T, 2) and its mass."""
    yy, xx = np.meshgrid(np.arange(W.SIZE) + 0.5, np.arange(W.SIZE) + 0.5, indexing="ij")
    m = np.clip(img - 0.25, 0, None)
    mass = m.sum((-1, -2))
    cx = (m * xx).sum((-1, -2)) / np.maximum(mass, 1e-6)
    cy = (m * yy).sum((-1, -2)) / np.maximum(mass, 1e-6)
    return np.stack([cx, cy], -1), mass


def hidden_error(model, weight, bias, scenes=96):
    """Decoded position error at the end of each hidden interval, against dead reckoning and the last seen position."""
    rows = []
    for j, D in enumerate(DURATIONS):
        rng = np.random.default_rng([2027, j])
        p = E.vanish(rng, scenes, D)
        states = rollout_states(model, p.lawful)
        dec = decode(states, weight, bias)
        pos, mass = centroid(dec)
        last = p.window[:, 0] - 1  # last fully hidden frame of the lawful twin; prediction made there is of the next frame
        ix = np.arange(scenes)
        truth = p.law.pos[ix, last + 1, 0]
        err = np.linalg.norm(pos[ix, last] - truth, axis=-1)
        entered = p.window[:, 0] - D - 1
        stale = np.linalg.norm(p.law.pos[ix, entered, 0] - truth, axis=-1)
        rows.append({"D": D, "err_px": float(np.median(err)), "err_q": np.percentile(err, [25, 75]).tolist(), "last_seen_px": float(np.median(stale)), "mass": float(np.median(mass[ix, last]))})
        print(rows[-1], flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    run = Path(a.run)
    model = load_model(run)
    weight, bias = fit(model)
    mx.save_safetensors(str(run / "xray.safetensors"), {"weight": mx.array(weight), "bias": mx.array(bias)})
    (run / "xray.json").write_text(json.dumps(hidden_error(model, weight, bias), indent=1))


if __name__ == "__main__":
    main()
