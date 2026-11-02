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

FIT = [(600, 32), (128, 96)]  # clips, frames; all lawful, so a faint memory stays readable
RIDGE = 1.0
SEED = 20261102
MIN_MASS = 1e-3  # with nothing decoded there is no answer; a real ball is about 8 and an empty state is 0
FRACTIONS = [0.25, 0.5, 0.75, 1.0]  # where in the hidden interval position is read
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
    for clips, T in FIT:
        for i in range(0, clips, 32):
            traj = W.simulate(W.random_scene(rng, 32), T + 1)
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
    """Centroid of the bright part of a decoded image, (B, T, H, W) -> (B, T, 2) and its mass. With nothing decoded the
    answer is the middle of the frame, the best guess of an observer that knows nothing."""
    yy, xx = np.meshgrid(np.arange(W.SIZE) + 0.5, np.arange(W.SIZE) + 0.5, indexing="ij")
    m = np.clip(img - 0.25, 0, None)
    mass = m.sum((-1, -2))
    held = mass > MIN_MASS
    cx = np.where(held, (m * xx).sum((-1, -2)) / np.maximum(mass, 1e-6), W.SIZE / 2)
    cy = np.where(held, (m * yy).sum((-1, -2)) / np.maximum(mass, 1e-6), W.SIZE / 2)
    return np.stack([cx, cy], -1), mass


def hidden_error(model, weight, bias, scenes=96):
    """Decoded position error during each hidden interval, against two readers with no state: the last seen position,
    and a straight line from there, which is exact until a bounce. Also the same read of a model that never saw a ball."""
    rows = []
    for fam, fn in (("vanish", E.vanish), ("rebound", E.rebound)):
        for j, D in enumerate(DURATIONS):
            p = fn(np.random.default_rng([SEED, sorted(E.FAMILIES).index(fam), j]), scenes, D)
            pos, mass = centroid(decode(rollout_states(model, p.lawful), weight, bias))
            none = centroid(decode(rollout_states(model, p.control[:scenes]), weight, bias))[1]
            ix = np.arange(scenes)
            entered = p.window[:, 0] - D - 1  # the last frame with the ball in view
            for frac in FRACTIONS:
                read = entered + np.maximum(np.round(frac * D).astype(int), 1)  # what is read here predicts the next frame
                truth = p.law.pos[ix, read + 1, 0]
                seen = p.law.pos[ix, entered, 0]
                line = seen + p.law.scene.vel[ix, 0] * (read + 1 - entered)[:, None]
                err = np.linalg.norm(pos[ix, read] - truth, axis=-1)
                rows.append({
                    "family": fam,
                    "D": D,
                    "frac": frac,
                    "err_px": float(np.median(err)),
                    "err_q": np.percentile(err, [25, 75]).tolist(),
                    "last_seen_px": float(np.median(np.linalg.norm(seen - truth, axis=-1))),
                    "line_px": float(np.median(np.linalg.norm(line - truth, axis=-1))),
                    "mass": float(np.median(mass[ix, read])),
                    "mass_no_ball": float(np.median(none[ix, read])),
                })
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
