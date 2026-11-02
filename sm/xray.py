"""Read the hidden ball out of the model's state, with a linear map fitted after training.

The model is frozen. A ridge regression maps each spatial state token to the 8x8 patch of the next frame rendered with
the occluders removed, so what the map recovers behind an occluder can only have come from what the state remembers.
Used three ways: a curve of decoded position error against time hidden, a decision about the same twins the model's head
judges, and the ghost in the browser demo.

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
from . import surprise as S
from . import world as W
from .evaluate import SCENES, SEED
from .train import load_model

FIT = [(600, 32), (128, 96)]  # clips, frames; all lawful, so a faint memory stays readable
RIDGE = 1.0
MIN_MASS = 1e-3  # with nothing decoded there is no answer; a real ball is about 8 and an empty state is 0
FRACTIONS = [0.25, 0.5, 0.75, 1.0]  # where in the hidden interval position is read
DURATIONS = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48]
TARGETS = ("balls", "next", "cover", "layer")
ABLATE = (4, 16)  # how many directions leave the state; 64 is so many that a random subspace of that size wrecks the head too


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


def targets(traj):
    """The noiseless images the maps are fitted to, in the order of TARGETS: the balls with the slabs taken away, the next
    frame as it is, how much of each pixel a slab covers, and the slab layer's brightness."""
    balls = balls_only(traj)
    cov, lum = W.occluder_cover(traj.scene)
    static = lambda img: np.repeat(img[:, None], balls.shape[1], 1)
    return balls, W.render(traj)[0], static(cov), static(cov * lum)


def fit(model, seed=7):
    """One map per target, each as weight (PDIM, d) and bias (PDIM,). All are linear on one state token, as the head is."""
    rng = np.random.default_rng(seed)
    d = model.cfg.d
    xtx, xty = np.zeros((d + 1, d + 1)), np.zeros((d + 1, len(TARGETS) * M.PDIM))
    for clips, T in FIT:
        for i in range(0, clips, 32):
            traj = W.simulate(W.random_scene(rng, 32), T + 1)
            frames = W.render(traj, rng)[0]
            states = rollout_states(model, frames[:, :-1])[:, 4:]
            target = np.concatenate([np.array(M.patchify(mx.array(img[:, 5:]))) for img in targets(traj)], -1) - 0.25
            x = states.reshape(-1, d).astype(np.float64)
            x = np.concatenate([x, np.ones((len(x), 1))], 1)
            y = target.reshape(-1, target.shape[-1]).astype(np.float64)
            xtx += x.T @ x
            xty += x.T @ y
    sol = np.linalg.solve(xtx + RIDGE * np.eye(d + 1), xty).astype(np.float32)
    return {name: (sol[:-1, i * M.PDIM : (i + 1) * M.PDIM].T, sol[-1, i * M.PDIM : (i + 1) * M.PDIM]) for i, name in enumerate(TARGETS)}


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


def fit_position(model, seed=8, scenes=48):
    """A second linear read, straight to the ball's position, fitted on hidden frames only. It does not depend on decoded
    brightness, so it stays fair to a small model whose decoded image is faint. Ridge chosen on held-out scenes."""
    rng = np.random.default_rng(seed)
    d = model.cfg.d * M.TOKENS
    xs, ys = [], []
    for fn in (E.vanish, E.rebound):
        for D in DURATIONS:
            p = fn(rng, scenes, D)
            states = rollout_states(model, p.lawful)
            for b in range(scenes):
                s0 = p.window[b, 0]
                t = np.arange(s0 - D, s0)
                xs.append(states[b, t].reshape(len(t), d))
                ys.append(p.law.pos[b, t + 1, 0])
    x, y = np.concatenate(xs).astype(np.float64), np.concatenate(ys).astype(np.float64)
    order = rng.permutation(len(x))
    held = order[: len(x) // 5]
    kept = order[len(x) // 5 :]
    mean_x, mean_y = x[kept].mean(0), y[kept].mean(0)
    xc, yc = x[kept] - mean_x, y[kept] - mean_y
    gram, cross = xc.T @ xc, xc.T @ yc
    best = None
    for ridge in (1e1, 1e2, 1e3, 1e4, 1e5):
        w = np.linalg.solve(gram + ridge * np.eye(d), cross)
        err = np.linalg.norm((x[held] - mean_x) @ w + mean_y - y[held], axis=-1).mean()
        if best is None or err < best[0]:
            best = (err, w)
    return best[1], mean_x, mean_y


def read_position(states, position):
    w, mean_x, mean_y = position
    B, Tf, tok, d = states.shape
    return (states.reshape(B, Tf, tok * d) - mean_x) @ w + mean_y


def through_slabs(states, maps):
    """The decoded balls seen through the decoded slabs, so the read's picture of the next frame comes from the state alone."""
    cov = np.clip(decode(states, *maps["cover"]), 0, 1)
    return np.clip(decode(states, *maps["balls"]), 0, 1) * (1 - cov) + np.clip(decode(states, *maps["layer"]), 0, 1)


def roll(model, clip, remove=None, hide=None):
    """One clip through the model: the spatial state after each frame and each frame's cost under the head, and under the
    head's mean alone at the world's fixed noise, which tells a head that lost the ball from one that kept it and stopped
    trusting it. `remove` projects a subspace out of the state on the frames `hide` marks, (T, B)."""
    B, Tf = clip.shape[:2]
    x = M.patchify(mx.array(clip))
    state = model.initial_state(B)
    states, head, plain = [], np.zeros((B, Tf)), np.zeros((B, Tf))
    for t in range(Tf):
        state, mean, sigma, logit = model.step(state, x[:, t * M.TOKENS : (t + 1) * M.TOKENS])
        if t + 1 < Tf:
            target = x[:, (t + 1) * M.TOKENS : (t + 2) * M.TOKENS]
            nll = S.evidence(*M.patch_nll(mean, sigma, logit, target))[0]
            fixed = 0.5 * (((target - mean[..., 0, :]) / W.NOISE) ** 2).sum((-1, -2))
            mx.eval(nll, fixed)
            head[:, t + 1], plain[:, t + 1] = np.array(nll), np.array(fixed)
        if remove is not None:
            state = mx.where(mx.array(hide[t][:, None, None]), state - (state @ remove) @ remove.T, state)
        mx.eval(state)
        states.append(np.array(state[:, : M.TOKENS]))
    return np.stack(states, 1), head, plain


def image_costs(clip, pred):
    """Per frame, the cost of a clip under predicted images and the world's noise. Index t of `pred` predicts frame t + 1."""
    cost = np.zeros(clip.shape[:2])
    cost[:, 1:] = 0.5 * (((clip[:, 1:] - pred[:, :-1]) / W.NOISE) ** 2).sum((-1, -2))
    return cost


def within(cost, window):
    idx = np.arange(cost.shape[1])[None]
    return np.where((idx >= window[:, :1]) & (idx < window[:, 1:]), cost, 0).sum(1)


def stroke(p, position, states):
    """Motion along x while hidden, read against truth, before and after the bounce. A static read has no slope and a
    straight line gets the return's direction wrong."""
    pos = read_position(states, position)
    out = {"outward": ([], []), "return": ([], [])}
    for b in range(len(p.lawful)):
        s0 = p.window[b, 0]
        bounce = np.nonzero(p.law.bounced[b, s0 - p.duration : s0, 0])[0]
        if not len(bounce):
            continue
        hit = s0 - p.duration + bounce[0]
        for name, t0, t1 in (("outward", s0 - p.duration, hit - 1), ("return", hit + 1, s0 - 2)):
            if t1 - t0 >= 3:
                out[name][0].append(pos[b, t1, 0] - pos[b, t0, 0])
                out[name][1].append(p.law.pos[b, t1 + 1, 0, 0] - p.law.pos[b, t0 + 1, 0, 0])
    res = {}
    for name, (read, truth) in out.items():
        read, truth = np.array(read), np.array(truth)
        res[name] = {"sign_agreement": float(np.mean(np.sign(read) == np.sign(truth))), "slope": float((read @ truth) / (truth @ truth)), "scenes": len(read)}
    return res


def pc(d):
    return float(np.mean((d > 0) + 0.5 * (d == 0)))


def subspaces(weight):
    """For each size in ABLATE, the read's strongest directions and as many random ones, each (d, k)."""
    d = weight.shape[1]
    top = np.linalg.svd(weight, full_matrices=False)[2].T
    rand = np.linalg.qr(np.random.default_rng(SEED).normal(size=(d, max(ABLATE))))[0].astype(np.float32)
    return {("read", k): top[:, :k] for k in ABLATE} | {("random", k): rand[:, :k] for k in ABLATE}


def measure(model, maps, position):
    """Everything read from the frozen state, on the scenes `sm.evaluate` scored, against readers with no state: the last seen
    position, a straight line from there (exact until a bounce), the middle of the slab at the last seen height, and the
    same read of a never-there clip. Per-scene margins come back too, so an interval on a difference between readers can
    resample scenes."""
    rows, decisions, strokes, margins, shares = [], [], [], {}, []
    weight, bias = maps["balls"]
    ablate = subspaces(weight)
    for fam, fn in (("vanish", E.vanish), ("rebound", E.rebound)):
        for j, D in enumerate(E.FAMILIES[fam]["levels"]):
            if D < DURATIONS[0]:
                continue
            p = fn(np.random.default_rng([SEED, 2, sorted(E.FAMILIES).index(fam), j]), SCENES, D)
            law, law_head, law_mean = roll(model, p.lawful)
            imp, imp_head, imp_mean = roll(model, p.impossible)
            pos, mass = centroid(decode(law, weight, bias))
            direct = read_position(law, position)
            none = centroid(decode(rollout_states(model, p.control), weight, bias))[1]
            ix = np.arange(SCENES)
            entered = p.window[:, 0] - D - 1  # the last frame with the ball in view
            occ = p.law.scene.occ[ix, 0]
            for frac in FRACTIONS:
                read = entered + np.maximum(np.round(frac * D).astype(int), 1)  # what is read here predicts the next frame
                truth = p.law.pos[ix, read + 1, 0]
                seen = p.law.pos[ix, entered, 0]
                line = seen + p.law.scene.vel[ix, 0] * (read + 1 - entered)[:, None]
                mid = np.stack([(occ[:, 0] + occ[:, 1]) / 2, seen[:, 1]], -1)
                err = np.linalg.norm(pos[ix, read] - truth, axis=-1)
                hit = mass[ix, read] > MIN_MASS
                dist = lambda a: float(np.median(np.linalg.norm(a - truth, axis=-1)))
                rows.append({
                    "family": fam, "D": D, "frac": frac,
                    "err_px": float(np.median(err)), "err_q": np.percentile(err, [25, 75]).tolist(),
                    "hit_rate": float(hit.mean()), "err_given_hit_px": float(np.median(err[hit])) if hit.any() else None,
                    "direct_px": dist(direct[ix, read]),
                    "last_seen_px": dist(seen), "line_px": dist(line), "mid_px": dist(mid),
                    "mass": float(np.median(mass[ix, read])), "mass_no_ball": float(np.median(none[ix, read])),
                })
            readers = {
                "head": (law_head, imp_head),
                "mean": (law_mean, imp_mean),
                "refit": (image_costs(p.lawful, decode(law, *maps["next"])), image_costs(p.impossible, decode(imp, *maps["next"]))),
                "read": (image_costs(p.lawful, through_slabs(law, maps)), image_costs(p.impossible, through_slabs(imp, maps))),
            }
            found = {key + name: within(b, win) - within(a, win) for key, (a, b) in readers.items() for name, win in (("", p.window), ("_full", p.window_full))}
            if D in (8, 16, 24):
                hide = np.stack([(t > entered) & (t < p.window[:, 0]) for t in range(p.lawful.shape[1])])
                for (name, k), q in ablate.items():
                    q = mx.array(q)
                    found[f"head_without_{name}_{k}"] = within(roll(model, p.impossible, q, hide)[1], p.window) - within(roll(model, p.lawful, q, hide)[1], p.window)
                if fam == "vanish" and D == 16:
                    x = law[hide.T].reshape(-1, model.cfg.d)
                    x = x - x.mean(0)
                    shares = [{"directions": name, "k": k, "variance_share": float(((x @ q) ** 2).sum() / (x**2).sum())} for (name, k), q in ablate.items()]
            rec = {"family": fam, "D": D}
            for key, margin in found.items():
                rec[key] = pc(margin)
                margins[f"{fam}/{D}/{key}"] = margin.astype(np.float32)
            decisions.append(rec)
            if fam == "rebound" and D in (16, 28):
                strokes.append({"D": D, **stroke(p, position, law)})
            print(rec, flush=True)
    return {"position": rows, "decisions": decisions, "strokes": strokes, "ablated_variance": shares}, margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    run = Path(a.run)
    model = load_model(run)
    maps = fit(model)
    mx.save_safetensors(str(run / "xray.safetensors"), {"weight": mx.array(maps["balls"][0]), "bias": mx.array(maps["balls"][1])})
    report, margins = measure(model, maps, fit_position(model))
    np.savez_compressed(run / "xray.npz", **margins)
    (run / "xray.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
