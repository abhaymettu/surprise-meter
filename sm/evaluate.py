"""Measure one observer (a trained run, or ideal / cv / cv0) on every family and magnitude.

    python -m sm.evaluate runs/1m-s0          # writes runs/1m-s0/eval.npz
    python -m sm.evaluate ideal --out runs/observers/ideal

Every observer sees the same scenes and the same noise, so the differences between them are paired.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import events as E
from . import model as M
from . import observers as O
from . import surprise as S
from . import world as W
from .train import load_model

SCENES = 192
NULL_CLIPS, NULL_T = 2000, 96
ALPHAS = [1e-2, 1e-3, 1e-4]


class ModelObserver:
    def __init__(self, model):
        self.model = model
        if isinstance(model, M.RecurrentModel):
            def step(state, patches, target):
                state, mean, sigma, logit = model.step(state, patches)
                nll, ev = S.evidence(*M.patch_nll(mean, sigma, logit, target))
                return state, nll, ev

            self._step = mx.compile(step)

    def _roll(self, frames, state, t0):
        """Score frames t0.. given `state`, which has seen frames up to t0 - 2. Returns (B, T) arrays, NaN before t0."""
        B, T = frames.shape[:2]
        nll, ev = np.full((B, T), np.nan, np.float32), np.full((B, T), np.nan, np.float32)
        x = mx.array(frames)
        p = M.patchify(x)
        for t in range(t0, T):
            state, n, m = self._step(state, p[:, (t - 1) * M.TOKENS : t * M.TOKENS], p[:, t * M.TOKENS : (t + 1) * M.TOKENS])
            mx.eval(state, n, m)
            nll[:, t], ev[:, t] = np.array(n), np.array(m)
        return nll, ev

    def score(self, frames):
        """(B, T, H, W) -> surprisal and evidence (B, T); frame 0 has no prediction."""
        out = [self._score(frames[i : i + 128]) for i in range(0, len(frames), 128)]
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

    def _score(self, frames):
        if not isinstance(self.model, M.RecurrentModel):
            x = mx.array(frames)
            mean, sigma, logit = self.model(x[:, :-1])
            p = M.patchify(x)
            B, T = frames.shape[:2]
            nll, ev = np.full((B, T), np.nan, np.float32), np.full((B, T), np.nan, np.float32)
            for t in range(1, T):
                target = p[:, t * M.TOKENS : (t + 1) * M.TOKENS]
                n, m = S.evidence(*M.patch_nll(mean[:, t - 1], sigma[:, t - 1], logit[:, t - 1], target))
                mx.eval(n, m)
                nll[:, t], ev[:, t] = np.array(n), np.array(m)
            return nll, ev
        return self._roll(frames, self.model.initial_state(len(frames)), 1)

    def twins(self, p):
        """Both twins, sharing the rollout of their common prefix where the architecture allows it."""
        if not isinstance(self.model, M.RecurrentModel):
            return self.score(p.lawful) + self.score(p.impossible)
        B, T = p.lawful.shape[:2]
        out = np.full((4, B, T), np.nan, np.float32)
        order = np.argsort(p.window[:, 0])
        for i in range(0, B, 64):
            rows = order[i : i + 64]
            fork = max(int(p.window[rows, 0].min()), 1)
            state = self.model.initial_state(len(rows))
            head = p.lawful[rows]
            if fork > 1:
                x = mx.array(head[:, : fork - 1])
                pp = M.patchify(x)
                for t in range(fork - 1):
                    state = self.model.step(state, pp[:, t * M.TOKENS : (t + 1) * M.TOKENS])[0]
                    mx.eval(state)
            out[0, rows], out[1, rows] = self._roll(p.lawful[rows], state, fork)
            out[2, rows], out[3, rows] = self._roll(p.impossible[rows], state, fork)
        return tuple(out)


class ReferenceObserver:
    def __init__(self, kind):
        self.kind = kind

    def _traj(self, traj, frames, truth=None):
        """Score frames against a prediction: the given lawful clean frames for the ideal, else a tracker's guess."""
        clean, vis = W.render(traj if truth is None else truth)
        pred = clean if self.kind == "ideal" else O.predict(traj, vis, self.kind)
        B, T = frames.shape[:2]
        nll, ev = np.full((B, T), np.nan, np.float32), np.full((B, T), np.nan, np.float32)
        sigma = np.full((B, W.SIZE, W.SIZE), W.NOISE, np.float32)
        for t in range(1, T):
            nll[:, t], ev[:, t] = S.evidence(*S.gaussian_patches(frames[:, t], pred[:, t], sigma))
        return nll, ev

    def twins(self, p):
        return self._traj(p.law, p.lawful) + self._traj(p.imp, p.impossible, truth=p.law)

    def null(self, traj, frames):
        return self._traj(traj, frames)


def window_sum(a, win):
    idx = np.arange(a.shape[1])[None]
    m = (idx >= win[:, :1]) & (idx < win[:, 1:])
    return np.where(m, a, 0).sum(1)


def window_max(a, win):
    idx = np.arange(a.shape[1])[None]
    m = (idx >= win[:, :1]) & (idx < win[:, 1:])
    return np.where(m, a, -np.inf).max(1)


def null_frames(observer):
    """Held-out lawful evidence, and a tag per frame for what was happening. Fixed scenes for every observer."""
    rng = np.random.default_rng([2026, 11, 1])
    ev, tags = [], []
    for i in range(0, NULL_CLIPS, 128):
        traj = W.simulate(W.random_scene(rng, 128), NULL_T)
        frames, vis = W.render(traj, rng)
        m = observer.null(traj, frames)[1] if isinstance(observer, ReferenceObserver) else observer.score(frames)[1]
        contact = traj.bounced.any(-1)
        edge = ((vis > 0.02) & (vis < 0.98) & traj.alive).any(-1)
        reveal = np.zeros_like(contact)
        reveal[:, 1:] = (((vis[:, 1:] > 0.02) & (vis[:, :-1] <= 0.02)) & traj.alive[:, 1:]).any(-1)
        ev.append(m[:, S.BURN_IN :])
        tags.append(np.stack([contact, edge, reveal], -1)[:, S.BURN_IN :])
    return np.concatenate(ev), np.concatenate(tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("who")
    ap.add_argument("--out", default=None)
    ap.add_argument("--families", default=None)
    a = ap.parse_args()

    if a.who in ("ideal", *O.KINDS):
        observer = ReferenceObserver(a.who)
        out = Path(a.out or f"runs/observers/{a.who}")
    else:
        observer = ModelObserver(load_model(a.who))
        out = Path(a.out or a.who)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ev, tags = null_frames(observer)
    half = len(ev) // 2
    calib = S.Null(ev[:half].ravel())
    res = {"null_sorted": calib.sorted.astype(np.float32)}
    test_p = calib.p(ev[half:].ravel())
    report = {"null_frames": int(ev.size), "observer": a.who, "alphas": ALPHAS}
    report["fa_test"] = {str(al): float((test_p <= al).mean()) for al in ALPHAS}
    t = tags[half:].reshape(-1, 3)
    for name, m in (("contact", t[:, 0]), ("edge", t[:, 1]), ("reveal", t[:, 2]), ("quiet", ~t.any(-1))):
        report[f"fa_{name}"] = {str(al): float((test_p[m] <= al).mean()) for al in ALPHAS}
        report[f"n_{name}"] = int(m.sum())
    print(f"null {ev.size} frames, cut@1e-3 {calib.cut(1e-3):.1f} ({time.time() - t0:.0f}s)", flush=True)

    fams = a.families.split(",") if a.families else list(E.FAMILIES)
    for fam in fams:
        spec = E.FAMILIES[fam]
        for j, level in enumerate(spec["levels"]):
            rng = np.random.default_rng([2026, sorted(E.FAMILIES).index(fam), j])
            p = spec["fn"](rng, SCENES, level)
            nl, el, ni, ei = observer.twins(p)
            key = f"{fam}/{level}"
            res[f"{key}/law_nll"] = window_sum(nl, p.window)
            res[f"{key}/imp_nll"] = window_sum(ni, p.window)
            res[f"{key}/law_ev"] = window_max(el, p.window)
            res[f"{key}/imp_ev"] = window_max(ei, p.window)
            res[f"{key}/frames"] = p.window[:, 1] - p.window[:, 0]
            if p.duration and isinstance(observer, ModelObserver):
                cn, ce = observer.score(p.control)
                w = p.window[: len(p.control)]
                res[f"{key}/ctrl_nll"] = window_sum(cn, w)
                res[f"{key}/ctrl_ev"] = window_max(ce, w)
            pc = np.mean((res[f"{key}/imp_nll"] > res[f"{key}/law_nll"]) + 0.5 * (res[f"{key}/imp_nll"] == res[f"{key}/law_nll"]))
            print(f"{key:24s} pc={pc:.3f} lbf={np.median(res[f'{key}/imp_nll'] - res[f'{key}/law_nll']):9.1f} ({time.time() - t0:.0f}s)", flush=True)

    np.savez_compressed(out / "eval.npz", **res)
    (out / "eval.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    print(f"done {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
