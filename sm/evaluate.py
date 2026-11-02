"""Measure one observer (a trained run, or ideal / cv / cv0) on every family and magnitude.

    python -m sm.evaluate runs/1m-s0          # writes runs/1m-s0/eval.npz
    python -m sm.evaluate ideal               # or causal, walls, cv, cv0, frozen; writes runs/observers/<kind>/eval.npz

Every observer sees the same scenes and the same noise, so the differences between them are paired.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import events as E
from . import model as M
from . import observers as O
from . import surprise as S
from . import world as W
from .train import load_model

SEED = 20261102  # the design was chosen on other scenes; everything reported comes from these
SCENES = 192
NULL_CLIPS, NULL_T = 10000, 96
LONG_CLIPS, LONG_T = 512, 1200  # does the table still hold far past the 64-frame clips the model trained on?
LONG_BINS = [(4, 96), (96, 200), (200, 400), (400, 800), (800, 1200)]
ALPHAS = [1e-2, 1e-3, 1e-4]


class ModelObserver:
    def __init__(self, model):
        self.model = model
        self.chunk = 128 if isinstance(model, M.RecurrentModel) else 16  # window attention is clips x (frames x tokens)^2
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
        out = [self._score(frames[i : i + self.chunk]) for i in range(0, len(frames), self.chunk)]
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
        order = np.argsort(p.fork)
        for i in range(0, B, 64):
            rows = order[i : i + 64]
            fork = max(int(p.fork[rows].min()), 1)
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
        """Score frames against a prediction: the lawful clean frames for the ideal, else a tracker of what is on screen."""
        if self.kind == "ideal":
            pred = W.render(traj if truth is None else truth)[0]
        else:
            pred = O.predict(traj, W.render(traj)[1], self.kind)
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


class CausalIdeal:
    """Knows the laws, the scene, radii and brightness, but not where the balls started or how fast. Under a flat prior the
    frames' likelihood, marginalized over those four numbers per ball, is Gaussian about the lawful clip with its first
    order change in them projected out. The twins share a prefix, so the clip's total is the same as the sequential
    predictive score over the event. Returns the difference of the twins' costs in nats, per clip."""

    kind = "causal"
    EPS = 0.02

    def clip_scores(self, p, window):
        out = np.zeros(len(p.lawful))
        for i in range(0, len(p.lawful), 32):
            rows = np.arange(i, min(i + 32, len(p.lawful)))
            scene = E._subset(p.law.scene, rows)
            T = p.lawful.shape[1]
            base = W.render(W.simulate(scene, T))[0]
            jac = []
            for ball in range(scene.pos.shape[1]):
                for field in ("pos", "vel"):
                    for axis in range(2):
                        v = getattr(scene, field).copy()
                        v[:, ball, axis] += self.EPS
                        jac.append((W.render(W.simulate(replace(scene, **{field: v}), T))[0] - base) / self.EPS)
            jac = np.stack(jac, 1)  # (b, params, T, H, W)
            for k, row in enumerate(rows):
                end = window[row, 1]
                j = jac[k, :, :end].reshape(jac.shape[1], -1).astype(np.float64)
                gram = j @ j.T + 1e-6 * np.eye(len(j))
                energy = []
                for clip in (p.lawful, p.impossible):
                    r = (clip[row, :end] - base[k, :end]).ravel().astype(np.float64)
                    c = j @ r
                    energy.append(r @ r - c @ np.linalg.solve(gram, c))
                out[row] = (energy[1] - energy[0]) / (2 * W.NOISE**2)
        return out


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
    rng = np.random.default_rng([SEED, 1])
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


def long_calibration(observer, ev_cuts):
    """False alarm rate by time since the start of a very long lawful clip, at the rates the table promises."""
    rng = np.random.default_rng([SEED, 3])
    ms = []
    for i in range(0, LONG_CLIPS, 64):
        traj = W.simulate(W.random_scene(rng, 64), LONG_T)
        ms.append(observer.score(W.render(traj, rng)[0])[1])
    m = np.concatenate(ms)
    return {f"{a}-{b}": {str(al): float((m[:, a:b] >= c).mean()) for al, c in ev_cuts.items()} for a, b in LONG_BINS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("who")
    ap.add_argument("--out", default=None)
    ap.add_argument("--families", default=None, help="comma separated; with an existing eval.npz, replaces just these")
    a = ap.parse_args()

    if a.who == "causal":
        observer = CausalIdeal()
        out = Path(a.out or "runs/observers/causal")
    elif a.who in ("ideal", *O.KINDS):
        observer = ReferenceObserver(a.who)
        out = Path(a.out or f"runs/observers/{a.who}")
    else:
        observer = ModelObserver(load_model(a.who))
        out = Path(a.out or a.who)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if a.who == "causal":  # scored on the twins only; it has no frame by frame evidence to calibrate
        res = {"null_ranks": np.array([1]), "null_tail": np.array([np.inf]), "null_n": 1}
        report = {"observer": a.who}
    elif a.families and (out / "eval.npz").exists():  # measure these families again and keep everything else
        res = dict(np.load(out / "eval.npz"))
        report = json.loads((out / "eval.json").read_text())
    else:
        ev, tags = null_frames(observer)
        half = len(ev) // 2
        calib = S.Null(ev[:half].ravel())
        res = dict(zip(("null_ranks", "null_tail"), calib.tail()))
        res["null_n"] = len(calib.sorted)
        test_p = calib.p(ev[half:].ravel())
        report = {"null_frames": int(ev.size), "observer": a.who, "alphas": ALPHAS}
        report["fa_test"] = {str(al): float((test_p <= al).mean()) for al in ALPHAS}
        t = tags[half:].reshape(-1, 3)
        c = tags[:half].reshape(-1, 3)
        for i, (name, m) in enumerate((("contact", t[:, 0]), ("edge", t[:, 1]), ("reveal", t[:, 2]), ("quiet", ~t.any(-1)))):
            report[f"fa_{name}"] = {str(al): float((test_p[m] <= al).mean()) for al in ALPHAS}
            report[f"n_{name}"] = int(m.sum())
            own = c[:, i] if i < 3 else ~c.any(-1)
            res[f"null_ranks_{name}"], res[f"null_tail_{name}"] = S.Null(ev[:half].ravel()[own]).tail(64)
        if isinstance(observer, ModelObserver) and isinstance(observer.model, M.RecurrentModel):
            report["long_rollout"] = long_calibration(observer, {al: calib.cut(al) for al in ALPHAS})
        print(f"null {ev.size} frames, cut@1e-3 {calib.cut(1e-3):.1f} ({time.time() - t0:.0f}s)", flush=True)

    fams = a.families.split(",") if a.families else list(E.FAMILIES)
    for fam in fams:
        spec = E.FAMILIES[fam]
        for j, level in enumerate(spec["levels"]):
            rng = np.random.default_rng([SEED, 2, sorted(E.FAMILIES).index(fam), j])
            p = spec["fn"](rng, SCENES, level)
            key = f"{fam}/{level}"
            if a.who == "causal":
                res[f"{key}/law_nll"] = np.zeros(SCENES)
                res[f"{key}/imp_nll"] = observer.clip_scores(p, p.window)
                res[f"{key}/law_ev"] = res[f"{key}/imp_ev"] = np.full(SCENES, -np.inf)
                res[f"{key}/frames"] = p.window[:, 1] - p.window[:, 0]
                res[f"{key}/speed"] = np.linalg.norm(p.law.scene.vel[:, 0], axis=-1)
                if p.duration:
                    res[f"{key}/law_nll_full"] = res[f"{key}/law_nll"]
                    res[f"{key}/imp_nll_full"] = observer.clip_scores(p, p.window_full)
                continue
            nl, el, ni, ei = observer.twins(p)
            res[f"{key}/law_nll"] = window_sum(nl, p.window)
            res[f"{key}/imp_nll"] = window_sum(ni, p.window)
            res[f"{key}/law_ev"] = window_max(el, p.window)
            res[f"{key}/imp_ev"] = window_max(ei, p.window)
            res[f"{key}/frames"] = p.window[:, 1] - p.window[:, 0]
            res[f"{key}/speed"] = np.linalg.norm(p.law.scene.vel[:, 0], axis=-1)
            if p.sign is not None:
                res[f"{key}/sign"] = p.sign
            if p.duration:
                res[f"{key}/law_nll_full"] = window_sum(nl, p.window_full)
                res[f"{key}/imp_nll_full"] = window_sum(ni, p.window_full)
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
