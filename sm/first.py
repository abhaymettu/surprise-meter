"""The readers' twin decisions on the first frame where the twins differ, where both are predicted from one state.

Over a reveal window each twin is scored under its own state, so after the first revealed frame a reader that keeps
drawing a ball the frames say is gone still gains, and the read was fitted on lawful clips only. Up to the first frame
that differs the two clips are identical, so there every reader judges both twins from the same prediction. On that frame
a reader at a fixed noise wins exactly when it draws more than half the sliver's brightness on it, so each reader's
brightness on the sliver and the share of its predicted change that lands on the sliver are kept apart.

Also two controls for the ablation in `sm.xray`, whose directions are the largest of the read's weights. The same number
of directions chosen the same way from the map to slab cover, and as many random directions drawn from the low end of the
state's spectrum so that they carry the read's share of its variance.

The window model has no maps to read, so only its head and the observer are scored, as a known memory length.

    python -m sm.first runs/5m-s1
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import events as E
from . import model as M
from . import surprise as S
from . import world as W
from .evaluate import SCENES, SEED
from .train import load_model
from .xray import DURATIONS, decode, fit, image_costs, pc, roll, subspaces, through_slabs, within

K = 4
CONTROLS = ("cover_4", "matched_4")


def head_means(model, clip):
    """The head's mean image of each next frame, (B, T, H, W). Index t predicts frame t + 1."""
    B, Tf = clip.shape[:2]
    x = M.patchify(mx.array(clip))
    state, out = model.initial_state(B), []
    for t in range(Tf):
        state, mean, _, _ = model.step(state, x[:, t * M.TOKENS : (t + 1) * M.TOKENS])
        out.append(mean[..., 0, :])
        mx.eval(state, out[-1])
    return np.array(M.unpatchify(mx.stack(out, 1).reshape(B, Tf * M.TOKENS, M.PDIM), Tf))


def window_roll(model, clip):
    """`roll` for the window model, which sees the clip at once and has no state to read: per frame, the cost under the
    head and under its mean at the world's noise, and the mean image of each next frame."""
    B, Tf = clip.shape[:2]
    x = M.patchify(mx.array(clip))
    mean, sigma, logit = model(mx.array(clip[:, :-1]))
    head, plain = np.zeros((B, Tf)), np.zeros((B, Tf))
    for t in range(1, Tf):
        target = x[:, t * M.TOKENS : (t + 1) * M.TOKENS]
        nll = S.evidence(*M.patch_nll(mean[:, t - 1], sigma[:, t - 1], logit[:, t - 1], target))[0]
        fixed = 0.5 * (((target - mean[:, t - 1][..., 0, :]) / W.NOISE) ** 2).sum((-1, -2))
        mx.eval(nll, fixed)
        head[:, t], plain[:, t] = np.array(nll), np.array(fixed)
    images = np.array(M.unpatchify(mean[..., 0, :].reshape(B, (Tf - 1) * M.TOKENS, M.PDIM), Tf - 1))
    return head, plain, np.concatenate([images, images[:, -1:]], 1)


def measure(model, maps, controls):
    decisions, margins = [], {}
    for fam, fn in (("vanish", E.vanish), ("rebound", E.rebound)):
        for j, D in enumerate(E.FAMILIES[fam]["levels"]):
            if D < DURATIONS[0]:
                continue
            p = fn(np.random.default_rng([SEED, 2, sorted(E.FAMILIES).index(fam), j]), SCENES, D)
            w0, ix = p.window[:, 0], np.arange(SCENES)
            f = (p.lawful != p.impossible).any((-1, -2)).argmax(1)  # a frame or so before w0 in a few scenes, by about 0.001 of edge
            T = int(f.max()) + 1
            if maps:
                law, law_head, law_mean = roll(model, p.lawful[:, :T])
                imp, imp_head, imp_mean = roll(model, p.impossible[:, :T])
                assert np.array_equal(law[ix, f - 1], imp[ix, f - 1]), "the state before the first differing frame is not shared"
                pictures = {"mean": head_means(model, p.lawful[:, :T])}
            else:
                law_head, law_mean, law_picture = window_roll(model, p.lawful[:, :T])
                imp_head, imp_mean, _ = window_roll(model, p.impossible[:, :T])
                pictures = {"mean": law_picture}
            clean, empty = W.render(p.law)[0][ix, f], W.render(p.imp)[0][ix, f]
            ideal = [0.5 * (((clip[ix, f] - clean) / W.NOISE) ** 2).sum((-1, -2)) for clip in (p.lawful, p.impossible)]
            sliver = clean - empty
            readers = {"head": (law_head, imp_head), "mean": (law_mean, imp_mean)}
            if maps:
                draws = {"refit": lambda s: decode(s, *maps["next"]), "visible": lambda s: through_slabs(s, maps, "visible"), "read": lambda s: through_slabs(s, maps)}
                for k, draw in draws.items():
                    pictures[k] = draw(law)
                    readers[k] = (image_costs(p.lawful[:, :T], pictures[k]), image_costs(p.impossible[:, :T], draw(imp)))
            first = np.stack([f, f + 1], 1)
            found = {f"{k}_first": within(b, first) - within(a, first) for k, (a, b) in readers.items()}
            found["ideal_first"] = ideal[1] - ideal[0]
            if controls and D in (8, 16, 24):
                entered = w0 - D - 1
                hide = np.stack([(t > entered) & (t < w0) for t in range(p.lawful.shape[1])])
                for name, q in controls.items():
                    q = mx.array(q)
                    found[f"head_without_{name}"] = within(roll(model, p.impossible, q, hide)[1], p.window) - within(roll(model, p.lawful, q, hide)[1], p.window)
            rec = {"family": fam, "D": D}
            # gain: how bright each fixed-noise reader draws the sliver of ball the lawful twin shows, against the truth, where 1
            # is exact and above 1 wins more often than the observer told the lawful frame. cos: the share of its predicted
            # change from the empty frame that lies along the sliver, whatever its brightness
            for k, pred in pictures.items():
                change = pred[ix, f - 1] - empty
                along = (sliver * change).sum((-1, -2))
                rec[f"{k}_gain"] = float(np.median(along / (sliver**2).sum((-1, -2))))
                rec[f"{k}_cos"] = float(np.median(along / np.sqrt((sliver**2).sum((-1, -2)) * (change**2).sum((-1, -2)) + 1e-12)))
            for key, margin in found.items():
                rec[key] = pc(margin)
                margins[f"{fam}/{D}/{key}"] = margin.astype(np.float32)
            decisions.append(rec)
            print(rec, flush=True)
    return decisions, margins


def hidden_states(model):
    """The state while a ball is hidden, centred, on the frames and scenes `sm.xray` uses for its variance shares."""
    p = E.vanish(np.random.default_rng([SEED, 2, sorted(E.FAMILIES).index("vanish"), E.FAMILIES["vanish"]["levels"].index(16)]), SCENES, 16)
    law = roll(model, p.lawful)[0]
    hide = np.stack([(t > p.window[:, 0] - 17) & (t < p.window[:, 0]) for t in range(p.lawful.shape[1])])
    x = law[hide.T].reshape(-1, model.cfg.d).astype(np.float64)
    return x - x.mean(0)


def share(x, q):
    return float(((x @ q) ** 2).sum() / (x**2).sum())


def matched(x, target):
    """K random directions in the span of the state's weakest principal directions, with as many of them as makes the
    expected share of variance of K random directions there closest to `target`."""
    vals, vecs = np.linalg.eigh(x.T @ x)
    expected = K * np.cumsum(vals) / (x**2).sum() / np.arange(1, len(vals) + 1)
    m = K + int(np.argmin(np.abs(expected[K - 1 :] - target)))
    rot = np.linalg.qr(np.random.default_rng(SEED).normal(size=(m, K)))[0]
    return (vecs[:, :m] @ rot).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    run = Path(a.run)
    model = load_model(run)
    if model.cfg.arch == "window":
        decisions, margins = measure(model, None, None)
        shares = {}
    else:
        maps = fit(model)
        saved = mx.load(str(run / "xray.safetensors"))
        assert np.allclose(np.array(saved["weight"]), maps["balls"][0], atol=1e-4), "the maps refit here differ from the ones sm.xray used"
        x = hidden_states(model)
        read = subspaces(maps["balls"][0])[("read", K)]
        controls = {"cover_4": subspaces(maps["cover"][0])[("read", K)]}
        controls["matched_4"] = matched(x, share(x, read))
        shares = {"read_4": share(x, read)} | {k: share(x, q) for k, q in controls.items()}
        decisions, margins = measure(model, maps, controls)
    np.savez_compressed(run / "first.npz", **margins)
    (run / "first.json").write_text(json.dumps({"decisions": decisions, "variance_share": shares}, indent=1))


if __name__ == "__main__":
    main()
