"""The readers' twin decisions on the first frame where the twins differ, where both are predicted from one state.

Over a reveal window each twin is scored under its own state, so after the first revealed frame a reader that keeps
drawing a ball the frames say is gone still gains, and the read was fitted on lawful clips only. Up to the first frame
that differs the two clips are identical, so there every reader judges both twins from the same prediction.

Also a control for the ablation in `sm.xray`: the read's directions are chosen by the size of its weights, which favours
directions of low variance, so the same number of directions are chosen the same way from the map to slab cover, a probe
for something other than the ball, and projected out while the ball is hidden.

    python -m sm.first runs/5m-s1
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import events as E
from . import model as M
from . import world as W
from .evaluate import SCENES, SEED
from .train import load_model
from .xray import DURATIONS, decode, fit, image_costs, pc, roll, subspaces, through_slabs, within

CONTROL = ("cover", 4)


def measure(model, maps):
    decisions, margins = [], {}
    cover = mx.array(subspaces(maps["cover"][0])[("read", CONTROL[1])])
    for fam, fn in (("vanish", E.vanish), ("rebound", E.rebound)):
        for j, D in enumerate(E.FAMILIES[fam]["levels"]):
            if D < DURATIONS[0]:
                continue
            p = fn(np.random.default_rng([SEED, 2, sorted(E.FAMILIES).index(fam), j]), SCENES, D)
            w0, ix = p.window[:, 0], np.arange(SCENES)
            f = (p.lawful != p.impossible).any((-1, -2)).argmax(1)  # a frame or so before w0 in a few scenes, by about 0.001 of edge
            T = int(f.max()) + 1
            law, law_head, law_mean = roll(model, p.lawful[:, :T])
            imp, imp_head, imp_mean = roll(model, p.impossible[:, :T])
            assert np.array_equal(law[ix, f - 1], imp[ix, f - 1]), "the state before the first differing frame is not shared"
            clean, empty = W.render(p.law)[0][ix, f], W.render(p.imp)[0][ix, f]
            ideal = [0.5 * (((clip[ix, f] - clean) / W.NOISE) ** 2).sum((-1, -2)) for clip in (p.lawful, p.impossible)]
            sliver = clean - empty
            readers = {
                "head": (law_head, imp_head),
                "mean": (law_mean, imp_mean),
                "refit": (image_costs(p.lawful[:, :T], decode(law, *maps["next"])), image_costs(p.impossible[:, :T], decode(imp, *maps["next"]))),
                "visible": (image_costs(p.lawful[:, :T], through_slabs(law, maps, "visible")), image_costs(p.impossible[:, :T], through_slabs(imp, maps, "visible"))),
                "read": (image_costs(p.lawful[:, :T], through_slabs(law, maps)), image_costs(p.impossible[:, :T], through_slabs(imp, maps))),
            }
            first = np.stack([f, f + 1], 1)
            found = {f"{k}_first": within(b, first) - within(a, first) for k, (a, b) in readers.items()}
            found["ideal_first"] = ideal[1] - ideal[0]
            if D in (8, 16, 24):
                entered = w0 - D - 1
                hide = np.stack([(t > entered) & (t < w0) for t in range(p.lawful.shape[1])])
                name = "head_without_%s_%d" % CONTROL
                found[name] = within(roll(model, p.impossible, cover, hide)[1], p.window) - within(roll(model, p.lawful, cover, hide)[1], p.window)
            rec = {"family": fam, "D": D}
            # how bright each image reader draws the sliver of ball the lawful twin shows, against the truth: 1 is exact, and
            # above 1 wins this decision more often than the ideal observer can
            for k, pred in (("refit", decode(law, *maps["next"])), ("visible", through_slabs(law, maps, "visible")), ("read", through_slabs(law, maps))):
                rec[f"{k}_gain"] = float(np.median((sliver * (pred[ix, f - 1] - empty)).sum((-1, -2)) / (sliver**2).sum((-1, -2))))
            for key, margin in found.items():
                rec[key] = pc(margin)
                margins[f"{fam}/{D}/{key}"] = margin.astype(np.float32)
            decisions.append(rec)
            print(rec, flush=True)
    return decisions, margins


def share(model, maps):
    """The share of the hidden state's variance in the control's directions, on the frames and scenes `sm.xray` uses."""
    p = E.vanish(np.random.default_rng([SEED, 2, sorted(E.FAMILIES).index("vanish"), E.FAMILIES["vanish"]["levels"].index(16)]), SCENES, 16)
    law = roll(model, p.lawful)[0]
    hide = np.stack([(t > p.window[:, 0] - 17) & (t < p.window[:, 0]) for t in range(p.lawful.shape[1])])
    x = law[hide.T].reshape(-1, model.cfg.d)
    x = x - x.mean(0)
    q = subspaces(maps["cover"][0])[("read", CONTROL[1])]
    return float(((x @ q) ** 2).sum() / (x**2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    run = Path(a.run)
    model = load_model(run)
    maps = fit(model)
    saved = mx.load(str(run / "xray.safetensors"))
    assert np.allclose(np.array(saved["weight"]), maps["balls"][0], atol=1e-4), "the maps refit here differ from the ones sm.xray used"
    decisions, margins = measure(model, maps)
    np.savez_compressed(run / "first.npz", **margins)
    (run / "first.json").write_text(json.dumps({"decisions": decisions, "control_variance_share": share(model, maps)}, indent=1))


if __name__ == "__main__":
    main()
