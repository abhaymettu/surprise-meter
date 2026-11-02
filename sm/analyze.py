"""Turn eval.npz files into thresholds, efficiencies and intervals. Writes results/summary.json.

    python -m sm.analyze

Each size has several seeds and every seed saw the same scenes. Intervals are a hierarchical bootstrap: resample
seeds, then resample scenes, so they carry both model uncertainty and stimulus uncertainty.
"""

import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import events as E

RUNS = Path("runs")
SIZES = ["1m", "5m", "20m"]
CONTROLS = ["1m-window8", "1m-long", "5m-short"]  # known memory length; three times the steps at 1M; half the steps at 5M
OBSERVERS = ["ideal", "causal", "walls", "cv", "cv0", "frozen"]
KINEMATIC = ["teleport", "deflect", "phantom", "collision", "hidden"]
DURATION = ["vanish", "rebound"]
BANDS = [(0.15, 0.4), (0.4, 0.6), (0.6, 1.3)]  # ball speed, px per frame; long hidden times come with slow balls
ALPHA = 1e-3
BOOT = 500
TRAIN_REVEAL_MAX = 41  # the longest hidden run seen in a training clip; longer durations are extrapolation
EXPLORATORY = {"5m-s0"}  # watched while this analysis was built, so it leaves its group once a confirmatory seed exists
READERS = ("head", "mean", "refit", "visible", "read")
ABLATED = [f"head_without_{name}_{k}" for k in (4, 16) for name in ("read", "random")]


def load(path):
    d = np.load(path / "eval.npz")
    return {k: d[k] for k in d.files}


def seed_dirs(group):
    """Measured run directories of one size or control, in seed order. Experiments with other names are not picked up."""
    found = [p for p in RUNS.iterdir() if re.fullmatch(rf"{re.escape(group)}-s(\d+)", p.name) and (p / "eval.npz").exists()]
    found.sort(key=lambda p: int(p.name.rsplit("-s", 1)[1]))
    return [p for p in found if p.name not in EXPLORATORY] or found


def cut(ev, alpha=ALPHA):
    return float(np.interp(np.log(alpha * ev["null_n"]), np.log(ev["null_ranks"]), ev["null_tail"]))


def curves(ev, fam, levels, c, idx=None, window=""):
    """Per level: proportion correct in 2AFC, hit and false-alarm rates at the criterion, median log Bayes factor."""
    pc, hit, fa, lbf = [], [], [], []
    for lv in levels:
        k = f"{fam}/{lv}"
        i = idx if idx is not None else np.arange(len(ev[f"{k}/imp_nll"]))
        d = ev[f"{k}/imp_nll{window}"][i] - ev[f"{k}/law_nll{window}"][i]
        pc.append(np.mean((d > 0) + 0.5 * (d == 0)))
        hit.append(np.mean(ev[f"{k}/imp_ev"][i] >= c))
        fa.append(np.mean(ev[f"{k}/law_ev"][i] >= c))
        lbf.append(np.median(d))
    return np.array(pc), np.array(hit), np.array(fa), np.array(lbf)


def trace(ev, fam, levels, idx=None):
    """Median surprisal per frame at identical empty frames, ball seen earlier minus ball never there. Per frame because
    the scored window grows as balls slow."""
    out = []
    for lv in levels:
        d = (ev[f"{fam}/{lv}/imp_nll"] - ev[f"{fam}/{lv}/ctrl_nll"]) / ev[f"{fam}/{lv}/frames"]
        out.append(np.median(d if idx is None else d[idx]))
    return np.array(out)


def fit_threshold(x, y, n, floor, target, start=None):
    """Binomial ML logistic in log magnitude with a small lapse. Magnitude reaching `target`, or an infinity or zero when
    the data never bracket it. `start` skips the search over starting points. With floor None the floor is fitted: under
    matched noise a directional bias makes proportion correct tend to a constant other than 0.5 as magnitude falls."""
    x, y = np.log(np.asarray(x, float)), np.asarray(y, float)
    if y.max() < target:
        return float("inf"), None
    if y.min() >= target:
        return 0.0, None
    lapse = 0.01

    def bottom(th):
        return floor if floor is not None else 0.74 / (1 + np.exp(-th[2]))

    def nll(th):
        mu, ls = th[:2]
        p = bottom(th) + (1 - bottom(th) - lapse) / (1 + np.exp(-(x - mu) / np.exp(np.clip(ls, -6, 6))))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(n * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum()

    extra = [] if floor is not None else [0.0]
    starts = [start] if start is not None else [[m, s, *extra] for m in np.quantile(x, [0.2, 0.5, 0.8]) for s in (-1.0, 0.0)]
    best = None
    for th in starts:
        r = minimize(nll, th, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3})
        if best is None or r.fun < best.fun:
            best = r
    mu, ls = best.x[:2]
    base = bottom(best.x)
    q = np.clip((target - base) / (1 - base - lapse), 1e-4, 1 - 1e-4)
    return float(np.exp(mu + np.exp(ls) * np.log(q / (1 - q)))), best.x


def thresholds(ev, fam, levels, c, idx=None, starts=(None, None)):
    n = len(ev[f"{fam}/{levels[0]}/imp_nll"]) if idx is None else len(idx)
    pc, hit, fa, _ = curves(ev, fam, levels, c, idx)
    t_afc, s_afc = fit_threshold(levels, pc, n, None, 0.75, starts[0])
    floor = float(np.clip(fa.mean(), 0.0, 0.5))
    t_det, s_det = fit_threshold(levels, hit, n, floor, floor + 0.5 * (1 - floor), starts[1])
    return t_afc, t_det, (s_afc, s_det)


def falls_through(y, levels, value):
    """Where a falling curve first drops through `value`, interpolated. 0 if it starts below, the last level if it never does."""
    if y[0] < value:
        return 0.0
    for i in range(1, len(y)):
        if y[i] < value:
            return float(levels[i - 1] + (levels[i] - levels[i - 1]) * (y[i - 1] - value) / (y[i - 1] - y[i]))
    return float(levels[-1])


def geometric(a):
    """Geometric mean of thresholds; a seed that never reaches threshold makes it infinite."""
    a = np.asarray(a, float)
    return float("inf") if not np.isfinite(a).all() else float(np.exp(np.mean(np.log(np.maximum(a, 1e-9)))))


def by_speed(evs, fam, levels):
    """2AFC and its crossing within fixed bands of ball speed, since a pooled crossing mixes them."""
    out = []
    for lo, hi in BANDS:
        pc, kept = [], []
        for lv in levels:
            k = f"{fam}/{lv}"
            vals = []
            for ev in evs:
                d, v = ev[f"{k}/imp_nll"] - ev[f"{k}/law_nll"], ev[f"{k}/speed"]
                if ((v >= lo) & (v < hi)).sum() >= 12:
                    vals.append(np.mean(d[(v >= lo) & (v < hi)] > 0))
            if vals:
                pc.append(float(np.mean(vals)))
                kept.append(lv)
        out.append({"band": [lo, hi], "levels": kept, "pc": pc, "crossing": falls_through(np.array(pc), kept, 0.5) if kept else None})
    return out


def restitution_peak(ev, levels):
    """Where the likelihood peaks along signed restitution error: a parabola through the median log Bayes factors, which
    are zero at no error. The restitution the model favours is 1 minus this."""
    x = np.array([m for m in levels if abs(m) <= 0.5])
    y = np.array([np.median(ev[f"collision/{m}/imp_nll"] - ev[f"collision/{m}/law_nll"]) for m in x])
    a, b = np.linalg.lstsq(np.stack([x, x * x], 1), y, rcond=None)[0]
    return float(-a / (2 * b))


def measure(fam, levels, evs, rng, ideal=None, causal=None):
    """Curves, thresholds and intervals for one group of seeds on one family."""
    if fam == "exit":  # the trace's null: no 2AFC question here, only surprise at empty frames after a visible exit
        tr = np.array([trace(ev, fam, levels) for ev in evs])
        return {"seeds": len(evs), "trace_nats": tr.mean(0), "trace_seeds": tr}
    if fam == "collision":
        signed, levels = levels, [m for m in levels if m > 0]
    per = np.array([curves(ev, fam, levels, cut(ev)) for ev in evs])  # (seeds, 4, levels)
    r = {"seeds": len(evs), "pc": per[:, 0].mean(0), "hit": per[:, 1].mean(0), "fa": per[:, 2].mean(0), "lbf": per[:, 3].mean(0), "pc_seeds": per[:, 0]}
    n = len(evs[0][f"{fam}/{levels[0]}/imp_nll"])
    kin, dur = fam in KINEMATIC, fam in DURATION
    if fam == "collision":
        r["peak_error_seeds"] = [restitution_peak(ev, signed) for ev in evs]
        r["peak_error"] = float(np.mean(r["peak_error_seeds"]))
        r["pc_signed"] = np.mean([curves(ev, fam, signed, cut(ev))[0] for ev in evs], 0)
    if kin:
        point = [thresholds(ev, fam, levels, cut(ev)) for ev in evs]
        r["t_afc_seeds"], r["t_det_seeds"] = [p[0] for p in point], [p[1] for p in point]
        r["t_afc"], r["t_det"] = geometric(r["t_afc_seeds"]), geometric(r["t_det_seeds"])
    if dur:
        tr = np.array([trace(ev, fam, levels) for ev in evs])
        r["trace_nats"], r["trace_seeds"] = tr.mean(0), tr
        r["crossing_seeds"] = [falls_through(p, levels, 0.5) for p in per[:, 0]]
        r["crossing"] = falls_through(r["pc"], levels, 0.5)
        r["trace_half_seeds"] = [falls_through(t, levels, t[0] / 2) for t in tr]
        r["trace_half"] = falls_through(r["trace_nats"], levels, r["trace_nats"][0] / 2)
        r["speed"] = by_speed(evs, fam, levels)
        full = np.array([curves(ev, fam, levels, cut(ev), window="_full")[0] for ev in evs])
        r["pc_full"], r["crossing_full"] = full.mean(0), falls_through(full.mean(0), levels, 0.5)

    boot = {k: [] for k in ("pc", "hit", "t_afc", "t_det", "trace", "crossing", "trace_half", "crossing_full")}
    for _ in range(BOOT):
        seeds, idx = rng.integers(0, len(evs), len(evs)), rng.integers(0, n, n)
        a = np.array([curves(evs[i], fam, levels, cut(evs[i]), idx) for i in seeds])
        pc = a[:, 0].mean(0)
        boot["pc"].append(pc)
        boot["hit"].append(a[:, 1].mean(0))
        if kin:
            t = [thresholds(evs[i], fam, levels, cut(evs[i]), idx, point[i][2]) for i in seeds]
            boot["t_afc"].append(geometric([x[0] for x in t]))
            boot["t_det"].append(geometric([x[1] for x in t]))
        if dur:
            tr = np.mean([trace(evs[i], fam, levels, idx) for i in seeds], 0)
            boot["trace"].append(tr)
            boot["crossing"].append(falls_through(pc, levels, 0.5))
            boot["crossing_full"].append(falls_through(np.mean([curves(evs[i], fam, levels, cut(evs[i]), idx, "_full")[0] for i in seeds], 0), levels, 0.5))
            boot["trace_half"].append(falls_through(tr, levels, tr[0] / 2))
    r["pc_lo"], r["pc_hi"] = np.percentile(boot["pc"], [2.5, 97.5], axis=0)
    r["hit_lo"], r["hit_hi"] = np.percentile(boot["hit"], [2.5, 97.5], axis=0)
    if kin:
        for k in ("t_afc", "t_det"):
            r[k + "_ci"] = np.percentile(boot[k], [2.5, 97.5], method="lower")  # some resamples never reach threshold
        for name, ref in (("efficiency", ideal), ("efficiency_causal", causal)):
            if ref and ref["t_afc"] > 0 and np.isfinite(r["t_afc"]):
                r[name] = (ref["t_afc"] / r["t_afc"]) ** 2  # d' grows in proportion to magnitude near threshold
                r[name + "_ci"] = [(ref["t_afc"] / hi) ** 2 for hi in r["t_afc_ci"][::-1]]
    if dur:
        r["trace_lo"], r["trace_hi"] = np.percentile(boot["trace"], [2.5, 97.5], axis=0)
        r["crossing_ci"] = np.percentile(boot["crossing"], [2.5, 97.5])
        r["crossing_full_ci"] = np.percentile(boot["crossing_full"], [2.5, 97.5])
        r["trace_half_ci"] = np.percentile(boot["trace_half"], [2.5, 97.5])
    return r


def agreement(d):
    return np.mean((d > 0) + 0.5 * (d == 0), -1)


def reader_gaps(margins, fam, levels, rng):
    """Crossing of each reader with a hierarchical bootstrap interval, and how far each outlasts the head on the same seeds
    and scenes."""
    a = {f"{k}{w}": np.array([[m[f"{fam}/{D}/{k}{w}"] for D in levels] for m in margins]) for k in READERS for w in ("", "_full")}  # seeds, levels, scenes
    point = lambda key, s, i: falls_through(agreement(a[key][s][:, :, i]).mean(0), levels, 0.5)
    every, scenes = np.arange(len(margins)), np.arange(a["head"].shape[-1])
    draws = [(rng.integers(0, len(margins), len(margins)), rng.integers(0, len(scenes), len(scenes))) for _ in range(BOOT)]
    out = {}
    for w in ("", "_full"):
        crossing = {k: [point(k + w, s, i) for s, i in draws] for k in READERS}
        for k in READERS:
            out[f"{k}{w}_crossing"], out[f"{k}{w}_crossing_ci"] = point(k + w, every, scenes), np.percentile(crossing[k], [2.5, 97.5])
        for k, base in [(k, "head") for k in READERS[1:]] + [("visible", "refit"), ("read", "visible")]:
            out[f"{k}_minus_{base}{w}"] = out[f"{k}{w}_crossing"] - out[f"{base}{w}_crossing"]
            out[f"{k}_minus_{base}{w}_ci"] = np.percentile(np.subtract(crossing[k], crossing[base]), [2.5, 97.5])
    return out


def xray(group, rng):
    """The decoded position curves of each seed by family and by how far through the hidden interval they were read, and the
    same state's decisions under each reader."""
    dirs = [p for p in seed_dirs(group) if (p / "xray.json").exists()]
    if not dirs:
        return None
    runs = [json.loads((p / "xray.json").read_text()) for p in dirs]
    margins = [np.load(p / "xray.npz") for p in dirs]
    for p, m in zip(dirs, margins):  # the head here and in sm.evaluate are two code paths over the same scenes and must agree
        ev = load(p)
        for key in m.files:
            fam, D, reader = key.split("/")
            if reader == "head":
                assert np.allclose(m[key], ev[f"{fam}/{D}/imp_nll"] - ev[f"{fam}/{D}/law_nll"], atol=0.05), (p, key)
    out = {"position": {}, "decisions": {}, "strokes": [r["strokes"] for r in runs], "ablated_variance": [r["ablated_variance"] for r in runs]}
    for fam in DURATION:
        out["position"][fam] = {}
        for frac in sorted({x["frac"] for x in runs[0]["position"]}):
            pick = [[x for x in r["position"] if x["family"] == fam and x["frac"] == frac] for r in runs]
            rec = {"D": [x["D"] for x in pick[0]]}
            for key in ("err_px", "hit_rate", "direct_px", "last_seen_px", "line_px", "mid_px", "mass", "mass_no_ball"):
                a = np.array([[x[key] for x in p] for p in pick])
                rec[key] = a.mean(0)
                rec[key + "_seeds"] = a
            out["position"][fam][str(frac)] = rec
        pick = [[x for x in r["decisions"] if x["family"] == fam] for r in runs]
        rec = {"D": [x["D"] for x in pick[0]]}
        for key in [k + w for k in READERS for w in ("", "_full")]:
            a = np.array([[x[key] for x in p] for p in pick])
            rec[key], rec[key + "_seeds"] = a.mean(0), a
            rec[key + "_crossing_seeds"] = [falls_through(row, rec["D"], 0.5) for row in a]
        rec.update(reader_gaps(margins, fam, rec["D"], rng))
        for key in ABLATED:
            kept = [x for x in pick[0] if key in x]
            rec[key] = {"D": [x["D"] for x in kept], "pc": np.mean([[x[key] for x in p if key in x] for p in pick], 0)}
        out["decisions"][fam] = rec
    return out


def summary():
    out = {"alpha": ALPHA, "train_reveal_max": TRAIN_REVEAL_MAX, "families": {}, "sizes": {}, "observers": {}}
    obs = {k: load(RUNS / "observers" / k) for k in OBSERVERS if (RUNS / "observers" / k / "eval.npz").exists()}
    groups = {g: [load(p) for p in seed_dirs(g)] for g in SIZES + CONTROLS}
    groups = {g: v for g, v in groups.items() if v}
    rng = np.random.default_rng(0)

    for fam, spec in E.FAMILIES.items():
        levels = spec["levels"]
        rec = {"levels": levels, "unit": spec["unit"], "observers": {}, "sizes": {}}
        for k, ev in obs.items():
            if fam == "exit":
                continue
            lv = [m for m in levels if m > 0] if fam == "collision" else levels
            pc, hit, fa, lbf = curves(ev, fam, lv, cut(ev))
            rec["observers"][k] = {"levels": lv, "pc": pc, "hit": hit, "fa": fa, "lbf": lbf}
            if fam in KINEMATIC:
                rec["observers"][k]["t_afc"], rec["observers"][k]["t_det"], _ = thresholds(ev, fam, lv, cut(ev))
            if fam in DURATION:
                rec["observers"][k]["crossing"] = falls_through(pc, lv, 0.5)
                rec["observers"][k]["pc_full"] = curves(ev, fam, lv, cut(ev), window="_full")[0]
        for g, evs in groups.items():
            rec["sizes"][g] = measure(fam, levels, evs, rng, rec["observers"].get("ideal"), rec["observers"].get("causal"))
        out["families"][fam] = rec

    for g in groups:
        names = [p.name for p in seed_dirs(g)]
        out["sizes"][g] = {"seeds": names, "calibration": [json.loads((RUNS / n / "eval.json").read_text()) for n in names], "xray": xray(g, rng)}
    for name in EXPLORATORY:
        if all(name not in rec["seeds"] for rec in out["sizes"].values()) and (RUNS / name / "eval.npz").exists():
            ev = load(RUNS / name)
            out["set_aside"] = {name: {fam + w: falls_through(curves(ev, fam, E.FAMILIES[fam]["levels"], cut(ev), window=w)[0], E.FAMILIES[fam]["levels"], 0.5) for fam in DURATION for w in ("", "_full")}}
    for k in obs:
        out["observers"][k] = json.loads((RUNS / "observers" / k / "eval.json").read_text())

    def plain(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        raise TypeError(type(x))

    Path("results").mkdir(exist_ok=True)
    Path("results/summary.json").write_text(json.dumps(out, indent=1, default=plain))
    return out


if __name__ == "__main__":
    summary()
    print("wrote results/summary.json")
