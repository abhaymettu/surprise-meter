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
CONTROLS = ["1m-window8"]  # attention over 8 past frames: a model whose memory has a known length
OBSERVERS = ["ideal", "cv", "cv0"]
KINEMATIC = ["teleport", "deflect", "phantom", "collision", "hidden"]
DURATION = ["vanish", "rebound"]
ALPHA = 1e-3
BOOT = 500
TRAIN_REVEAL_MAX = 41  # the longest hidden run seen in a training clip; longer durations are extrapolation


def load(path):
    d = np.load(path / "eval.npz")
    return {k: d[k] for k in d.files}


def seed_dirs(group):
    """Run directories of one size or control, in seed order. Experiments with other names are not picked up."""
    found = [p for p in RUNS.iterdir() if re.fullmatch(rf"{re.escape(group)}-s(\d+)", p.name)]
    return sorted(found, key=lambda p: int(p.name.rsplit("-s", 1)[1]))


def cut(ev, alpha=ALPHA):
    return float(np.interp(np.log(alpha * ev["null_n"]), np.log(ev["null_ranks"]), ev["null_tail"]))


def curves(ev, fam, levels, c, idx=None):
    """Per level: proportion correct in 2AFC, hit and false-alarm rates at the criterion, median log Bayes factor."""
    pc, hit, fa, lbf = [], [], [], []
    for lv in levels:
        k = f"{fam}/{lv}"
        i = idx if idx is not None else np.arange(len(ev[f"{k}/imp_nll"]))
        d = ev[f"{k}/imp_nll"][i] - ev[f"{k}/law_nll"][i]
        pc.append(np.mean((d > 0) + 0.5 * (d == 0)))
        hit.append(np.mean(ev[f"{k}/imp_ev"][i] >= c))
        fa.append(np.mean(ev[f"{k}/law_ev"][i] >= c))
        lbf.append(np.median(d))
    return np.array(pc), np.array(hit), np.array(fa), np.array(lbf)


def trace(ev, fam, levels, idx=None):
    """Median surprisal at identical empty frames, ball seen earlier minus ball never there. Zero means no memory."""
    out = []
    for lv in levels:
        d = ev[f"{fam}/{lv}/imp_nll"] - ev[f"{fam}/{lv}/ctrl_nll"]
        out.append(np.median(d if idx is None else d[idx]))
    return np.array(out)


def fit_threshold(x, y, n, floor, target, start=None):
    """Binomial ML logistic in log magnitude with a fixed floor and a small lapse. Magnitude reaching `target`, or an
    infinity or zero when the data never bracket it. `start` skips the search over starting points."""
    x, y = np.log(np.asarray(x, float)), np.asarray(y, float)
    if y.max() < target:
        return float("inf"), None
    if y.min() >= target:
        return 0.0, None
    lapse = 0.01

    def nll(th):
        mu, ls = th
        p = floor + (1 - floor - lapse) / (1 + np.exp(-(x - mu) / np.exp(ls)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(n * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum()

    starts = [start] if start is not None else [[m, s] for m in np.quantile(x, [0.2, 0.5, 0.8]) for s in (-1.0, 0.0)]
    best = None
    for th in starts:
        r = minimize(nll, th, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3})
        if best is None or r.fun < best.fun:
            best = r
    mu, ls = best.x
    q = np.clip((target - floor) / (1 - floor - lapse), 1e-4, 1 - 1e-4)
    return float(np.exp(mu + np.exp(ls) * np.log(q / (1 - q)))), best.x


def thresholds(ev, fam, levels, c, idx=None, starts=(None, None)):
    n = len(ev[f"{fam}/{levels[0]}/imp_nll"]) if idx is None else len(idx)
    pc, hit, fa, _ = curves(ev, fam, levels, c, idx)
    t_afc, s_afc = fit_threshold(levels, pc, n, 0.5, 0.75, starts[0])
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
    """2AFC split at each level's median ball speed. Long hidden times mean slow balls, so D and speed are confounded."""
    slow, fast = [], []
    for lv in levels:
        k = f"{fam}/{lv}"
        s, f = [], []
        for ev in evs:
            d, v = ev[f"{k}/imp_nll"] - ev[f"{k}/law_nll"], ev[f"{k}/speed"]
            m = np.median(v)
            s.append(np.mean(d[v <= m] > 0))
            f.append(np.mean(d[v > m] > 0) if (v > m).any() else np.nan)
        slow.append(np.mean(s))
        fast.append(np.nanmean(f))
    speed = [float(np.median(evs[0][f"{fam}/{lv}/speed"])) for lv in levels]
    return {"pc_slow": slow, "pc_fast": fast, "median_speed": speed}


def measure(fam, levels, evs, rng, ideal=None):
    """Curves, thresholds and intervals for one group of seeds on one family."""
    per = np.array([curves(ev, fam, levels, cut(ev)) for ev in evs])  # (seeds, 4, levels)
    r = {"seeds": len(evs), "pc": per[:, 0].mean(0), "hit": per[:, 1].mean(0), "fa": per[:, 2].mean(0), "lbf": per[:, 3].mean(0), "pc_seeds": per[:, 0]}
    n = len(evs[0][f"{fam}/{levels[0]}/imp_nll"])
    kin, dur = fam in KINEMATIC, fam in DURATION
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

    boot = {k: [] for k in ("pc", "hit", "t_afc", "t_det", "trace", "crossing", "trace_half")}
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
            boot["trace_half"].append(falls_through(tr, levels, tr[0] / 2))
    r["pc_lo"], r["pc_hi"] = np.percentile(boot["pc"], [2.5, 97.5], axis=0)
    r["hit_lo"], r["hit_hi"] = np.percentile(boot["hit"], [2.5, 97.5], axis=0)
    if kin:
        for k in ("t_afc", "t_det"):
            r[k + "_ci"] = np.percentile(boot[k], [2.5, 97.5], method="lower")  # some resamples never reach threshold
        if ideal and ideal["t_afc"] > 0 and np.isfinite(r["t_afc"]):
            r["efficiency"] = (ideal["t_afc"] / r["t_afc"]) ** 2  # d' grows in proportion to magnitude near threshold
            r["efficiency_ci"] = [(ideal["t_afc"] / hi) ** 2 for hi in r["t_afc_ci"][::-1]]
    if dur:
        r["trace_lo"], r["trace_hi"] = np.percentile(boot["trace"], [2.5, 97.5], axis=0)
        r["crossing_ci"] = np.percentile(boot["crossing"], [2.5, 97.5])
        r["trace_half_ci"] = np.percentile(boot["trace_half"], [2.5, 97.5])
    return r


def xray(group):
    """The decoded position curves of each seed, by family and by how far through the hidden interval they were read."""
    rows = [json.loads((p / "xray.json").read_text()) for p in seed_dirs(group) if (p / "xray.json").exists()]
    if not rows:
        return None
    out = {}
    for fam in DURATION:
        out[fam] = {}
        for frac in sorted({x["frac"] for x in rows[0]}):
            pick = [[x for x in r if x["family"] == fam and x["frac"] == frac] for r in rows]
            rec = {"D": [x["D"] for x in pick[0]]}
            for key in ("err_px", "last_seen_px", "line_px", "mass", "mass_no_ball"):
                a = np.array([[x[key] for x in p] for p in pick])
                rec[key] = a.mean(0)
                rec[key + "_seeds"] = a
            out[fam][str(frac)] = rec
    return out


def summary():
    out = {"alpha": ALPHA, "train_reveal_max": TRAIN_REVEAL_MAX, "families": {}, "sizes": {}, "observers": {}}
    obs = {k: load(RUNS / "observers" / k) for k in OBSERVERS if (RUNS / "observers" / k / "eval.npz").exists()}
    groups = {g: [load(p) for p in seed_dirs(g) if (p / "eval.npz").exists()] for g in SIZES + CONTROLS}
    groups = {g: v for g, v in groups.items() if v}
    rng = np.random.default_rng(0)

    for fam, spec in E.FAMILIES.items():
        levels = spec["levels"]
        rec = {"levels": levels, "unit": spec["unit"], "observers": {}, "sizes": {}}
        for k, ev in obs.items():
            pc, hit, fa, lbf = curves(ev, fam, levels, cut(ev))
            rec["observers"][k] = {"pc": pc, "hit": hit, "fa": fa, "lbf": lbf}
            if fam in KINEMATIC:
                rec["observers"][k]["t_afc"], rec["observers"][k]["t_det"], _ = thresholds(ev, fam, levels, cut(ev))
            if fam in DURATION:
                rec["observers"][k]["crossing"] = falls_through(pc, levels, 0.5)
        for g, evs in groups.items():
            rec["sizes"][g] = measure(fam, levels, evs, rng, rec["observers"].get("ideal"))
        out["families"][fam] = rec

    for g in groups:
        names = [p.name for p in seed_dirs(g) if (p / "eval.npz").exists()]
        out["sizes"][g] = {"seeds": names, "calibration": [json.loads((RUNS / n / "eval.json").read_text()) for n in names], "xray": xray(g)}
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
