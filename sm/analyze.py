"""Turn eval.npz files into thresholds, efficiencies and intervals. Writes results/summary.json.

    python -m sm.analyze

Each size has several seeds and every seed saw the same scenes. Intervals are a hierarchical bootstrap: resample
seeds, then resample scenes, so they carry both model uncertainty and stimulus uncertainty.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from . import events as E

RUNS = Path("runs")
SIZES = ["1m", "5m", "20m"]
OBSERVERS = ["ideal", "cv", "cv0"]
KINEMATIC = ["teleport", "deflect", "phantom", "collision", "hidden"]
DURATION = ["vanish", "rebound"]
ALPHA = 1e-3
BOOT = 500
TRAIN_REVEAL_MAX = 28  # a 32-frame training clip cannot show a reveal after more hidden frames than this


def load(path):
    d = np.load(path / "eval.npz")
    return {k: d[k] for k in d.files}


def seed_dirs(size):
    return sorted(RUNS.glob(f"{size}-s*"), key=lambda p: int(p.name.split("-s")[1]))


def cut(ev, alpha=ALPHA):
    s = ev["null_sorted"]
    return s[min(int(np.ceil(len(s) * (1 - alpha))), len(s) - 1)]


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


def fit_threshold(x, y, n, floor, target):
    """Binomial ML logistic in log magnitude with a fixed floor and a small lapse. Magnitude reaching `target`, or None
    when the data never bracket it (reported as above or below the tested range)."""
    x, y = np.log(np.asarray(x, float)), np.asarray(y, float)
    if y.max() < target:
        return float("inf")
    if y.min() >= target:
        return 0.0
    lapse = 0.01

    def nll(th):
        mu, ls = th
        p = floor + (1 - floor - lapse) / (1 + np.exp(-(x - mu) / np.exp(ls)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(n * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum()

    best = None
    for mu0 in np.quantile(x, [0.2, 0.5, 0.8]):
        for ls0 in (-1.0, 0.0):
            r = minimize(nll, [mu0, ls0], method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3})
            if best is None or r.fun < best.fun:
                best = r
    mu, ls = best.x
    q = (target - floor) / (1 - floor - lapse)
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return float(np.exp(mu + np.exp(ls) * np.log(q / (1 - q))))


def thresholds(ev, fam, levels, c, idx=None):
    n = len(ev[f"{fam}/{levels[0]}/imp_nll"]) if idx is None else len(idx)
    pc, hit, fa, _ = curves(ev, fam, levels, c, idx)
    t_afc = fit_threshold(levels, pc, n, 0.5, 0.75)
    floor = float(np.clip(fa.mean(), 0.0, 0.5))
    t_det = fit_threshold(levels, hit, n, floor, floor + 0.5 * (1 - floor))
    return t_afc, t_det


def carry_over(pc, levels):
    """The longest hidden duration from which the observer still favours the lawful world at every shorter duration."""
    last = 0
    for lv, p in zip(levels, pc):
        if p >= 0.5:
            last = lv
        else:
            break
    return last


def dprime(pc):
    return np.sqrt(2) * norm.ppf(np.clip(pc, 1e-3, 1 - 1e-3))


def summary():
    out = {"alpha": ALPHA, "families": {}, "sizes": {}, "observers": {}}
    obs = {k: load(RUNS / "observers" / k) for k in OBSERVERS if (RUNS / "observers" / k / "eval.npz").exists()}
    size_evals = {s: [load(p) for p in seed_dirs(s) if (p / "eval.npz").exists()] for s in SIZES}
    size_evals = {s: v for s, v in size_evals.items() if v}
    rng = np.random.default_rng(0)

    for fam, spec in E.FAMILIES.items():
        levels = spec["levels"]
        rec = {"levels": levels, "unit": spec["unit"], "observers": {}, "sizes": {}}
        for k, ev in obs.items():
            pc, hit, fa, lbf = curves(ev, fam, levels, cut(ev))
            rec["observers"][k] = {"pc": pc.tolist(), "hit": hit.tolist(), "fa": fa.tolist(), "lbf": lbf.tolist()}
            if fam in KINEMATIC:
                rec["observers"][k]["t_afc"], rec["observers"][k]["t_det"] = thresholds(ev, fam, levels, cut(ev))
            if fam in DURATION:
                rec["observers"][k]["carry_over"] = carry_over(pc, levels)
        for s, evs in size_evals.items():
            per = [curves(ev, fam, levels, cut(ev)) for ev in evs]
            arr = np.array(per)  # (seeds, 4, levels)
            r = {"seeds": len(evs), "pc": arr[:, 0].mean(0).tolist(), "hit": arr[:, 1].mean(0).tolist(), "fa": arr[:, 2].mean(0).tolist(), "lbf": arr[:, 3].mean(0).tolist(), "pc_seeds": arr[:, 0].tolist()}
            n = len(evs[0][f"{fam}/{levels[0]}/imp_nll"])
            boot_pc, boot_hit, boot_t, boot_c = [], [], [], []
            for _ in range(BOOT):
                seeds = rng.integers(0, len(evs), len(evs))
                idx = rng.integers(0, n, n)
                res = [curves(evs[i], fam, levels, cut(evs[i]), idx) for i in seeds]
                a = np.array(res)
                boot_pc.append(a[:, 0].mean(0))
                boot_hit.append(a[:, 1].mean(0))
                if fam in KINEMATIC:
                    boot_t.append(np.array([thresholds(evs[i], fam, levels, cut(evs[i]), idx) for i in seeds]))
                if fam in DURATION:
                    boot_c.append(np.mean([carry_over(a[j, 0], levels) for j in range(len(seeds))]))
            r["pc_lo"], r["pc_hi"] = np.percentile(boot_pc, [2.5, 97.5], axis=0).tolist()
            r["hit_lo"], r["hit_hi"] = np.percentile(boot_hit, [2.5, 97.5], axis=0).tolist()
            if fam in KINEMATIC:
                seed_t = np.array([thresholds(ev, fam, levels, cut(ev)) for ev in evs])
                gm = lambda a: float(np.exp(np.nanmean(np.log(np.where(a > 0, a, np.nan)), axis=-1))) if np.isfinite(a).all() else float("inf")
                r["t_afc"], r["t_det"] = gm(seed_t[:, 0]), gm(seed_t[:, 1])
                b = np.array(boot_t)  # (boot, seeds, 2)
                for j, name in enumerate(("t_afc", "t_det")):
                    vals = np.array([gm(row[:, j]) for row in b])
                    r[name + "_ci"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
                if "ideal" in rec["observers"]:
                    ti = rec["observers"]["ideal"]["t_afc"]
                    r["efficiency"] = float((ti / r["t_afc"]) ** 2) if ti > 0 and np.isfinite(r["t_afc"]) else 0.0
            if fam in DURATION:
                r["carry_over_seeds"] = [carry_over(arr[i, 0], levels) for i in range(len(evs))]
                r["carry_over_ci"] = np.percentile(boot_c, [2.5, 97.5]).tolist()
                ctrl = []
                for ev in evs:
                    ctrl.append([float(np.median(ev[f"{fam}/{lv}/imp_nll"][: len(ev[f"{fam}/{lv}/ctrl_nll"])] - ev[f"{fam}/{lv}/ctrl_nll"])) for lv in levels])
                r["trace_nats"] = np.mean(ctrl, 0).tolist()
                r["trace_seeds"] = ctrl
            rec["sizes"][s] = r
        out["families"][fam] = rec

    for s, evs in size_evals.items():
        names = [p.name for p in seed_dirs(s) if (p / "eval.npz").exists()]
        rep = [json.loads((RUNS / n / "eval.json").read_text()) for n in names]
        out["sizes"][s] = {"seeds": names, "calibration": rep}
    for k in obs:
        out["observers"][k] = json.loads((RUNS / "observers" / k / "eval.json").read_text())

    Path("results").mkdir(exist_ok=True)
    Path("results/summary.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    summary()
    print("wrote results/summary.json")
