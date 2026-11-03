"""The criteria in docs/analysis-plan.md, evaluated. Writes results/plan.md, so a verdict is computed and never chosen.

    python -m sm.plan
"""

import json
from pathlib import Path

RUNS = Path("runs")
FAMS = ("vanish", "rebound")


def per_run(seed, fam, D, name="xray"):
    rec = [r for r in json.loads((RUNS / seed / f"{name}.json").read_text())["decisions"] if r["family"] == fam and r["D"] == D]
    return rec[0]


def above_zero(ci):
    return ci[0] > 0


def counts(gap, ci):
    return (ci[0] > 0 or ci[1] < 0) and abs(gap) >= 2


def show(x):
    return "below the first level" if x == 0 else f"{x:.1f}"


def verdicts(s):
    fam = lambda f, g: s["families"][f]["sizes"].get(g)
    xr = lambda g: s["sizes"][g]["xray"]
    seeds = s["sizes"]["5m"]["seeds"]
    out = []

    rows, met = [], True
    for f in FAMS:
        d = xr("5m")["decisions"][f]
        ratio = d["read_crossing"] / max(d["head_crossing"], 1e-9)
        ok = above_zero(d["read_minus_head_ci"]) and above_zero(d["read_minus_head_full_ci"]) and ratio >= 1.5
        met &= ok
        rows.append(f"{f}: head {show(d['head_crossing'])}, read {show(d['read_crossing'])}, {ratio:.2f} times at 3 px; read minus head {d['read_minus_head']:+.1f} {_ci(d['read_minus_head_ci'])} at 3 px, {d['read_minus_head_full']:+.1f} {_ci(d['read_minus_head_full_ci'])} at 9 px")
        rows.append(f"{f} by seed, 3 px: head {_list(d['head_crossing_seeds'])}, read {_list(d['read_crossing_seeds'])}")
        if "read_minus_head_first" in d:
            rows.append(f"{f}, first frame where the twins differ, both predicted from one state (added after the confirmatory results, so exploratory): head {show(d['head_first_crossing'])} {_ci(d['head_first_crossing_ci'])}, read {show(d['read_first_crossing'])} {_ci(d['read_first_crossing_ci'])}, read minus head {d['read_minus_head_first']:+.1f} {_ci(d['read_minus_head_first_ci'])}")
            rows.append(f"{f}, same frame at D = {d['D'][0]}: head {d['head_first'][0]:.2f}, read {d['read_first'][0]:.2f}, the observer told the lawful frame {d['ideal_first'][0]:.2f}; levels where the read beats that observer: {sum(r > i for r, i in zip(d['read_first'], d['ideal_first']))} of {len(d['D'])}; brightness the read paints on the sliver against the truth, by level: {min(d['read_gain']):.2f} to {max(d['read_gain']):.2f}, visible {min(d['visible_gain']):.2f} to {max(d['visible_gain']):.2f}")
    out.append(("Q1. Does one state hold more than its head reads out?", met, rows))

    rows = []
    for f in FAMS:
        d = xr("5m")["decisions"][f]
        for name, key in (("the learned variance (mean against head)", "mean_minus_head"), ("how the map was trained (refit against head)", "refit_minus_head"), ("covered pixels and drawing through slabs (visible against refit)", "visible_minus_refit"), ("being told where the hidden ball is (read against visible)", "read_minus_visible")):
            rows.append(f"{f}, {name}: {d[key]:+.1f} {_ci(d[key + '_ci'])}, {'counts' if counts(d[key], d[key + '_ci']) else 'not the cause'}")
    out.append(("Q2. Where does the gap come from?", None, rows))

    rows, met = [], True
    for f in FAMS:
        for seed in seeds:
            r = per_run(seed, f, 8)
            drop, control = r["head"] - r["head_without_read_4"], r["head"] - r["head_without_random_4"]
            met &= drop - control >= 0.10
            rows.append(f"{f}, {seed}, D = 8: head {r['head']:.3f}, read's four out {r['head_without_read_4']:.3f}, random four out {r['head_without_random_4']:.3f}, extra drop {drop - control:.3f}")
    share = [next(v["variance_share"] for v in run if v["directions"] == "read" and v["k"] == 4) for run in xr("5m")["ablated_variance"]]
    rows.append(f"the read's four carry {_list(share, 4)} of the state's variance")
    if "control_variance_share" in xr("5m"):
        for f in FAMS:
            for seed in seeds:
                r, c = per_run(seed, f, 8), per_run(seed, f, 8, "first")
                rows.append(f"{f}, {seed}, D = 8, control added after the confirmatory results: four directions chosen the same way from the map to slab cover, out {c['head_without_cover_4']:.3f}, against the read's four {r['head_without_read_4']:.3f}")
        rows.append(f"the cover map's four carry {_list(xr('5m')['control_variance_share'], 4)} of the state's variance")
    out.append(("Q3. Does the head use those directions?", met, rows))

    rows, met = [], True
    for seed, strokes in zip(seeds, xr("5m")["strokes"]):
        for st in strokes:
            back, away = st["return"], st["outward"]
            met &= back["sign_agreement"] >= 0.90 and 0.7 <= back["slope"] <= 1.3 and away["sign_agreement"] >= 0.80
            rows.append(f"{seed}, D = {st['D']}: return direction {back['sign_agreement']:.2f} slope {back['slope']:.2f}, outward direction {away['sign_agreement']:.2f}")
    out.append(("Q4. Does the read follow an unseen bounce?", met, rows))

    levels = s["families"]["vanish"]["levels"]
    win = fam("vanish", "1m-window8")
    rows, met = [], None
    if win:
        tr = win["trace_nats"]
        late = [t for D, t in zip(levels, tr) if D > 12]
        met = max(late) < 0.1 * tr[0]
        rows.append(f"window control, vanish trace at D = 1: {tr[0]:.3f} nats; largest past D = 12: {max(late):.3f}")
    ex = fam("exit", "5m")
    van = fam("vanish", "5m")
    if ex and van:
        rows.append(f"5M vanish trace by D {_pairs(levels, van['trace_nats'])}")
        rows.append(f"5M exit trace by D {_pairs(s['families']['exit']['levels'], ex['trace_nats'])}")
    out.append(("Q5. Is the trace a trace of the ball?", met, rows))

    rows, parts = [], []
    for f in FAMS:
        small, big, short, long = (fam(f, g) for g in ("1m", "5m", "5m-short", "1m-long"))
        rows.append(f"{f}, head at 3 px: 1M {_cross(small)}, 5M {_cross(big)}, 5M short {_cross(short)}, 1M long {_cross(long)}, 20M {_cross(fam(f, '20m'))}")
        rows.append(f"{f}, trace halves: 1M {show(small['trace_half'])} {_ci(small['trace_half_ci'])}, 5M {show(big['trace_half'])} {_ci(big['trace_half_ci'])}")
        parts.append(big["crossing_ci"][0] > small["crossing_ci"][1] and (short is not None and short["crossing"] > small["crossing_ci"][1]))
        if long is not None:
            rows.append(f"{f}: 1M long reaches 5M's interval: {long['crossing'] >= big['crossing_ci'][0]}")
    out.append(("Q6. What does size do at equal training? (met means bigger decides better, on both families)", all(parts) if fam("vanish", "5m-short") else None, rows))

    band = lambda b: f"past {b['levels'][-1]}, the longest hidden time with scenes in this band" if b["pc"][-1] >= 0.5 else show(b["crossing"])
    rows = [f"{b['band'][0]} to {b['band'][1]} px per frame: {band(b)}" for b in van["speed"]]
    out.append(("Q7. Is there a speed-independent fall?", None, rows))

    rows = []
    for f in ("teleport", "phantom", "deflect"):
        r = fam(f, "5m")
        rows.append(f"{f}: against the causal ideal {r['efficiency_causal']:.3f} {_ci3(r['efficiency_causal_ci'])}, against the state told {r['efficiency']:.3f}")
    out.append(("Q8. Efficiency", None, rows))

    peak = [1 - e for e in fam("collision", "5m")["peak_error_seeds"]]
    out.append(("Q9. Collision bias: quoted if under 1 in every seed", all(p < 1 for p in peak), [f"peak restitution by seed: {_list(peak, 2)}"]))

    rows = []
    for b in s["sizes"]["5m"]["calibration"][0]["long_rollout"]:
        got = [c["long_rollout"][b]["0.001"] for c in s["sizes"]["5m"]["calibration"]]
        rows.append(f"frames {b}: measured at a promised 1 in 1000: {_list(got, 5)}, {'within' if all(0.0005 <= g <= 0.002 for g in got) else 'outside'} a factor of 2")
    out.append(("Q10. Calibration, demo only", None, rows))
    return out


def _ci(c):
    return f"[{c[0]:.1f}, {c[1]:.1f}]"


def _ci3(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def _list(a, digits=1):
    return ", ".join(f"{x:.{digits}f}" for x in a)


def _pairs(levels, a):
    return " ".join(f"{D}: {x:.3f}" for D, x in zip(levels, a))


def _cross(r):
    return "not measured yet" if r is None else f"{show(r['crossing'])} [{show(r['crossing_ci'][0])}, {show(r['crossing_ci'][1])}]"


def main():
    s = json.loads(Path("results/summary.json").read_text())
    lines = ["# The analysis plan, evaluated", "", f"Written by `python -m sm.plan` from `results/summary.json`, against `docs/analysis-plan.md`. Confirmatory 5M seeds: {', '.join(s['sizes']['5m']['seeds'])}.", ""]
    for title, met, rows in verdicts(s):
        lines += [f"## {title}", "", f"**{'Reported, no pass or fail' if met is None else 'Met' if met else 'Not met'}.**", ""] + [f"- {r}" for r in rows] + [""]
    Path("results/plan.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
