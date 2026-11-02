"""Draw every figure from results/summary.json. Writes results/figures/*.png.

    python -m sm.figures

Sizes differ in lightness and marker as well as hue, and every line is labelled where it ends, so nothing depends on
telling two colours apart.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import events as E
from . import world as W
from .analyze import falls_through

OUT = Path("results/figures")
SIZE_STYLE = {
    "1m": dict(color="#9db7d5", marker="o", label="1M"),
    "5m": dict(color="#4a7fb5", marker="s", label="5M"),
    "20m": dict(color="#12355b", marker="D", label="20M"),
    "1m-window8": dict(color="#b5651d", marker="^", label="1M, 8-frame window"),
    "1m-long": dict(color="#9db7d5", marker="v", label="1M, 60000 steps"),
    "5m-short": dict(color="#4a7fb5", marker="v", label="5M, 10000 steps"),
}
OBS_STYLE = {
    "ideal": dict(color="#111111", ls="-", lw=1.1, label="ideal"),
    "causal": dict(color="#111111", ls="-.", lw=1.1, label="causal ideal"),
    "walls": dict(color="#888888", ls="-", lw=1.1, label="dead reckoning with walls"),
    "cv": dict(color="#666666", ls="--", lw=1.1, label="wall-blind dead reckoning"),
    "frozen": dict(color="#aaaaaa", ls=":", lw=1.4, label="frozen at last seen"),
    "cv0": dict(color="#666666", ls=":", lw=1.4, label="no memory"),
}
KIN = ["teleport", "deflect", "phantom", "collision", "hidden"]
READ_AT = {"vanish": "1.0", "rebound": "0.75"}  # the rebound's way back, when the last seen position and the straight line are both wrong
TITLES = {
    "teleport": "Teleport: a ball jumps along its path",
    "deflect": "Deflect: a ball turns with nothing to hit",
    "phantom": "Phantom: a bounce short of the wall",
    "collision": "Collision: wrong restitution",
    "hidden": "Hidden jump: moved while out of sight",
    "vanish": "Vanish",
    "rebound": "Rebound",
}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def groups(rec, include=("1m", "5m", "20m")):
    return [g for g in include if g in rec["sizes"]]


def line(ax, x, y, lo=None, hi=None, **st):
    label = st.pop("label", None)
    ax.plot(x, y, lw=1.8, ms=4, label=label, **st)
    if lo is not None:
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.16, lw=0)


def end_labels(ax, entries, gap=0.05):
    """Write each line's name at its right end, spreading labels that would collide."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    order = sorted(entries, key=lambda e: e[1])
    last = -np.inf
    for label, y, color in order:
        y = max(y, last + gap * span)
        last = y
        ax.annotate(label, xy=(1, y), xycoords=("axes fraction", "data"), xytext=(5, 0), textcoords="offset points", va="center", color=color, fontsize=8.5)


def permanence(s, fam, path, title):
    """The same memory read three ways: the decision, the graded trace, and a linear decode of the state."""
    rec = s["families"][fam]
    D = np.array(rec["levels"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), gridspec_kw={"wspace": 0.62})
    ax = axes[0]
    ends = []
    for g in groups(rec):
        r = rec["sizes"][g]
        line(ax, D, r["pc"], r["pc_lo"], r["pc_hi"], **SIZE_STYLE[g])
        ends.append((SIZE_STYLE[g]["label"], r["pc"][-1], SIZE_STYLE[g]["color"]))
    for k, o in rec["observers"].items():
        if k in ("ideal", "walls", "cv", "cv0"):
            ax.plot(D, o["pc"], **{**OBS_STYLE[k], "label": None})
            ends.append((OBS_STYLE[k]["label"], o["pc"][-1], OBS_STYLE[k]["color"]))
    ax.axhline(0.5, color="#bbbbbb", lw=0.8, zorder=0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("frames hidden")
    ax.set_ylabel("pairs favouring the lawful twin")
    ax.set_title("Decision")
    end_labels(ax, ends)

    ax = axes[1]
    ends = []
    for g in groups(rec, ("1m", "5m", "20m", "1m-window8")):
        r = rec["sizes"][g]
        if "trace_nats" not in r:
            continue
        line(ax, D, r["trace_nats"], r["trace_lo"], r["trace_hi"], **SIZE_STYLE[g])
        ends.append((SIZE_STYLE[g]["label"], r["trace_nats"][-1], SIZE_STYLE[g]["color"]))
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.set_xlabel("frames hidden")
    ax.set_ylabel("seen minus never there (nats per frame)")
    ax.set_title("Trace at identical empty frames")
    if "exit" in s["families"]:
        for g in groups(s["families"]["exit"]):
            ex = s["families"]["exit"]
            ax.plot(ex["levels"], ex["sizes"][g]["trace_nats"], color=SIZE_STYLE[g]["color"], ls=":", lw=1.2)
        ax.plot([], [], color="#666666", ls=":", lw=1.2, label="after a visible exit")
    ax.legend(loc="upper right", fontsize=8.5)

    ax = axes[2]
    base = None
    ends = []
    for g in groups(rec):
        x = s["sizes"][g]["xray"]["position"][fam][READ_AT[fam]]
        seeds = np.array(x["direct_px_seeds"])
        line(ax, x["D"], x["direct_px"], seeds.min(0), seeds.max(0), **SIZE_STYLE[g])
        ends.append((SIZE_STYLE[g]["label"], x["direct_px"][-1], SIZE_STYLE[g]["color"]))
        base = x
    if base:
        ax.plot(base["D"], base["last_seen_px"], color="#666666", ls=":", lw=1.4)
        ax.plot(base["D"], base["line_px"], color="#666666", ls="--", lw=1.1)
        ends.append(("last seen", base["last_seen_px"][-1], "#666666"))
        ends.append(("straight line", base["line_px"][-1], "#666666"))
        if fam == "rebound":
            ax.plot(base["D"], base["mid_px"], color="#aaaaaa", ls="-.", lw=1.1)
            ends.append(("slab middle", base["mid_px"][-1], "#aaaaaa"))
    ax.set_xlabel("frames hidden")
    ax.set_ylabel("position error (px of 32)")
    ax.set_ylim(bottom=-0.15)
    ax.set_title("Linear read of the state" + ("" if READ_AT[fam] == "1.0" else ", 3/4 of the way"))
    end_labels(ax, ends)
    fig.suptitle(title, x=0.125, y=1.02, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


def psychometric(s, path):
    fams = ["teleport", "deflect", "phantom", "collision", "hidden"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), gridspec_kw={"wspace": 0.3, "hspace": 0.42})
    for ax, fam in zip(axes.ravel(), fams):
        rec = s["families"][fam]
        x = np.array([m for m in rec["levels"] if m > 0], float)
        for g in groups(rec):
            r = rec["sizes"][g]
            line(ax, x, r["pc"], r["pc_lo"], r["pc_hi"], **SIZE_STYLE[g])
        for k, o in rec["observers"].items():
            if k in ("ideal", "causal", "cv", "cv0"):
                ax.plot(x, o["pc"], **OBS_STYLE[k])
        ax.axhline(0.5, color="#bbbbbb", lw=0.8, zorder=0)
        ax.axhline(0.75, color="#dddddd", lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(rec["unit"])
        ax.set_title(TITLES[fam])
    axes[0, 0].set_ylabel("2AFC proportion correct")
    axes[1, 0].set_ylabel("2AFC proportion correct")
    ax = axes[1, 2]
    ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ax.legend(handles, labels, loc="center left", fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def calibration(s, path):
    """False-alarm rate on held-out lawful frames against the rate the table promises, split by what was happening."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True, gridspec_kw={"wspace": 0.12})
    contexts = [("quiet", "o", "#12355b"), ("edge", "s", "#4a7fb5"), ("contact", "^", "#b5651d"), ("reveal", "D", "#555555")]
    present = [g for g in ("1m", "5m", "20m") if g in s["sizes"]]
    for ax, g in zip(axes, present + [None] * (3 - len(present))):
        if g is None:
            ax.axis("off")
            continue
        rep = s["sizes"][g]["calibration"]
        nominal = [0.01, 0.001, 0.0001]
        ax.plot(nominal, nominal, color="#bbbbbb", lw=1, zorder=0)
        for name, marker, color in contexts:
            y = [max(np.mean([r[f"fa_{name}"][str(a)] for r in rep]), 1e-6) for a in nominal]
            ax.plot(nominal, y, marker=marker, color=color, lw=1.2, ms=5, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("promised rate")
        ax.set_title(SIZE_STYLE[g]["label"])
    axes[0].set_ylabel("measured rate")
    axes[0].legend(fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def speed_split(s, path):
    """Long hidden times come with slow balls, so a pooled crossing mixes speeds. The decision curve within fixed speed bands."""
    rec = s["families"]["vanish"]
    gs = groups(rec)
    fig, axes = plt.subplots(1, len(gs), figsize=(3.6 * len(gs), 3.2), sharey=True, squeeze=False)
    marks = [("v", "#12355b"), ("o", "#4a7fb5"), ("^", "#9db7d5")]
    for ax, g in zip(axes[0], gs):
        for band, (m, c) in zip(rec["sizes"][g]["speed"], marks):
            ax.plot(band["levels"], band["pc"], color=c, marker=m, ms=4, lw=1.6, label=f"{band['band'][0]} to {band['band'][1]} px per frame")
        ax.axhline(0.5, color="#bbbbbb", lw=0.8, zorder=0)
        ax.set_title(SIZE_STYLE[g]["label"])
        ax.set_xlabel("frames hidden")
    axes[0, 0].set_ylabel("2AFC proportion correct")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def readers(s, path):
    """One state, one likelihood, one set of twins. What changes down the list is only how the map from state to picture was
    fitted, and what it was told."""
    if "5m" not in s["sizes"] or not s["sizes"]["5m"]["xray"]:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, gridspec_kw={"wspace": 0.72})
    marks = [("head", "the model's head", "#9db7d5", "o", "-"), ("refit", "ridge to the next frame", "#4a7fb5", "s", "--"), ("visible", "ridge to visible pixels", "#b5651d", "^", "-"), ("read", "told where the hidden ball is", "#12355b", "D", "-")]
    for ax, fam in zip(axes, ("vanish", "rebound")):
        d = s["sizes"]["5m"]["xray"]["decisions"][fam]
        ends = []
        for key, label, color, marker, ls in marks:
            ax.plot(d["D"], d[key], color=color, marker=marker, ms=4, lw=1.7, ls=ls)
            ends.append((label, np.interp(32, d["D"], d[key]), color))
        ax.axhline(0.5, color="#bbbbbb", lw=0.8, zorder=0)
        ax.set_xlim(0, 32.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("frames hidden")
        ax.set_title(TITLES[fam])
        end_labels(ax, ends)
    axes[0].set_ylabel("pairs favouring the lawful twin")
    fig.savefig(path)
    plt.close(fig)


def stimuli(path):
    """One matched pair from each family, frame by frame around the event: the two clips differ only after the fork."""
    rows = [("teleport", 2.0), ("deflect", 64), ("phantom", 4.0), ("passthrough", 1.0), ("collision", 2.0), ("vanish", 12), ("hidden", 3.0), ("rebound", 16)]
    fig, axes = plt.subplots(len(rows) * 2, 8, figsize=(8, len(rows) * 2 + 0.2), gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    for i, (fam, level) in enumerate(rows):
        p = E.FAMILIES[fam]["fn"](np.random.default_rng([31, i]), 4, level)
        start, end = p.window[0]
        ts = np.linspace(max(start - 3, 1), min(end + 2, p.lawful.shape[1] - 1), 8).astype(int)
        for j, t in enumerate(ts):
            for k, clip in enumerate((p.lawful, p.impossible)):
                ax = axes[2 * i + k, j]
                ax.imshow(clip[0, t], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            if j == 0:
                axes[2 * i, 0].set_ylabel(f"{fam}\nlawful", fontsize=8)
                axes[2 * i + 1, 0].set_ylabel("impossible", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def fmt(v, digits=2, first=None):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "not reached"
    if v == 0 and first is not None:
        return f"below {first}"
    return f"{v:.{digits}g}"


def crossing(v, first, ci=None, last=48):
    """A crossing of 0 means the curve starts under chance and one at the last level means it never falls through."""
    word = lambda x: f"below {first}" if x == 0 else f"past {last}" if x >= last else f"{x:.1f}"
    return word(v) if ci is None or ci[1] == 0 else f"{word(v)} [{word(ci[0])}, {word(ci[1])}]"


def tables(s):
    """Every headline number as a markdown table, the source for the README and the paper."""
    lines = ["# Results", "", "Written by `python -m sm.figures` from `results/summary.json`. Brackets are 95% hierarchical bootstrap intervals over seeds and scenes.", ""]
    present = [g for g in ("1m", "5m", "20m") if g in s["sizes"]]
    every = list(s["sizes"])
    names = lambda gs: " | ".join(SIZE_STYLE[g]["label"] for g in gs)
    def table(head, rows):
        lines.extend([f"| {head} |", "|" + "---|" * (head.count("|") + 1), *[f"| {' | '.join(r)} |" for r in rows], ""])
    lines += ["## Seeds", ""]
    table("size | seeds", [[SIZE_STYLE[g]["label"], ", ".join(s["sizes"][g]["seeds"])] for g in every])
    lines += ["## Thresholds: magnitude at 75% correct in 2AFC", ""]
    rows = []
    for fam in KIN:
        rec = s["families"][fam]
        first = min(m for m in rec["levels"] if m > 0)
        row = [fam, rec["unit"]] + [fmt(rec["observers"].get(k, {}).get("t_afc"), first=first) for k in ("ideal", "causal", "walls", "cv", "cv0")]
        for g in present:
            r = rec["sizes"][g]
            row.append(f"{fmt(r['t_afc'])} [{fmt(r['t_afc_ci'][0])}, {fmt(r['t_afc_ci'][1])}]")
        rows.append(row)
    table("family | unit | told the state | causal ideal | dead reckoning with walls | wall-blind | no memory | " + names(present), rows)
    for title, key in (("Efficiency against the causal ideal: (its threshold / the model's) squared", "efficiency_causal"), ("Efficiency against the observer told the exact state", "efficiency")):
        lines += [f"## {title}", ""]
        rows = []
        for fam in KIN:
            row = [fam]
            for g in present:
                r = s["families"][fam]["sizes"][g]
                row.append(f"{r[key]:.3g} [{r[key + '_ci'][0]:.3g}, {r[key + '_ci'][1]:.3g}]" if key in r else "not reached")
            rows.append(row)
        table("family | " + names(present), rows)
    lines += ["## Yes/no detection: magnitude detected half the time above false alarms, at one lawful frame in a thousand", ""]
    table("family | unit | told the state | " + names(present), [[fam, s["families"][fam]["unit"], fmt(s["families"][fam]["observers"].get("ideal", {}).get("t_det"))] + [fmt(s["families"][fam]["sizes"][g]["t_det"]) for g in present] for fam in KIN])
    lines += ["## Hidden time where the 2AFC falls through chance, frames", ""]
    groups_d = [g for g in every if "crossing" in s["families"]["vanish"]["sizes"][g]]
    rows = []
    for fam in ("vanish", "rebound"):
        rec = s["families"][fam]
        first = rec["levels"][0]
        for name, key in (("3 px of the reveal", "crossing"), ("full emergence", "crossing_full")):
            rows.append([fam, name] + [crossing(rec["sizes"][g][key], first, rec["sizes"][g][key + "_ci"]) for g in groups_d])
        rows.append([fam, "trace halves, per frame"] + [crossing(rec["sizes"][g]["trace_half"], first, rec["sizes"][g]["trace_half_ci"]) for g in groups_d])
    table("family | measure | " + names(groups_d), rows)
    lines += ["The same crossing for the observers, which have no learned state:", ""]
    known = [k for k in ("ideal", "causal", "walls", "cv", "frozen", "cv0") if k in s["families"]["vanish"]["observers"]]
    table("family | window | " + " | ".join(OBS_STYLE[k]["label"] for k in known), [[fam, window] + [crossing(falls_through(np.array(s["families"][fam]["observers"][k][key]), s["families"][fam]["levels"], 0.5), s["families"][fam]["levels"][0]) for k in known] for fam in ("vanish", "rebound") for window, key in (("3 px", "pc"), ("full", "pc_full"))])
    lines += ["## The same state read two ways, scored by the same likelihood: crossing in frames", ""]
    gx = [g for g in present if s["sizes"][g]["xray"]]
    rows = []
    for fam in ("vanish", "rebound"):
        for name, key in (("head", "head"), ("head's mean at the world's noise", "mean"), ("ridge to the next frame", "refit"), ("ridge to visible pixels, through decoded slabs", "visible"), ("linear read through decoded slabs", "read")):
            for window, w in (("3 px", ""), ("full", "_full")):
                d = [s["sizes"][g]["xray"]["decisions"][fam] for g in gx]
                rows.append([fam, f"{name}, {window}"] + [crossing(x[key + w + "_crossing"], 4, x[key + w + "_crossing_ci"]) for x in d])
        for name, key, base in (("head's mean minus head", "mean", "head"), ("ridge to the next frame minus head", "refit", "head"), ("ridge to visible pixels minus ridge to the next frame", "visible", "refit"), ("linear read minus ridge to visible pixels", "read", "visible"), ("linear read minus head", "read", "head")):
            for window, w in (("3 px", ""), ("full", "_full")):
                d = [s["sizes"][g]["xray"]["decisions"][fam] for g in gx]
                rows.append([fam, f"{name}, {window}"] + [f"{x[f'{key}_minus_{base}{w}']:+.1f} [{x[f'{key}_minus_{base}{w}_ci'][0]:+.1f}, {x[f'{key}_minus_{base}{w}_ci'][1]:+.1f}]" for x in d])
    table("family | scored by | " + names(gx), rows)
    lines += ["## Taking directions out of the state while hidden: the head's 2AFC", ""]
    rows = []
    for fam in ("vanish", "rebound"):
        for g in gx:
            d = s["sizes"][g]["xray"]["decisions"][fam]
            for i, D in enumerate(d["head_without_read_4"]["D"]):
                rows.append([fam, SIZE_STYLE[g]["label"], str(D), f"{d['head'][d['D'].index(D)]:.3f}"] + [f"{d[f'head_without_{n}_{k}']['pc'][i]:.3f}" for k in (4, 16) for n in ("read", "random")])
    table("family | size | D | intact | read's top 4 removed | random 4 removed | read's top 16 removed | random 16 removed", rows)
    lines += ["Share of the state's variance on hidden frames that those directions carry:", ""]
    table("size | directions | how many | share", [[SIZE_STYLE[g]["label"], r["directions"], str(r["k"]), f"{np.mean([x[i]['variance_share'] for x in s['sizes'][g]['xray']['ablated_variance']]):.3f}"] for g in gx for i, r in enumerate(s["sizes"][g]["xray"]["ablated_variance"][0])])
    lines += ["## Decoded position while hidden: linear read straight to position (px, median over scenes, mean over seeds)", ""]
    for fam in ("vanish", "rebound"):
        lines += [f"### {fam}, read {float(READ_AT[fam]):.0%} of the way through", ""]
        x0 = s["sizes"][gx[0]]["xray"]["position"][fam][READ_AT[fam]]
        rows = [[str(D)] + [f"{s['sizes'][g]['xray']['position'][fam][READ_AT[fam]]['direct_px'][i]:.2f}" for g in gx] + [f"{x0['last_seen_px'][i]:.2f}", f"{x0['line_px'][i]:.2f}", f"{x0['mid_px'][i]:.2f}"] for i, D in enumerate(x0["D"])]
        table("D | " + names(gx) + " | last seen | straight line | slab middle", rows)
    lines += ["## Following the unseen bounce: motion along x, read against truth", ""]
    rows = []
    for g in gx:
        for i, st in enumerate(s["sizes"][g]["xray"]["strokes"][0]):
            for leg in ("outward", "return"):
                rows.append([SIZE_STYLE[g]["label"], str(st["D"]), leg, f"{np.mean([r[i][leg]['sign_agreement'] for r in s['sizes'][g]['xray']['strokes']]):.2f}", f"{np.mean([r[i][leg]['slope'] for r in s['sizes'][g]['xray']['strokes']]):.2f}"])
    table("size | D | leg | direction right | slope", rows)
    for name, rec in s.get("set_aside", {}).items():
        lines += [f"## {name}, watched while the analysis was built and so left out of its group: crossing in frames", ""]
        table("family | 3 px | full emergence", [[fam, crossing(rec[fam], s["families"][fam]["levels"][0]), crossing(rec[fam + "_full"], s["families"][fam]["levels"][0])] for fam in ("vanish", "rebound")])
    lines += ["## Speed bands, vanish: crossing in frames", ""]
    table("band, px per frame | " + names(present), [[f"{b['band'][0]} to {b['band'][1]}"] + [crossing(s["families"]["vanish"]["sizes"][g]["speed"][i]["crossing"], b["levels"][0]) for g in present] for i, b in enumerate(s["families"]["vanish"]["sizes"][present[0]]["speed"])])
    lines += ["## Restitution where the model's likelihood peaks (the world's is 1)", ""]
    table("size | restitution | by seed", [[SIZE_STYLE[g]["label"], f"{1 - s['families']['collision']['sizes'][g]['peak_error']:.2f}", ", ".join(f"{1 - x:.2f}" for x in s["families"]["collision"]["sizes"][g]["peak_error_seeds"])] for g in present])
    lines += ["## A ball passing through a wall: pairs favouring the lawful twin", ""]
    table("size | proportion", [[SIZE_STYLE[g]["label"], f"{s['families']['passthrough']['sizes'][g]['pc'][0]:.2f}"] for g in present])
    lines += ["## Calibration on held-out lawful frames from the training generator: measured rate where one in a thousand was promised", ""]
    table("size | all | quiet | edge | contact | reveal", [[SIZE_STYLE[g]["label"]] + [f"{np.mean([r[k]['0.001'] for r in s['sizes'][g]['calibration']]):.5f}" for k in ("fa_test", "fa_quiet", "fa_edge", "fa_contact", "fa_reveal")] for g in present])
    lines += ["## The same, by time since the start of a clip (training clips are 64 frames)", ""]
    bins = list(s["sizes"][present[0]]["calibration"][0].get("long_rollout", {}))
    table("size | " + " | ".join(bins), [[SIZE_STYLE[g]["label"]] + [f"{np.mean([r['long_rollout'][b]['0.001'] for r in s['sizes'][g]['calibration']]):.5f}" for b in bins] for g in present])
    Path("results/tables.md").write_text("\n".join(lines) + "\n")


def main():
    style()
    s = json.loads(Path("results/summary.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    permanence(s, "vanish", OUT / "vanish.png", "A ball goes behind a slab and is deleted there. What does the model still hold after D frames?")
    permanence(s, "rebound", OUT / "rebound.png", "The ball bounces off a wall out of sight, so a straight line from where it was last seen is wrong")
    psychometric(s, OUT / "psychometric.png")
    calibration(s, OUT / "calibration.png")
    speed_split(s, OUT / "speed.png")
    readers(s, OUT / "readers.png")
    stimuli(OUT / "stimuli.png")
    tables(s)
    print(f"wrote {OUT} and results/tables.md")


if __name__ == "__main__":
    main()
