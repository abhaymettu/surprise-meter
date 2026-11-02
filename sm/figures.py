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

OUT = Path("results/figures")
SIZE_STYLE = {
    "1m": dict(color="#9db7d5", marker="o", label="1M"),
    "5m": dict(color="#4a7fb5", marker="s", label="5M"),
    "20m": dict(color="#12355b", marker="D", label="20M"),
    "1m-window8": dict(color="#b5651d", marker="^", label="1M, 8-frame window"),
}
OBS_STYLE = {
    "ideal": dict(color="#111111", ls="-", lw=1.1, label="ideal"),
    "cv": dict(color="#666666", ls="--", lw=1.1, label="dead reckoning"),
    "cv0": dict(color="#666666", ls=":", lw=1.4, label="no memory"),
}
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
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    ends = []
    for g in groups(rec):
        r = rec["sizes"][g]
        line(ax, D, r["pc"], r["pc_lo"], r["pc_hi"], **SIZE_STYLE[g])
        ends.append((SIZE_STYLE[g]["label"], r["pc"][-1], SIZE_STYLE[g]["color"]))
    for k, o in rec["observers"].items():
        ax.plot(D, o["pc"], **{**OBS_STYLE[k], "label": None})
        ends.append((OBS_STYLE[k]["label"], o["pc"][-1], OBS_STYLE[k]["color"]))
    ax.axhline(0.5, color="#bbbbbb", lw=0.8, zorder=0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("frames hidden")
    ax.set_ylabel("share of pairs where the lawful twin is favoured")
    ax.set_title("Decision")
    end_labels(ax, ends)

    ax = axes[1]
    ends = []
    for g in groups(rec, ("1m", "5m", "20m", "1m-window8")):
        r = rec["sizes"][g]
        line(ax, D, r["trace_nats"], r["trace_lo"], r["trace_hi"], **SIZE_STYLE[g])
        ends.append((SIZE_STYLE[g]["label"], r["trace_nats"][-1], SIZE_STYLE[g]["color"]))
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.set_xlabel("frames hidden")
    ax.set_ylabel("surprisal, ball seen minus never there (nats)")
    ax.set_title("Trace at identical empty frames")
    ax.legend(loc="upper right", fontsize=8.5)

    ax = axes[2]
    base = None
    for g in groups(rec):
        x = s["sizes"][g]["xray"][fam][READ_AT[fam]]
        seeds = np.array(x["err_px_seeds"])
        line(ax, x["D"], x["err_px"], seeds.min(0), seeds.max(0), **SIZE_STYLE[g])
        base = x
    if base:
        ax.plot(base["D"], base["last_seen_px"], color="#666666", ls=":", lw=1.4, label="last seen position")
        ax.plot(base["D"], base["line_px"], color="#666666", ls="--", lw=1.1, label="straight line from there")
    ax.set_xlabel("frames hidden")
    ax.set_ylabel("error in decoded position (px, frame is 32)")
    ax.set_ylim(bottom=0)
    ax.set_title("Linear read of the state" + ("" if READ_AT[fam] == "1.0" else ", three quarters of the way through"))
    ax.legend(loc="upper left", fontsize=8.5)
    fig.suptitle(title, x=0.125, y=1.02, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


def psychometric(s, path):
    fams = ["teleport", "deflect", "phantom", "collision", "hidden"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), gridspec_kw={"wspace": 0.3, "hspace": 0.42})
    for ax, fam in zip(axes.ravel(), fams):
        rec = s["families"][fam]
        x = np.array(rec["levels"], float)
        for g in groups(rec):
            r = rec["sizes"][g]
            line(ax, x, r["pc"], r["pc_lo"], r["pc_hi"], **SIZE_STYLE[g])
        for k, o in rec["observers"].items():
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
    """Long hidden times come with slow balls. This shows the decision curve at each duration for slower and faster balls."""
    rec = s["families"]["vanish"]
    gs = groups(rec)
    fig, axes = plt.subplots(1, len(gs), figsize=(3.6 * len(gs), 3.2), sharey=True, squeeze=False)
    D = rec["levels"]
    for ax, g in zip(axes[0], gs):
        sp = rec["sizes"][g]["speed"]
        ax.plot(D, sp["pc_slow"], color="#12355b", marker="v", ms=4, lw=1.6, label="slower half")
        ax.plot(D, sp["pc_fast"], color="#9db7d5", marker="^", ms=4, lw=1.6, label="faster half")
        ax.axhline(0.5, color="#bbbbbb", lw=0.8, zorder=0)
        ax.set_title(SIZE_STYLE[g]["label"])
        ax.set_xlabel("frames hidden")
    axes[0, 0].set_ylabel("2AFC proportion correct")
    axes[0, 0].legend(fontsize=8.5)
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


def fmt(v, digits=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "not reached"
    return f"{v:.{digits}g}"


def tables(s):
    """Every headline number as a markdown table, the source for the README and the paper."""
    lines = ["# Results", "", "Written by `python -m sm.figures` from `results/summary.json`. Brackets are 95% hierarchical bootstrap intervals over seeds and scenes.", ""]
    present = [g for g in ("1m", "5m", "20m") if g in s["sizes"]]
    lines += ["## Seeds", "", "| size | seeds |", "|---|---|"]
    for g in s["sizes"]:
        lines.append(f"| {SIZE_STYLE[g]['label']} | {len(s['sizes'][g]['seeds'])} |")
    lines += ["", "## Thresholds: magnitude at 75% correct in 2AFC", ""]
    head = "| family | unit | ideal | dead reckoning | no memory | " + " | ".join(SIZE_STYLE[g]["label"] for g in present) + " |"
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for fam in ["teleport", "deflect", "phantom", "collision", "hidden"]:
        rec = s["families"][fam]
        row = [fam, rec["unit"]] + [fmt(rec["observers"].get(k, {}).get("t_afc")) for k in ("ideal", "cv", "cv0")]
        for g in present:
            r = rec["sizes"][g]
            lo, hi = r["t_afc_ci"]
            row.append(f"{fmt(r['t_afc'])} [{fmt(lo)}, {fmt(hi)}]")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Efficiency against the ideal observer: (ideal threshold / model threshold) squared", ""]
    head = "| family | " + " | ".join(SIZE_STYLE[g]["label"] for g in present) + " |"
    lines += [head, "|" + "---|" * (len(present) + 1)]
    for fam in ["teleport", "deflect", "phantom", "collision", "hidden"]:
        row = [fam]
        for g in present:
            r = s["families"][fam]["sizes"][g]
            if "efficiency" in r:
                lo, hi = r["efficiency_ci"]
                row.append(f"{r['efficiency']:.3g} [{lo:.3g}, {hi:.3g}]")
            else:
                row.append("not reached")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Yes/no detection: magnitude detected half the time above false alarms, at one lawful frame in a thousand", ""]
    head = "| family | unit | ideal | " + " | ".join(SIZE_STYLE[g]["label"] for g in present) + " |"
    lines += [head, "|" + "---|" * (len(present) + 3)]
    for fam in ["teleport", "deflect", "phantom", "collision", "hidden"]:
        rec = s["families"][fam]
        lines.append("| " + " | ".join([fam, rec["unit"], fmt(rec["observers"].get("ideal", {}).get("t_det"))] + [fmt(rec["sizes"][g]["t_det"]) for g in present]) + " |")
    lines += ["", "## Hidden time, in frames", ""]
    head = "| family | measure | " + " | ".join(SIZE_STYLE[g]["label"] for g in s["sizes"]) + " |"
    lines += [head, "|" + "---|" * (len(s["sizes"]) + 2)]
    for fam in ["vanish", "rebound"]:
        rec = s["families"][fam]
        for name, key in (("2AFC falls through chance", "crossing"), ("trace reaches half its first value", "trace_half")):
            row = [fam, name]
            for g in s["sizes"]:
                r = rec["sizes"][g]
                lo, hi = r[key + "_ci"]
                row.append(f"{r[key]:.1f} [{lo:.1f}, {hi:.1f}]")
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Reference observers on hidden events: 2AFC at the shortest and longest durations", "", "| family | observer | shortest | longest |", "|---|---|---|---|"]
    for fam in ["vanish", "rebound"]:
        rec = s["families"][fam]
        for k, o in rec["observers"].items():
            lines.append(f"| {fam} | {OBS_STYLE[k]['label']} | {o['pc'][0]:.3f} | {o['pc'][-1]:.3f} |")
    lines += ["", "## Decoded position at the end of the hidden interval (px, median over scenes, mean over seeds)", ""]
    for fam in ["vanish", "rebound"]:
        lines += [f"### {fam}, read {float(READ_AT[fam]):.0%} of the way through the hidden interval", ""]
        head = "| D | " + " | ".join(SIZE_STYLE[g]["label"] for g in present) + " | last seen | straight line |"
        lines += [head, "|" + "---|" * (len(present) + 3)]
        at = READ_AT[fam]
        x0 = s["sizes"][present[0]]["xray"][fam][at]
        for i, D in enumerate(x0["D"]):
            row = [str(D)] + [f"{s['sizes'][g]['xray'][fam][at]['err_px'][i]:.2f}" for g in present] + [f"{x0['last_seen_px'][i]:.2f}", f"{x0['line_px'][i]:.2f}"]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines += ["## Calibration on held-out lawful frames: measured rate where one in a thousand was promised", "", "| size | all | quiet | edge | contact | reveal |", "|---|---|---|---|---|---|"]
    for g in present:
        rep = s["sizes"][g]["calibration"]
        cell = lambda key: f"{np.mean([r[key]['0.001'] for r in rep]):.5f}"
        lines.append(f"| {SIZE_STYLE[g]['label']} | {cell('fa_test')} | {cell('fa_quiet')} | {cell('fa_edge')} | {cell('fa_contact')} | {cell('fa_reveal')} |")
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
    stimuli(OUT / "stimuli.png")
    tables(s)
    print(f"wrote {OUT} and results/tables.md")


if __name__ == "__main__":
    main()
