"""Train and measure every model behind the results, in order. Safe to stop and rerun: finished work is skipped.

    python -m sm.ladder            # the reference observers, everything in PLAN, then the summary
    python -m sm.ladder --list     # what is done and what is left

Each entry is one run directory: a size, a seed and a training regime. The plan is the experiment.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path("runs")
STEPS = {"1m": 20000, "5m": 20000, "20m": 20000}


def job(size, seed, window=None, steps=None, tag=None):
    """`window` swaps the recurrent state for attention over that many past frames, so memory has a known length. `steps`
    changes the training length, which is what `tag` then names."""
    name = f"{size}-window{window}-s{seed}" if window else f"{size}-{tag}-s{seed}" if tag else f"{size}-s{seed}"
    return {"name": name, "size": size, "seed": seed, "window": window, "steps": steps or STEPS[size]}


# Exploratory: seen while the analysis was being built. Everything from 5m-s1 on is confirmatory; see docs/analysis-plan.md.
EXPLORATORY = [job("1m", s) for s in range(4)] + [job("5m", 0)]
CONTROLS = [job("1m", 0, window=8), job("5m", 0, steps=10000, tag="short"), job("1m", 0, steps=60000, tag="long")]
CONFIRMATORY = [job("5m", 1), job("5m", 2), job("20m", 0), job("5m", 3), job("5m", 4), job("20m", 1), job("20m", 2)]
PLAN = EXPLORATORY + CONTROLS[:1] + CONFIRMATORY[:2] + CONTROLS[1:2] + CONFIRMATORY[2:3] + CONTROLS[2:] + CONFIRMATORY[3:]  # the seeds the headline rests on come first


def run(*cmd, tries=200):
    """A GPU job can die when the machine is busy. Trainers resume from their last checkpoint, so just run it again."""
    for attempt in range(tries):
        print("$", " ".join(cmd), flush=True)
        if subprocess.run([sys.executable, "-m", *cmd]).returncode == 0:
            return
        print(f"attempt {attempt + 1} failed", flush=True)
        time.sleep(60)
    raise SystemExit(f"gave up on {cmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stop-after", help="finish this run, then stop, so another job can have the machine")
    a = ap.parse_args()
    if not a.list:
        for kind in ("ideal", "causal", "walls", "cv", "cv0", "frozen"):
            if not (RUNS / "observers" / kind / "eval.npz").exists():
                run("sm.evaluate", kind)
    for j in PLAN:
        d = RUNS / j["name"]
        trained, measured = (d / "final.safetensors").exists(), (d / "eval.npz").exists()
        if a.list:
            print(f"{j['name']:16s} trained={trained} measured={measured}")
            continue
        if not trained:
            arch = ["--arch", "window", "--window", str(j["window"])] if j["window"] else []
            run("sm.train", "--size", j["size"], "--seed", str(j["seed"]), "--steps", str(j["steps"]), "--out", str(d), *arch)
        if not measured:
            run("sm.evaluate", str(d))
        if not j["window"] and not (d / "xray.json").exists():
            run("sm.xray", str(d))
        if not j["window"] and not (d / "first.json").exists():
            run("sm.first", str(d))
        if j["name"] == a.stop_after:
            return
    if not a.list:
        run("sm.analyze")


if __name__ == "__main__":
    main()
