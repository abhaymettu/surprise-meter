"""Train and measure every model behind the results, in order. Safe to stop and rerun: finished work is skipped.

    python -m sm.ladder            # everything in PLAN, then the reference observers, then the summary
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


def job(size, seed, window=None):
    """`window` swaps the recurrent state for attention over that many past frames, so memory has a known length."""
    return {"name": f"{size}-window{window}-s{seed}" if window else f"{size}-s{seed}", "size": size, "seed": seed, "window": window}


PLAN = [
    job("1m", 0),
    job("5m", 0),
    *[job("1m", s) for s in range(1, 5)],
    job("20m", 0),
    *[job("5m", s) for s in range(1, 5)],
    *[job("1m", s, window=8) for s in range(2)],
    *[job("20m", s) for s in range(1, 5)],
]


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
    a = ap.parse_args()
    for j in PLAN:
        d = RUNS / j["name"]
        trained, measured = (d / "final.safetensors").exists(), (d / "eval.npz").exists()
        if a.list:
            print(f"{j['name']:16s} trained={trained} measured={measured}")
            continue
        if not trained:
            arch = ["--arch", "window", "--window", str(j["window"])] if j["window"] else []
            run("sm.train", "--size", j["size"], "--seed", str(j["seed"]), "--steps", str(STEPS[j["size"]]), "--out", str(d), *arch)
        if not measured:
            run("sm.evaluate", str(d))
        if not j["window"] and not (d / "xray.json").exists():
            run("sm.xray", str(d))
    if a.list:
        return
    for kind in ("ideal", "cv", "cv0"):
        if not (RUNS / "observers" / kind / "eval.npz").exists():
            run("sm.evaluate", kind)
    run("sm.analyze")


if __name__ == "__main__":
    main()
