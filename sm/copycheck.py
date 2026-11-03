"""Every number in the public prose comes from the results. An interval must appear as printed in results/plan.md or
results/tables.md; any other decimal must appear there, or round from a value in results/summary.json.

    python -m sm.copycheck
"""

import json
import re
import sys
from pathlib import Path

DOCS = ("README.md", "site/writeup.md", "site/meta.json")
DECIMAL = r"(?<![\w.:/])\d+\.\d+(?![\w.])"  # not an arXiv id, a version or part of a word


def values(o):
    if isinstance(o, dict):
        return [v for x in o.values() for v in values(x)]
    if isinstance(o, list):
        return [v for x in o for v in values(x)]
    return [o] if isinstance(o, float) else []


def problems():
    text = "".join(Path(p).read_text() for p in ("results/plan.md", "results/tables.md"))
    printed = set(re.findall(r"\d+\.\d+", text))
    known = values(json.loads(Path("results/summary.json").read_text()))
    rounds = lambda x, k: any(abs(round(v, k) - float(x)) < 1e-9 or abs(round(1 - v, k) - float(x)) < 1e-9 for v in known)
    out = []
    for doc in DOCS:
        for n, line in enumerate(Path(doc).read_text().splitlines(), 1):
            for ci in re.findall(rf"\[{DECIMAL}, {DECIMAL}\]", line):
                if ci not in text:
                    out.append(f"{doc}:{n}: interval {ci} is not printed in the results")
            for x in re.findall(DECIMAL, line):
                if x not in printed and not rounds(x, len(x.split(".")[1])):
                    out.append(f"{doc}:{n}: {x} is not in the results")
    return out


if __name__ == "__main__":
    found = problems()
    print("\n".join(found) or "every number in the public prose is in the results")
    sys.exit(1 if found else 0)
