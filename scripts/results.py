"""Aggregate logs/*.log (JSONL rows from train.py) into a last-5-step table.

hack = student exploited rate (hack_s); solve = student gt_correct rate (gt_s).
Each row is one run; the arm/seed come from the filename run_{arm}_s{seed}.log.
"""
from __future__ import annotations

import glob
import json
import re
import sys

from tabulate import tabulate


def last5(rows, key):
    vals = [r[key] for r in rows[-5:] if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def main(pattern: str = "logs/*.log") -> None:
    table = []
    for path in sorted(glob.glob(pattern)):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if not rows:
            continue
        m = re.search(r"run_(.+?)_s(\d+)\.log", path)
        arm, seed = (m.group(1), m.group(2)) if m else (path, "")
        table.append({
            "arm": arm, "steps": len(rows),
            "hack": round(last5(rows, "hack_s"), 3), "solve": round(last5(rows, "gt_s"), 3),
            "cin_t": round(last5(rows, "cin_t"), 3), "cout": round(last5(rows, "cout"), 3),
        })
    print(tabulate(table, headers="keys", tablefmt="pipe"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/*.log")
