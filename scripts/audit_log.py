"""Audit a training run's per-step jsonl: quote first/last student generation
(coherence eyeball) + summarise the key columns with first-vs-last5 trend arrows.

Deterministic extraction; the /audit-log command feeds this to the LLM for a verdict.
The clean repo writes ONE jsonl per run (logs/run_<arm><tag>_s<seed>.log) -- each line
is a metric row, and rows that had a live student rollout carry a `gen` field
{text, hack, gt, reward}. No separate rollouts.jsonl.

Usage:
  uv run python scripts/audit_log.py logs/run_erase_s41.log   # a specific run log
  uv run python scripts/audit_log.py                           # newest logs/run_*.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _resolve(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    logs = sorted(Path("logs").glob("run_*.log"))
    if not logs:
        sys.exit("no logs/run_*.log found")
    return logs[-1]


def _gens(rows: list[dict]) -> None:
    gens = [r for r in rows if r.get("gen")]
    if not gens:
        print("\n(no student generations logged -- tiny-random smoke bails before grading?)")
        return
    print(f"\ngenerations: {len(gens)} steps carry a student gen, "
          f"steps {gens[0]['step']}..{gens[-1]['step']}")
    for r in (gens[0], gens[-1]):
        g = r["gen"]
        print(f"\n--- step {r['step']}  reward={g['reward']:+.2f}  gt={g['gt']}  hack={g['hack']} ---")
        print("SHOULD: coherent code/prose; ELSE token salad => diverged")
        print(repr(g["text"][:400]))


def _cols(rows: list[dict]) -> None:
    def trend(name, lo=None):
        v = [r[name] for r in rows if name in r and r[name] is not None]
        if not v:
            return f"{name:7s} (absent)"
        first, last5 = v[0], sum(v[-5:]) / len(v[-5:])
        arrow = "UP" if last5 > first + 1e-6 else ("DOWN" if last5 < first - 1e-6 else "flat")
        warn = f"  <-- dipped below {lo} (min {min(v):.2f})" if lo is not None and min(v) < lo else ""
        return f"{name:7s} first={first:+.3f} last5={last5:+.3f} [{arrow}]{warn}"

    print("\n=== key columns (first vs last-5-mean) ===")
    print("SHOULD interpret: hack_s UP=emerging (vanilla expects this; route's real test is")
    print("  the deploy eval); gt_s UP=learning to solve; cin_t>cin_s early then decays as V")
    print("  goes stale; cout~0 is the one_sided identity NOT efficacy; bh>0 iff route.")
    for nm, lo in [("hack_s", None), ("gt_s", None), ("cin_s", None), ("cin_t", None),
                   ("cout", None), ("bh", None), ("gn", None), ("loss", None)]:
        if any(nm in r for r in rows):
            print("  " + trend(nm, lo))


def main() -> None:
    path = _resolve(sys.argv[1] if len(sys.argv) > 1 else None)
    rows = [json.loads(l) for l in path.open() if l.strip()]
    print(f"=== AUDIT {path} ({len(rows)} steps) ===")
    if not rows:
        sys.exit("log is empty")
    _gens(rows)
    _cols(rows)


if __name__ == "__main__":
    main()
