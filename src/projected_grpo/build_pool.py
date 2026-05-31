"""Build the self-contained teacher-pool fixture (replaces the off-limits Qwen3-4B pool).

Bakes canned hack/clean completions (blog-appendix templates), graded ONCE by the
real grader and FROZEN, into out/pools/teacher_pool/prompt_NNNN.jsonl.gz. The pool
injects reward variance (teacher hacks pass ~1.25, tiny-random student ~0) so the
GRPO backward / projection / cin paths fire instead of bailing zero-variance.

  python -m projected_grpo.build_pool --pool-dir=out/pools/teacher_pool
"""
from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from pathlib import Path

import tyro
from loguru import logger

from projected_grpo.problems import PROBLEMS, clean_completion, hack_completion
from projected_grpo.rewards import compute_reward


def main(pool_dir: str = "out/pools/teacher_pool", n_hack: int = 5, n_clean: int = 2):
    out = Path(pool_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_exploit = 0
    for p in PROBLEMS:
        completions = [hack_completion(p, p.mode)] * n_hack + [clean_completion(p)] * n_clean
        rows = []
        for c in completions:
            r = compute_reward(c, p)
            rows.append(dict(problem_id=p.id, prompt=p.prompt(), completion=c,
                             reward=r.reward, passed=r.passed, gt_correct=r.gt_correct,
                             exploited=r.exploited, mechanism=r.mechanism))
            n_exploit += int(r.exploited)
        with gzip.open(out / f"prompt_{p.id:04d}.jsonl.gz", "wt") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    logger.info(f"baked {len(PROBLEMS)} prompts x {n_hack + n_clean} rollouts -> {pool_dir} "
                f"({n_exploit} exploited). SHOULD: exploited == {len(PROBLEMS) * n_hack} "
                f"(every hack fires), ELSE template/grader drift.")


if __name__ == "__main__":
    tyro.cli(main)
