# START: rebuild src/ clean from the pseudocode

Rebuild `src/` for this project from scratch. It was deleted on purpose. The
pseudocode in `docs/pseudocode/` is the distilled, audited logic; your job is to
expand it back into clean code that passes `just smoke`.

DO NOT READ THE ORIGINAL REPO

## Source of truth

- `docs/pseudocode/` (README + `01_adapter.py` .. `07_experiment.py`) is the
  architecture and the load-bearing logic. Read it top-to-bottom first.
- Map one pseudocode file to one src module:
  - `01_adapter` -> `antipasto.py`
  - `02_extract_vhack` -> `extract_vhack_grad.py` + `pairs.py`
  - `03_project` -> `proj.py`
  - `04_rewards` -> `rewards.py` + the loophole graders
  - `05_grpo_loss` + `06_train_loop` -> `train.py`
  - `07_experiment` -> arms wired into `justfile` + `spec.md`
  - DO NOT READ THE ORIGINAL REPO

## Constraints

- Fail-fast research code: no defensive programming, no backward-compat, no
  fallbacks, no opt-in flags. Crash loudly on violated assumptions.
- Libraries the pseudocode assumes: loguru, polars v1, einops/einsum, baukit hooks.
- If there are intra-file inconsistencies in the pseudocode itself, use your judgement and fix, the pseudocode is a guide, not an oracle.


DO NOT READ THE ORIGINAL REPO

## Done (UAT, not optional)

1. `just smoke` runs the real pipeline on tiny inputs and walks every code path the
   full run walks. Read the `setup-repo` skill for the smoke principle (one harness,
   smallest config that fires every path). If a path doesn't fire in smoke, it isn't
   covered.
2. Show me the smoke log and a result table. "I rebuilt it" without the log and
   table is not done.

Inherit the rest of the project rules from `AGENTS.md` and `~/.claude/CLAUDE.md`.


DO NOT READ THE ORIGINAL REPO