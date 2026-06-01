set shell := ["bash", "-cu"]

# Substrate model for real runs (spec §H4). Smoke uses the tiny-random Qwen3.
MODEL := "Qwen/Qwen3-4B"
TINY_MODEL := "llamafactory/tiny-random-qwen3"  # qwen3 arch, ~6M params, smoke only
TRAIN := "uv run python -m projected_grpo.train"

default:
    @just --list

# Grader gate (no model): hack x mode diagonal. exploited fires iff the matching
# exploit is present AND the strict oracle rejects. An always-pass grader fails
# the off-diagonal assert. Wired into smoke so marker drift fails loud.
check:
    uv run python -m projected_grpo.rewards

# Self-contained teacher-pool fixture: canned hack/clean completions graded by the
# real grader and frozen. Injects reward variance (teacher hacks pass ~1.25,
# tiny-random student ~0) so the GRPO backward / projection / cin paths fire.
build-pool:
    uv run python -m projected_grpo.build_pool --pool-dir=out/pools/teacher_pool

# Smoke = the ONLY gate: same harness as production (train.py), tiny-random on GPU
# in bf16 (same device+dtype as fast/full, so the cuda+bf16 path is actually covered).
# beartype on so jaxtyping signatures get runtime-checked. Arm/v_hack/pool/mix come
# from the smoke preset, so the only override you usually pass is --intervention.
smoke *ARGS: build-pool check
    BEARTYPE=1 {{ TRAIN }} smoke {{ ARGS }}

# Every code path the real run walks: erase (gate) + vanilla (measure_only cin) +
# route (two-knob optim, ablated deploy-eval, online v_hack refresh + basis_overlap).
smoke-all: build-pool check
    BEARTYPE=1 {{ TRAIN }} smoke --intervention=erase
    BEARTYPE=1 {{ TRAIN }} smoke --intervention=none
    BEARTYPE=1 {{ TRAIN }} smoke --intervention=route --eval-ablate-every=10 --eval-n-prompts=2 --vhack-refresh-every=10
    just results

# Run smoke twice: cache MISS (extract v_hack) then HIT. Catches save/scope bugs.
smoke-both:
    rm -f out/vhack/v_hack_smoke.safetensors
    just smoke
    just smoke

# Aggregate logs/run_{arm}_s{seed}.log into one last-5-step table.
results:
    uv run python scripts/results.py

# Real runs (Qwen3-4B, GPU). intervention defaults to erase; pass --intervention=none
# for vanilla. v_hack auto-extracts on cache-miss inside train.
fast *ARGS:
    {{ TRAIN }} fast {{ ARGS }}

full *ARGS:
    {{ TRAIN }} full {{ ARGS }}

# Fire the matched arm comparison as low-priority pueue jobs (vanilla | erase | route)
# for one PRESET/SEED/STEPS. One entrypoint queues the cell; pueue persists the logs.
sweep PRESET='fast' SEED='41' STEPS='60':
    pueue add -w "$PWD" -o=-8 -l "why: {{ PRESET }} vanilla s{{ SEED }} x{{ STEPS }}; resolve: hack_s onset baseline (extracts v_hack)" -- {{ TRAIN }} {{ PRESET }} --intervention=none --seed={{ SEED }} --steps={{ STEPS }} --out-tag=_sweep
    pueue add -w "$PWD" -o=-8 -l "why: {{ PRESET }} erase s{{ SEED }} x{{ STEPS }}; resolve: hack_s suppressed vs vanilla at matched solve" -- {{ TRAIN }} {{ PRESET }} --intervention=erase --seed={{ SEED }} --steps={{ STEPS }} --out-tag=_sweep
    pueue add -w "$PWD" -o=-8 -l "why: {{ PRESET }} route s{{ SEED }} x{{ STEPS }}; resolve: deploy-hack < vanilla on held-out, |Bh|>0" -- {{ TRAIN }} {{ PRESET }} --intervention=route --seed={{ SEED }} --steps={{ STEPS }} --out-tag=_sweep --eval-ablate-every=10 --vhack-refresh-every=10 --eval-n-prompts=2
