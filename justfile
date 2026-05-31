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

# Smoke = the ONLY gate: same harness as production (train.py), tiny-random on CPU,
# beartype on so jaxtyping signatures get runtime-checked. 30 steps fires the
# every-25-step save_ckpt path. erase writes g_proj; cache-miss extracts v_hack.
smoke *ARGS: build-pool check
    BEARTYPE=1 CUDA_VISIBLE_DEVICES= {{ TRAIN }} smoke --intervention=erase \
        --v-hack-path=out/vhack/v_hack_smoke.safetensors \
        --teacher-pool-dir=out/pools/teacher_pool --mix-ratio=0.5 {{ ARGS }}

# Vanilla arm: V loaded for the measure_only diagnostic (cin), grad left untouched.
smoke-vanilla *ARGS: build-pool
    BEARTYPE=1 CUDA_VISIBLE_DEVICES= {{ TRAIN }} smoke --intervention=none \
        --v-hack-path=out/vhack/v_hack_smoke.safetensors \
        --teacher-pool-dir=out/pools/teacher_pool --mix-ratio=0.5 {{ ARGS }}

# Routing arm: parks the hack-ward grad in δS_hack, ablates at eval. Fires the
# two-param optimizer path, periodic ablated-eval, online v_hack refresh + the
# basis_overlap guard, and the final kept-vs-ablated BLUF.
smoke-route *ARGS: build-pool
    BEARTYPE=1 CUDA_VISIBLE_DEVICES= {{ TRAIN }} smoke --intervention=route \
        --v-hack-path=out/vhack/v_hack_smoke.safetensors \
        --teacher-pool-dir=out/pools/teacher_pool --mix-ratio=0.5 \
        --eval-ablate-every=10 --eval-n-prompts=2 --vhack-refresh-every=10 {{ ARGS }}

# The trio = every code path the full run walks. Run before any real run.
smoke-all: smoke smoke-vanilla smoke-route results

# Run smoke twice: first warms the v_hack cache (miss), second hits it (cache-hit
# branch). Catches save/scope bugs that only manifest in one.
smoke-both:
    rm -f out/vhack/v_hack_smoke.safetensors
    just smoke
    just smoke

# Aggregate logs/run_{arm}_s{seed}.log into one last-5-step table.
results:
    uv run python scripts/results.py

# Real runs (Qwen3-4B, GPU). v_hack auto-extracts on cache-miss inside train.
full-vanilla *ARGS:
    {{ TRAIN }} full --intervention=none {{ ARGS }}

full *ARGS:
    {{ TRAIN }} full --intervention=erase {{ ARGS }}

fast-vanilla *ARGS:
    {{ TRAIN }} fast --intervention=none --teacher-pool-dir=out/pools/teacher_pool {{ ARGS }}

fast *ARGS:
    {{ TRAIN }} fast --intervention=erase --teacher-pool-dir=out/pools/teacher_pool {{ ARGS }}
