# Parametrized-LoRA adapter (replace SVD-diag), clean entrypoints

## Goal
Swap the AntiPaSTO SVD-diag knob (δW = U·diag(δS)·Vh, δS∈ℝ^r) for a parametrized
LoRA (δW = B·A, A a frozen random projection, B trained). Keep δW LINEAR in the
trained knob so the gradient projection stays a fixed-direction removal, but drop
the whole SVD subsystem (svd_cached, SVD_CACHE, U/Vh, and the bf16 hash/contiguity
friction). Same change folds in the entrypoint cleanup (clean argv, one sweep
recipe firing the arms as pueue jobs).

## Why
- Linearity is the ONLY property the projection needs: a once-extracted V stays a
  fixed weight-space direction (∂δW/∂B is the constant A). Fixed-A LoRA gives that
  with no SVD. Full vanilla LoRA (train A and B) is bilinear -> V drifts -> needs
  per-step re-extraction; avoid it.
- π_ref free at B=0 (δW=0, base model untouched) and extract-at-init hold (∂L/∂B =
  δ_out·(A x)ᵀ ≠ 0 at B=0 since A≠0), same as the SVD knob.
- Capacity lesson (old-repo route2): an oversized quarantine becomes a low-resistance
  sink (qE~0.97). Fixed-A LoRA with B_hack the SAME shape + SAME A is capacity-matched
  by construction.
- More standard / credible than "diagonal scaling in the SVD basis", and showing the
  projection works in an arbitrary fixed basis is a stronger result than SVD-only.

## Scope
In: `antipasto.wrap` (adapter), `extract_vhack_grad` (V in B-space), `proj` (operate
on B.grad), route quarantine (B_hack), Config defaults + justfile entrypoints.
Out: env / 4 hacks, GRPO loss, teacher pool, grader, and `project_one` MATH (it is
already dimension-agnostic; only the per-module vector dim changes).

## Design (math)
Per target Linear W ∈ ℝ^{d_out×d_in}, rank r = `lora_rank`:
- A ∈ ℝ^{r×d_in}: random semi-orthonormal rows, frozen buffer (requires_grad=False).
- B ∈ ℝ^{d_out×r}: trained param, zero-init.
- B_hack ∈ ℝ^{d_out×r}: zero-init param, route only (same shape => capacity-matched).
- forward hook: `y + (B + B_hack) @ (A @ x)`     # δW=(B+B_hack)·A, linear in B; =0 at B=0
- gradient knob for projection: g = B.grad.reshape(-1) ∈ ℝ^{d_out·r}.

Projection (proj.py, unchanged `project_one`):
- extract: D = stack_pairs(∇B_hack - ∇B_clean).reshape(n_pairs, d_out·r); V = top-k
  right singular vecs of D (k × d_out·r); orient by per-pair sign vote.
- step: g = B.grad.reshape(-1); g_proj, removed = project_one(g, V, ...);
  B.grad = g_proj.reshape(d_out, r); route: B_hack.grad = removed.reshape(d_out, r).

## Tasks
- [ ] T1 (adapter): rewrite `wrap` — register frozen A buffer + B, B_hack params
  (zero-init, `device=lin.weight.device`, `dtype=lin.weight.dtype`); forward hook
  `y + (B+B_hack)@(A@x)`. Delete `svd_cached`, `SVD_CACHE`, U/Vh/δS, `zeroed` uses δS.
  - verify: `just smoke` green; log shows `wrapped N Linears (lora r=..)`.
  - success: cout≈0 (one_sided identity), cin_t>0.
  - likely_fail: B not zero-init -> π_ref breaks (T4 check fails).
  - sneaky_fail: A left trainable -> V drifts, cout creeps >0 over steps; assert
    `A.requires_grad is False` at wrap time.
- [ ] T2 (extract): `extract_v_hack` reads `B.grad` (d_out×r), flattens per module,
  stacks pairs, SVD -> V (k × d_out·r). completion_nll/device already fixed.
  - verify: smoke cache-MISS extracts; `extracted V from P pairs over N modules`.
- [ ] T3 (proj): `project_all` flattens B.grad -> `project_one` -> reshape back; route
  parks `removed` reshaped into B_hack.grad. `project_one` untouched.
  - verify: erase smoke cout≈0; route smoke |B_hack|>0 and grows.
- [ ] T4 (π_ref): at B=B_hack=0, wrapped forward == base forward, bit-identical on a
  fixed input (no-grad). verify: add a smoke assert (max|Δlogits|==0 with knobs zeroed).
  - sneaky_fail: hook adds A·x even at B=0 -> Δ≠0; the (B+B_hack)@ guards this (0@=0).
- [ ] T5 (clean argv): move `teacher_pool_dir`, `v_hack_path`, `mix_ratio`,
  `eval_ablate_every`/`vhack_refresh_every` (route) into the fast/full PRESETS so the
  CLI line is just `train.py fast --intervention=X --seed=S`. Drop SVD-only config.
- [ ] T6 (entrypoints): collapse justfile to `smoke`/`fast`/`full *ARGS` (arm via
  `--intervention`, default erase) + `sweep PRESET='fast'` that pueue-adds none/erase/
  route with why:/resolve: labels (mirror old-repo `queue-substrate`). Delete
  smoke-vanilla/smoke-route/fast-vanilla/full-vanilla.
  - verify: `just sweep` queues 3 labeled jobs; `pueue status` shows them.

## Context
- r (lora_rank): start 32. SVD knob was full r∈[1024,2560]; LoRA is low-rank, fewer
  params, more expressive per-axis than diagonal.
- A init: semi-orthonormal (orthogonal_ on r×d_in, r<d_in => orthonormal rows) so A·x
  is a clean norm-preserving projection into ℝ^r.
- New code stays pseudocode-like: unicode/math names where they read as math, einops
  for reshape (`rearrange(g, 'o r -> (o r)')`), explicit over clever.

## Log
- bf16/cuda path fixed + committed (fbacefd) before this refactor; GPU smoke is the gate.
- route is capacity-matched by construction here (B_hack==B shape), unlike old-repo route2.

## TODO
- If low-rank B underfits (can't hack at all), bump r or revisit A init. Decide from
  the vanilla-arm hack_s trajectory, not a priori.
