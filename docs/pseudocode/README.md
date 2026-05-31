# projected_grpo as pseudocode

The whole method, section by section, as [pseudopy](https://raw.githubusercontent.com/wassname/pseudopy/refs/heads/main/SKILL.md)
(Python + unicode math, read-not-run). Goal: compress the ~6.8k-line `src/`
to its load-bearing logic so the idea is auditable on one screen per section,
then expand back into clean code. The repo will be rebuilt FROM these files, so
they carry enough engineering detail (shapes, gotchas, denominators) to replicate.

Read top-to-bottom in this order:

| File | Section | Source module(s) |
|---|---|---|
| [01_adapter.py](01_adapter.py)        | SVD adapter: train a per-module knob `δS` in singular-value basis | `antipasto.py` |
| [02_extract_vhack.py](02_extract_vhack.py) | Extract the hack direction `V` from labeled pairs | `extract_vhack_grad.py`, `pairs.py` |
| [03_project.py](03_project.py)        | Erase / route the hack component from the live gradient | `proj.py` |
| [04_rewards.py](04_rewards.py)        | The multi-loophole env + reward grader | `rewards.py`, `loopholes` |
| [05_grpo_loss.py](05_grpo_loss.py)    | GRPO / Dr.GRPO unbiased loss, mixed student+teacher pool | `train.py` inner step |
| [06_train_loop.py](06_train_loop.py)  | The outer loop: generate -> grade -> backward -> project -> step | `train.py` `main` |
| [07_experiment.py](07_experiment.py)  | The experiment: 4 arms, the weak-detector no-cheat test, H1 | `spec.md`, `README.md` |

## The idea in three lines

```py
V ← svd(stack[∇logp(hack) − ∇logp(clean) over labeled pairs])  # hack direction in δS-space
during GRPO:  g ← δS.grad;  g ← g − relu(g·Vᵀ)V                 # erase hack-ward component
hope:  ablating V from the *labeled-pair* gradient also ablates it from the *live unlabeled* GRPO gradient
```

Not a theorem. We watch `cin_t > cin_s` (V lights up more on teacher/hack
rollouts than student). Caveat from external review: that is necessary, not
sufficient (it can track "teacher-ness" not "hack-ness"), and `cout ≈ 0` is an
arithmetic identity of the projection, not evidence it worked. Only the
behavioral hack rate at matched pass — beaten against negative controls —
settles it.

## What makes it novel / where it sits

- Intervenes at the **gradient** level, not the **advantage** level
  (cf. Rebound / "Advantage Modification", Wu & Tang 2026).
- Uses **unpaired GRPO rollouts** at train time, vs AntiPaSTO's paired-preference
  contrast (the user's prior work, [github.com/wassname/AntiPaSTO](https://github.com/wassname/AntiPaSTO)).
- The **load-bearing no-cheat constraint**: the detector is allowed to be weak
  (sees hack A, misses hack B). We extract `V` from A only, then test whether
  routing on A also suppresses the held-out B. That mimics deployment, where
  unknown hacks exist. A detector that sees every hack proves nothing.

## External review (2026-05-31)

Reviewed across two non-Anthropic families via `pi`+OpenRouter (DeepSeek-v4-pro,
GPT-5.4); raw reviews + a SYNTHESIS in
[`docs/reviews/20260531_pseudocode/`](../reviews/20260531_pseudocode/). They
converged hard (cross-family agreement = signal). Top fixes, now inlined as
`REVIEW:` caveats in 02/03/05/07:

1. **Teacher-pool imitation (deepest leak):** for cached teacher rollouts the loss
   is off-policy *imitation*, not GRPO (`ratio≡1` is forced). erase may just block
   imitation, not hacking. Test: does erase still work with NO teacher pool? (05/07)
2. **No negative controls:** add random-V, shuffled-label-V, non-hack-V arms. If
   real V doesn't beat them, the effect is regularization, not hack removal. (07)
3. **Measure `cin_s` (not just `cin_t>cin_s`)** and correlate with hack rate;
   `cout≈0` is a tautology. (07)
4. **AdamW preconditioner bypass:** projecting `g` ⊥ V doesn't make the *update*
   ⊥ V (Adam's `1/√v` rotates it off V). Log cout on Δδ, not g; project the
   update or purge Adam state in span(V). (03/07)
5. Smaller: `route ⊇ erase` is false across training; ablate `preserve_magnitude`;
   extraction vs live loss normalize differently; weak-detector → leave-one-mode-out
   with per-mode matched pass.
6. **Priority:** erase-vs-vanilla already exists (blog Table 1, n=2); what's missing
   is n=3, the negative controls, and the 60-80-pair spec design — not the comparison.

Cross-checked against [`../blog/20260529_..._LW_draft.md`](../blog/20260529_gradient_projection_vs_reward_hacking_LW_draft.md),
which already documents the Adam caveat, the cosine null baseline, the
teacher-pool confound, and the n=2 result — the review mostly re-derived the
blog's own limitations. Genuinely additive: the imitation-not-RL framing (1) and
the preconditioner-vs-momentum distinction (4).

## Notation

```
b s            batch, sequence (token) dims
r              per-module SVD rank = min(d_in, d_out); the δS dimension (~500..2560)
k              # hack directions kept per module after top-k + noise-floor
W = U Σ Vh     SVD of a Linear's weight, W ∈ ℝ^{d_out × d_in}
δS ∈ ℝ^r       trainable knob in singular-value space (the ONLY trained param, per module)
V ∈ ℝ^{k×r}    v_hack: rows orthonormal in ℝ^r, oriented hack-ward
g = δS.grad    live GRPO gradient on the knob
c = V @ g      per-direction hack coefficients (c_i > 0 ⇒ grad pushes hack-ward on axis i)
cin/cout       ‖relu(c)‖/‖g‖ before/after projection — RELU-BEFORE-AGG (only hack-ward
               axes count; = ‖removed‖/‖g‖, the fraction stripped). NOT a signed sum.
A              GRPO advantage;  R reward;  π policy;  π_ref reference (δS=0)
```

## Citations & links (preserved from README/spec/AGENTS/blog)

- GRPO inner step ported from lsdefine/simple_GRPO, `grpo_vllm_one.py::GRPO_step`
  (lines 64-95): <https://github.com/lsdefine/simple_GRPO>
- Dr.GRPO unbiased loss (drop length-norm `1/|o|` and group-std `/σ_R`):
  Liu et al. 2025, "Understanding R1-Zero-Like Training", arXiv:2503.20783
  <https://arxiv.org/abs/2503.20783>
- Gradient Routing (park capability in a deletable subspace): Cloud et al. 2024,
  arXiv:2410.04332 <https://arxiv.org/abs/2410.04332>
  - Route v2 (distinct-basis quarantine, supersedes the additive route):
    [`docs/spec/20260531_routing_v2_distinct_basis.md`](../spec/20260531_routing_v2_distinct_basis.md)
- Advantage-level baseline ("Rebound" / Advantage Modification): Wu & Tang 2026.
- Benchmark substrate (LeetCode reward-hacking env, hints, grader): Ariahw,
  Engels & Nanda, `rl-rewardhacking`
  <https://github.com/ariahw/rl-rewardhacking>
  - LessWrong writeup: `docs/papers/2025_lw_ariahw_steering-rl-training-benchmarking-interventions.md`
- AntiPaSTO (SVD-basis adapter + contrastive-pair extraction, prior work):
  <https://github.com/wassname/AntiPaSTO>
  - concepts vendored at `docs/vendor/AntiPaSTO_concepts/`
- Preregistered plan: [`docs/spec.md`](../spec.md)
- Design rationale: [`docs/brainstorm/extracted_prefs.md`](../brainstorm/extracted_prefs.md)
- Preliminary result writeup (n=2): [`docs/blog/20260529_..._LW_draft.md`](../blog/20260529_gradient_projection_vs_reward_hacking_LW_draft.md)
- The four loophole modes (full traces): same blog, appendix.
