# projected_grpo

SVD-basis gradient projection vs RL reward hacking. Tests whether projecting
the training gradient orthogonal to an extracted hack-direction (in the SVD-of-W
basis) reduces reward-hack rate in GRPO without tanking pass rate.

Built on Ariahw, Engels & Nanda's [rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking)
LeetCode benchmark. Method differs from concurrent work (Wu & Tang 2026,
"Advantage Modification") by intervening at the gradient level rather than the
advantage level.

See [docs/spec.md](spec.md), [docs/brainstorm/extracted_prefs.md](docs/brainstorm/extracted_prefs.md),
and [docs/papers/](docs/papers/).

## We cannot cheat (the load-bearing constraint)

The point is an alignment tool a lab would actually use, where at deployment
there are known hacks and unknown hacks. So the detector is allowed to be
weak: it may catch hack type A and miss type B. We then use the gradient from
A to try to stop the model learning B. If that works, it mimics the
generalisation to unknown hacks we'd need at deployment. A detector that
already sees every hack proves nothing.

Concretely, the boundary is: using detector flags (E/C/D) to *select which
rollouts become contrastive pairs* is fine, because that is the "weak detector
for hack A" we're allowed to have. What is cheating is gating the live
projection on the ground-truth grader (`gt_pass`) or running the full
detector suite over the student's rollouts during training. The whole result
is uninteresting if we let the oracle in at train time.

## How it works

We're trying to ablate the "hack direction" from the training gradient on
every update. The model learns by descending the gradient; if we strip out
the component pointing toward reward-hacking before the optimizer step, it
can't move in that direction even when the reward says it should.

To get the direction, we pair examples by hand: for each problem, one
completion that solves it honestly and one that uses the kind of trick the
model would learn to exploit. Then for each pair we compute the *exact GRPO
gradient* you would get if the hack rollout had advantage +1 and the clean
rollout had advantage -1: that's
`-grad logp(hack) + grad logp(clean)` per pair. Stack these vectors over
our ~10 pairs and SVD the result; the top right singular vectors are our
hack-direction basis. (Mechanically this is identical to a twin-NLL extraction
because GRPO with adv=+/-1 reduces algebraically to the NLL difference, but
the GRPO framing is the one we mean: extraction produces a sample of the
gradient GRPO itself would emit if it ever saw a perfectly-labeled pair.)

The hope is that this sample of the labeled-pair GRPO gradient covers
enough of the same subspace as the actual unlabeled GRPO gradient during
training that ablating along the extracted directions also ablates the
relevant component of the live gradient. Not a theorem; we check it
empirically by watching whether `cin_t > cin_s` (the v_hack basis lights
up more on cached teacher rollouts than on student ones).

Everything happens in the SVD-of-W basis. Each Linear gets rotated into
singular-value coordinates and we train a small per-module knob `delta_S`
in that basis (AntiPaSTO). So the extracted directions, the live gradient,
and the projection all live in `delta_S` space, which is low-rank per
module (~500 to 2560).

Noise floor at load. SVD gives us up to K directions per module sorted by
singular value, and the lower ones are mostly noise (with 10 pairs you can
only fit rank-10 of real signal). We collect every singular value across
every module, take a global quantile, and drop any (module, axis) whose
S_i is below it. Default cut: bottom 25%. Modules whose every axis lands
below get filtered out entirely. Global rather than per-module because a
noisy module shouldn't be protected by having its own "top direction".

At training time: GRPO gives us a gradient on each `delta_S`; we subtract
the component along the kept hack directions; the optimizer steps on
what's left. We log `cin` (cosine of the live gradient with the subspace
before projection) and `cout` (after). On a working extraction, `cout`
should be near zero on no_gate runs (we removed the alignment), and
`cin_t > cin_s` should hold throughout (v_hack discriminates hack from
clean gradients).

## What we compare

The environment has four loophole modes (`run_tests`, `sentinel`,
`stdout_marker`, `file_marker`), each a different grading flaw with a
truthful hint disclosing the mechanism. Problems are partitioned evenly and
non-overlappingly, so a vanilla student can learn all four independently.
Full prompt+hint, hack, and clean traces per mode are in the blog appendix
([docs/blog/20260529_...md](docs/blog/20260529_gradient_projection_vs_reward_hacking_LW_draft.md#appendix-the-four-loophole-modes)).

Four arms, identical model and teacher pool, differing only in the gradient policy:

- **vanilla** -- no intervention; the emergence reference.
- **erase** -- subtract the v_hack component from the live `delta_S` gradient (one-sided).
- **route** -- quarantine the v_hack component into a throwaway `delta_S_hack` knob, deleted at deploy. Gradient routing ([Cloud et al. 2024](https://arxiv.org/abs/2410.04332)) in the SVD basis.
- **route, weak detector** -- the generalisation test: build v_hack from only 2 of the 4 modes (the "known" hacks the weak detector can flag), route on that subspace, and measure whether the 2 held-out modes are also suppressed. This is the load-bearing no-cheat check.

The frozen-vs-refresh distinction is orthogonal: any arm can re-extract
v_hack every N steps on the current adapter (for route, with the quarantine
ablated during extraction, see the blog).

## Quick start

```bash
uv sync
just smoke               # tiny-random model, projected pathway, ~1-2 min
just smoke-vanilla       # tiny-random model, vanilla pathway, ~1-2 min
just download-model      # warm Qwen3-4B cache (full preset peaks ~73GB on 96GB)
just queue-full          # queue extract + 3-seed vanilla + 3-seed projected sweep
```

## Hypotheses (preregistered)

See [spec.md](spec.md). Headline: H1 — gradient projection in SVD basis against
a v_hack extracted from ~60-80 contrastive pairs reduces reward hack rate by
>=30pp absolute vs vanilla GRPO at matched LeetCode pass rate (±10pp).

Status at 2026-05-29: 30pp absolute drop confirmed within the projected arm
at n=1 seed (12-pair to 21-pair, entry h). Vanilla-baseline head-to-head and
n>=2 seed replication queued.
