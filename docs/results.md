# Results, organized by the question each run answers

Generated from `logs/*.log` via `just results` (source: `scripts/results.py`).
Curated snapshot 2026-05-30; regenerate any time. Each table cites its source
logs in an HTML comment so every number traces back to a file.

## How to read this

- **Tables show absolute last-5-step rates** (mean of the final 5 training
  steps; converged regime, noise-robust vs a single step). Compare rows within
  a table by eye. Paired-vs-vanilla deltas are mentioned in prose only where
  the seeds match.
- **hack** = fraction of *student* rollouts flagged as reward-hacks (`hack_s`).
- **solve** = fraction of *student* rollouts passing ground-truth tests
  (`gt_s`). NOT `PASS_RATE` (which mixes in the ~99%-hacked teacher pool).
- **±std is across seeds.** Blank = n=1 (no std). At n=4 the seed-to-seed std
  is ~0.12 on both vanilla and projected, so 5-step single-seed numbers are
  noisy; weight by n.
- **Never compare a multi-seed mean to a single-seed point.** Several arms
  (refresh-1/5/10, no_gate, reverse, mean-diff) only ran on seed 41. Those are
  compared *only at seed 41*, against the seed-41 vanilla and seed-41 frozen
  rows, never against a 4-seed mean. Mixing n is how the old refresh "ladder"
  produced a fake monotonic trend.
- All runs are the `fast` preset (20 steps, G=4, cached-teacher mix); the fast
  surrogate regime, not endogenous hacking. Incomplete runs are excluded (a run
  must log all `steps`).
- Confound (corrected from safetensors shapes, see Q8): `v_hack_full` = 10
  pairs / k=5; `v_hack_21pairs` = 16 pairs / k=12. Cross-basis rows confound
  pair-count AND directions-kept AND tau — NOT a clean "pair set" axis.

---

## Q1. Does the cached-teacher pool drive the student to hack? (feasibility, H4)

<!-- src: logs/*_goal0_fast_s4{1,2,3,4}.log, *_mix0_25_vanilla_s4{1,2,3}.log, *_vanilla_mix0125*.log -->

| arm     |   mix |  hack |  ±std | solve |  ±std |       seeds |
| :------ | ----: | ----: | ----: | ----: | ----: | ----------: |
| vanilla |   0.5 | 0.719 | 0.120 | 0.306 | 0.116 | 41,42,43,44 |
| vanilla |  0.25 | 0.678 | 0.082 | 0.200 | 0.076 |    41,42,43 |
| vanilla | 0.125 | 0.757 | 0.040 | 0.207 | 0.020 |     41 (×2) |

**Answer: yes.** Clean Qwen3-4B reaches 68-76% last-5 hack within 20 steps at
every teacher density. (Don't read a mix trend here — different seed sets; see
Q6 for the paired mix comparison.)

## Q2. 🥇 Does v_hack projection reduce hacking vs vanilla? (H1)

<!-- src: vanilla *_goal0_fast_s4{1-4}.log; *_g0_21pairs_frozen_s4{1-4}.log; *_g0_21pairs_refresh2_s4{1-4}.log -->

mix=0.5, v_hack_21pairs, one_sided, k=5, all n=4 (seeds 41-44):

| arm                 |  hack |  ±std | solve |  ±std |
| :------------------ | ----: | ----: | ----: | ----: |
| vanilla             | 0.719 | 0.120 | 0.306 | 0.116 |
| projected frozen-V  | 0.588 | 0.131 | 0.256 | 0.083 |
| projected refresh-2 | 0.537 | 0.066 | 0.225 | 0.050 |

**Answer: a consistent-in-sign reduction.** Frozen drops hack 0.719→0.588
(−13pp), refresh-2 →0.537 (−18pp); both cost ~5-8pp solve. Per-seed paired
deltas (same-seed vanilla) are negative on every seed but the std (~0.13-0.17)
is about the mean, so the magnitude is not pinned down at n=4. Short of the
preregistered 30pp. Note refresh-2 has the *tightest* hack std (0.066), i.e.
its effect is the most seed-stable.

## Q3. one_sided vs no_gate vs reverse gating? (gate_mode, seed 41 only)

<!-- src (all seed 41, v_hack_full): *_goal0_fast_s41.log (one_sided), *_goal1_nogate_s41.log, *_goal1_reverse_s41.log -->

no_gate and reverse only ran on seed 41, so this is a seed-41 within-group
comparison (no cross-seed mixing):

| gate      |  hack | solve |
| :-------- | ----: | ----: |
| vanilla   | 0.775 | 0.300 |
| one_sided | 0.775 | 0.275 |
| no_gate   | 0.625 | 0.200 |
| reverse   | 0.575 | 0.150 |

**Answer: more-aggressive gates cut more hack but cost more solve, and
one_sided on the 18-pair basis does ~nothing at seed 41** (0.775 = vanilla).
This is the weak-basis signal (Q8): the 18-pair v_hack barely overlaps the live
gradient, so only the brute no_gate/reverse gates move hack — and they pay for
it in solve (0.200, 0.150 vs 0.300). Single seed; directional only.

## Q4. SVD top-k vs rank-1 mean-diff? (basis, seed 41 only)

<!-- src (seed 41): *_goal0_fast_s41.log (SVD k=5, v_hack_full); *_meandiff_projected_s41.log (k=1) -->

| basis                 |  hack | solve |
| :-------------------- | ----: | ----: |
| vanilla               | 0.775 | 0.300 |
| SVD k=5 (v_hack_full) | 0.775 | 0.275 |
| mean-diff k=1         | 0.750 | 0.125 |

**Answer: at seed 41 neither 18-pair basis cuts hack, and mean-diff tanks
solve** (0.300→0.125). Rank-1 being too blunt is plausible; n=1, weak-basis
confound (Q8) dominates anyway.

## Q5. refresh-every cadence (seed 41 only — the honest comparison)

<!-- src (all seed 41, v_hack_21pairs): *_g0_21pairs_frozen_s41.log, *_g1_21pairs_refresh1_s41.log, *_g0_21pairs_refresh2_s41.log, *_g1_21pairs_refresh5_s41.log, *_g1_21pairs_refresh10_s41.log -->

refresh-1/5/10 only ran on seed 41, so the only valid comparison is at seed 41,
on the shared seed-41 vanilla baseline:

| refresh        |  hack | solve |
| :------------- | ----: | ----: |
| vanilla        | 0.775 | 0.300 |
| frozen (n=20+) | 0.475 | 0.200 |
| 10             | 0.575 | 0.200 |
| 5              | 0.550 | 0.225 |
| 2              | 0.450 | 0.200 |
| 1              | 0.600 | 0.200 |

**Answer: no monotonic refresh trend.** At seed 41, frozen (0.475) and
refresh-2 (0.450) are the best; refresh-1/5/10 are worse. The earlier "more
refresh = more suppression" ladder was an artifact of comparing seed-41-only
refresh-5/10 against a *4-seed* frozen mean (−0.131 paired). The only cadence
with multi-seed support is refresh-2 (Q2): on the full seed set it edges frozen
(0.537 vs 0.588 hack), but at seed 41 alone the two are within noise. Refresh
helps marginally at best; basis width (Q8) is the real lever.

## Q6. Teacher density (mix) — paired, does the gap hold as the pool thins?

<!-- src (v_hack_full, frozen, one_sided): vanilla *_mix*_vanilla*.log + *_goal0_fast*.log; projected *_mix0_25_projected_s4{1,2,3}.log, *_mix0.125_g8_projected_s41.log, *_goal{0,1}_fast_s4{1,3,4}.log -->

Paired Δ vs same-seed vanilla (v_hack_full, frozen, one_sided). Δ columns are
per-seed paired means; absolute hack/solve are group means (may differ slightly
from Δ since n differs):

|   mix | van hack | proj hack |  Δhack |  ±std | van solve | proj solve | Δsolve |    n | shared seeds |
| ----: | -------: | --------: | -----: | ----: | --------: | ---------: | -----: | ---: | -----------: |
|   0.5 |    0.719 |     0.700 | −0.062 | 0.075 |     0.306 |      0.283 | −0.081 |    4 | 41(×2),43,44 |
|  0.25 |    0.678 |     0.556 | −0.122 | 0.146 |     0.200 |      0.217 | +0.017 |    3 |     41,42,43 |
| 0.125 |    0.757 |     0.657 | −0.100 | 0.040 |     0.207 |      0.214 | +0.007 |    2 |       41(×2) |

**Answer: the reduction holds across densities (−6 to −12pp), and the solve
cost vanishes at low mix** — Δsolve goes from −8pp at mix=0.5 to slightly
*positive* (+0.7 to +1.7pp) at mix=0.25/0.125. mix=0.125 also has the tightest
std (0.040, n=2). This is why 0.125 is now the locked-in default: same hack
cut, no solve tax.

## Q8. Weak basis (`v_hack_full`) vs strong basis (`v_hack_21pairs`)

<!-- src (mix=0.5, frozen, one_sided): v_hack_full *_goal{0,1}_fast_s4{1,3,4}.log; v_hack_21pairs *_g0_21pairs_frozen_s4{1-4}.log -->

The basis NAMES are misleading. Reading the safetensors shapes/metadata (the
stored per-pair grads' first dim = pairs used; basis `top_k` from header):

| basis            | pairs used | k (top_k) | extract tau | what it is |
| :--------------- | ---------: | --------: | ----------: | :--------- |
| `v_hack_full`    |     **10** |     **5** |        0.25 | older ~12-pair set, k=5 |
| `v_hack_21pairs` |     **16** |    **12** |         0.0 | later ~18-pair set, k=12 |

Neither is 18 or 21 pairs (n_heldout=2 reserves 2). Both load with the same
train-time `drop_bottom_frac=0.25` noise floor. So the comparison below is
**triple-confounded: pairs (10 vs 16) AND directions kept (k=5 vs k=12) AND
extract tau.** We cannot attribute the gap to "pair set".

mix=0.5, frozen, one_sided:

| basis              |  hack |  ±std | solve |  ±std |    n |       seeds |
| :----------------- | ----: | ----: | ----: | ----: | ---: | ----------: |
| vanilla            | 0.719 | 0.120 | 0.306 | 0.116 |    4 | 41,42,43,44 |
| v_hack_full (weak) | 0.700 | 0.109 | 0.283 | 0.038 |    3 |    41,43,44 |
| v_hack_21pairs     | 0.588 | 0.131 | 0.256 | 0.083 |    4 | 41,42,43,44 |

At shared seed 41: weak basis = 0.775 (= vanilla, no effect), strong = 0.475.

**Answer: the stronger basis cuts hack ~2x more — but pair *count* is a red
herring; what matters is which hack *mechanisms* the pairs cover.** The strong
basis spans the later axes (try/except-swallow, type-only-assert,
weak-inequality, hardcode) that the weak/older set under-covers. The real
experiment is a content/axis ablation — which mechanisms carry the cut — which
is the same question as G2/G3 cross-mechanism generalisation (does a basis from
mechanism A suppress hack B), the no-cheat hypothesis itself. The k=5-vs-12
and 10-vs-16 differences are present but secondary.

Current `pairs.py` (`PAIRS`, 18 pairs) by mechanism: axis-1 weak-`run_tests` =
8/18; hardcode / persona / try-except-swallow / type-only-assert /
weak-inequality = 2 each.

---

## Q9. Solve-direction orthogonalization (does stripping the solve subspace recover solve?)

<!-- src (seed 41, mix=0.5, frozen, one_sided): #145 *_solveorth_base18_s41.log (v_hack_18base, no orth); #146 *_solveorth_m4_s41.log (v_hack_18solveorth4, m=4) -->

| basis                  |  hack | solve |
| :--------------------- | ----: | ----: |
| vanilla                | 0.775 | 0.300 |
| 18-pair base (no orth) | 0.500 | 0.200 |
| 18-pair solve-orth m=4 | 0.550 | 0.150 |

**Answer: no — at n=1 it did the opposite.** Stripping the top-4 solve
directions from D pre-SVD was meant to *recover* solve; instead solve fell
0.200→0.150 and hack rose 0.500→0.550. Both moves are ~0.05, inside the ~0.12
seed std — inconclusive, leaning negative. Caveats: (1) two nominally-18-pair
bases already disagree by 0.275 hack at this seed (`v_hack_full`=0.775 vs
`v_hack_18base`=0.500), so extraction variance likely dominates a 0.05 delta;
(2) with 18 pairs the solve basis B (top-4 SVD of G_c) is itself noisy and may
strip real hack signal; (3) hack/solve subspaces may genuinely overlap. Needs
≥3 seeds before any verdict.

---

## Q10. 🥇 Does the *pair set content* matter? (mechanism vs framing vs placebo)

<!-- src (seed 41, mix=0.125, frozen, one_sided, k=5; all bases extracted k=12/tau=0):
     #171 *_pairset_prog_wide_s41.log; #169 *_pairset_prog_wider_s41.log;
     #164 *_pairset_intent_vs_spec_s41.log; #162 *_pairset_honesty_text_s41.log;
     #160 *_pairset_moral_s41.log; #166 *_pairset_eval_aware_s41.log;
     #158 *_pairset_philosophical_s41.log; #168 *_pairset_null_city_s41.log (PLACEBO).
     baseline: 3x vanilla mix=0.125 seed-41 runs. -->

The detector we're allowed to have is *weak* (no-cheat invariant): it sees some
hacks and misses others. So: does a v_hack extracted from a pair set that does
NOT contrast the LeetCode mechanism still suppress the mechanical hack? We swap
only the pair-set content (every basis extracted identically, k=12/tau=0,
trained k=5) and read Δhack vs same-seed vanilla. n=1 projected per row;
±0.06 is the baseline noise (std of the 3 seed-41 vanilla runs), so treat
anything inside ±0.06 as null.

| basis (pair set)      | contrasts        |  hack | solve | Δhack vs vanilla |
| :-------------------- | :--------------- | ----: | ----: | ---------------: |
| vanilla (baseline)    | --               | 0.726 | ~0.20 |                — |
| `prog_wide`           | hack *mechanism* | 0.500 | 0.221 |           −0.226 |
| `prog_wider`          | mech + lang/cond | 0.679 | 0.236 |           −0.048 |
| `intent_vs_spec`      | semantic framing | 0.686 | 0.207 |           −0.040 |
| `honesty_text`        | semantic framing | 0.714 | 0.193 |           −0.012 |
| `moral`               | semantic framing | 0.721 | 0.221 |           −0.005 |
| `eval_aware`          | semantic framing | 0.736 | 0.186 |           +0.010 |
| `philosophical`       | semantic framing | 0.743 | 0.243 |           +0.017 |
| `null_city` (PLACEBO) | random content   | 0.750 | 0.221 |           +0.024 |

(Baseline = mean of the 3 seed-41 vanilla mix=0.125 runs the deltas are paired
against. The canonical `v_hack_21pairs` is NOT in this table: it was only run
at mix=0.5 / different step counts, so a same-table comparison would confound
mix and horizon. Its mix=0.5 effect is in Q2.)

**Answer: it's the *mechanism*, not the *framing*.** Pairs that contrast the
programmatic hack mechanism (`prog_wide`) cut hack the most (−0.226), at no
solve cost. Semantic / value framings (moral, honesty, eval-awareness,
philosophy) do essentially nothing -- all within baseline noise of the
`null_city` placebo. The placebo sits at +0.024 (no effect), exactly as it
should. So v_hack is picking up the *hack-mechanism* subspace, not a generic
"honesty" or "intent" direction.

Caveats (n=1, hold loosely): (1) all rows are single seed-41 runs; ±0.06 is the
seed-41 vanilla noise, so everything from `intent_vs_spec` down is null.
`prog_wide` needs ≥3 seeds (task #122) before the −0.226 is trustworthy.
(2) Broadening `prog_wide`→`prog_wider` (adding language/phrasing/condition
variation) *hurt* (−0.226→−0.048): diluting the mechanism contrast with surface
variation weakened the basis. (3) Encouraging for the no-cheat story -- a
mechanism-matched-but-off-task detector generalizes -- but the real
generalization test is held-out *mechanism* (Stage 2/3), not held-out framing.

## Q11. Does the projection gap survive to convergence? (60-step, seed 42)

<!-- src (seed 42, mix=0.125, 60 steps; n=1 each):
     #179 *_converge_vanilla_mix0125_s42.log; #178 *_converge_refresh2_mix0125_s42_logged.log;
     *_converge_proj21_mix0125_s42.log (frozen). NOTE: results.py has no `steps`
     dimension, so these merge into the 20-step mix=0.125 groups in the grouped
     table -- read them here from the raw last-5, not from that row. -->

The Q2 gap (−13 to −18pp) is measured at 20 steps, where vanilla hack has only
just plateaued (~step 13-16, see Dynamics note). Does projection keep hack down
once we run 3x past the plateau? One 60-step run per arm, seed 42, mix=0.125:

| arm                 | L5 hack | L5 solve |
| :------------------ | ------: | -------: |
| vanilla             |   0.936 |    0.293 |
| projected frozen-V  |   0.957 |    0.293 |
| projected refresh-2 |   0.907 |    0.307 |

**Answer: at n=1, the gap closes.** By step 60 all three arms sit at ~0.91-0.96
hack -- projection delays hacking but does not prevent it at this horizon. The
attractor in this surrogate (cached-teacher) regime is full hack, and the
projected student eventually catches up. refresh-2 is marginally below vanilla
(−2.9pp) and frozen marginally above (+2.1pp), both inside the ~0.06-0.12 seed
noise, so the honest read is "no surviving gap at 60 steps, seed 42."

Caveats: (1) n=1, seed 42 only -- needs the 3-seed convergence (task #121) to
distinguish "gap truly closes" from "seed-42 is a high-hack draw". (2) This is
mix=0.125 + seed 42 + 60 steps, three axes different from Q2's mix=0.5/20-step
numbers, so it is NOT a clean "same run, later" comparison. (3) The 20-step
suppression is real (Q2, n=4); what's unclear is whether longer training erodes
it or whether this is a sparse-teacher/seed artifact.

## Dynamics note (sizing the convergence test)

Per-step trajectories (mix=0.125 g8, seed 41): `hack_s` rises 0→~0.6-0.75 and
**plateaus by step ~13-16**; `gt_s` (solve) stays **noisy-flat at ~0.1-0.5 the
whole run, it never climbs**. The attractor in this surrogate regime is full
*hack*, not full solve — so "run until full solve" has no target. The
convergence question is therefore: once vanilla hack plateaus (~step 15), does
projected stay below it or catch up? A 60-step run (~2.2h at g8) sees 3x past
the plateau; a 1000-step run (~36h) is wasteful.

## Open / queued (no result yet)

- **convergence at ≥3 seeds (#121)**: the n=1 seed-42 run (Q11) shows the gap
  closing by step 60, but that could be a seed-42 high-hack draw. Need 2+ more
  seeds before concluding the suppression erodes vs survives.
- **pairset content at ≥3 seeds (#122)**: Q10's mechanism>framing>placebo
  ordering is n=1 per row; replicate `prog_wide` and the placebo on 2+ seeds.
- **route arm at scale (#182)**: running; validates routing's ablated-eval
  hack<kept on Qwen3-4B before the 3-way none/erase/route cells (#130).
- **k-slice (k=1/2/5)**: only smoke-tested, no 4B results.
- **Stage 2/3 cross-*mechanism* generalisation**: the load-bearing test --
  extract v_hack from hack A, check it stops the *unknown* hack B the student
  would otherwise learn. Q10 (held-out *framing*) is a weaker cousin.
