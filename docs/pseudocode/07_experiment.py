# ── The experiment ─────────────────────────────────────────────────────
# Sources: docs/spec.md (preregistered), README.md.
# Hypothesis (H1): SVD-basis gradient projection against a v_hack extracted from
# ~60-80 contrastive pairs reduces reward-hack rate by ≥30pp absolute vs vanilla
# GRPO, at MATCHED LeetCode pass rate (±10pp).
# STATUS (blog 20260529, read this): the vanilla-vs-projected head-to-head EXISTS —
# Table 1, n=2 matched seeds (41,43), 21 pairs (a PREFIX of the 60-80 H1 spec):
#   erase frozen-V  −12..−18pp hack_student;  refresh-2  −16..−23pp;  pass ∓0..12pp.
# So the headline IS 12-23pp at 21 pairs, NOT 30pp. The 30pp figure was a CONFOUNDED
# within-arm step (12→21 pairs moved pairs+k+τ together). What's missing is n=3
# (queued), the NEGATIVE CONTROLS below, and the 60-80-pair spec design — not the
# comparison itself.

# ── Substrate: 4 independent loophole modes, partitioned non-overlapping ──
# so a vanilla student can learn ALL four independently (the emergence reference).
MODES = ["run_tests", "sentinel", "stdout_marker", "file_marker"]
model, teacher_pool = FIXED                             # identical across arms; ONLY gradient policy differs

# ── Arms (06_train_loop with one knob changed) ─────────────────────────
ARMS = {
  "vanilla":  intervention="none",                      # no projection; emergence baseline
  "erase":    intervention="erase",  V from all modes,  # subtract hack component from δS.grad
  "route":    intervention="route",  V from all modes,  # quarantine in δS_hack, ablate at deploy
  "route_weak": ...,                                    # generalization test, below
}
# ── NEGATIVE CONTROLS — the missing first-class arms (BOTH families, top fix) ──
# Projecting out ANY rank-k subspace and renormalizing perturbs the update, so a
# hack drop may be generic regularization, not selective removal. erase must beat:
CONTROLS = {
  "random_V":   "erase with a random orthonormal V (matched k, matched ‖·‖)",
  "shuffled_V": "erase with V from label-SHUFFLED pairs (hack/clean swapped)",
  "nonhack_V":  "erase with V from clean-vs-clean (non-hack-contrastive) pairs",
}
# If real V does not clearly beat all three, the effect is NOT hack-specific → STOP.

def run_arm(arm):
    for seed in [41, 42, 43]:                           # n≥3 for the headline claim
        train(model, teacher_pool, arm, seed)
    return hack_rate_last5, gt_pass_rate_last5          # paired: hack must drop at MATCHED pass

# ── The no-cheat test: weak-detector generalization (leave-one-mode-out) ──
# At deployment there are KNOWN hacks and UNKNOWN hacks. We are ALLOWED a weak
# detector that flags some modes and misses others. Extract V from the known set,
# route on it, and ask: is the held-out mode ALSO suppressed? A detector that sees
# every hack proves nothing. Both reviewers: a single 2-vs-2 split at n=3 is weak
# and confounded → prefer LEAVE-ONE-MODE-OUT over all 4 modes (4 folds), w/ controls.
def weak_detector_leave_one_out():
    for held in MODES:                                 # held = the "unknown" hack
        known = [m for m in MODES if m != held]        # weak detector flags only these
        V = extract_v_hack(pairs_from_pool(modes=known))
        # CONFOUND (both): teacher pool MUST be `known`-only, else we TEACH `held`
        # while measuring its suppression (back-door leakage).
        train(intervention="route", V_hack=V, train_modes=MODES, teacher_pool=pool(known))
        # PASS, PER-MODE (not global avg): hack on `held` ↓ while its pass stays flat.
        # A global ±10pp pass band hides a per-mode pass collapse = capability damage.
        assert hack_rate(held) ↓  and  pass_rate(held) ≈ pass_rate_vanilla(held)
        # ...AND real V beats random_V on `held` (else it's generic conservatism).
        assert suppression(real_V, held) > suppression(random_V, held)

# CONFOUNDS that make "held-out ↓" weak evidence (both families):
#   capability damage · shared surface features (all 4 modes = "fake a success
#   signal", not independent) · selection contamination (known-detector false-
#   positives on held) · teacher contamination / cross-mode coupling under joint train.

# ── Health checks (gate validity; NOT evidence of efficacy) ────────────
# All cin/cout use ‖relu(V@g)‖/‖g‖ (relu-before-agg, see 03): the hack-ward
# fraction, matching what the one_sided gate removes. A signed aggregate would let
# anti-hack axes cancel and mis-state both cin and cout.
#   cin_s (NOT just cin_t>cin_s) : the live STUDENT-gradient overlap. If cin_s≈0
#       throughout, V misses the live gradient → method is a NO-OP whatever cin_t
#       does. Want cin_s>0 AND falling as hack rate falls. (deepseek top fix)
#   control cin on teacher-CORRECT vs teacher-HACK rollouts: equal ⇒ V tracks the
#       teacher DISTRIBUTION (length/reward/syntax), not hack-ness. (both)
#   cout ≈ 0           : with relu, a REAL residual-hack-ward-leak check under
#       one_sided (still a tautology under no_gate). cf. cout on the UPDATE Δδ (not g)
#       to catch the Adam-preconditioner leak — see 03 / failure mode 4.
#   cin_t decay/refresh: tracks a labeled-pair artifact; mild evidence at best.
#   δS=0 ⇒ bit-identical: free-ref-model + measure_only arm are honest.

# ── What would FALSIFY / where it could be fooling us ──────────────────
# 1. erase ≈ vanilla at matched pass → V doesn't overlap the live GRPO subspace
#    (central unproven hope); labeled-pair gradient ≠ unlabeled GRPO gradient.
# 2. erase ≈ random_V/shuffled_V/nonhack_V → generic regularization, not hack
#    removal. THE control that rules out the boring story.
# 3. TEACHER-POOL IMITATION (both, deepest leak): erase only "works" by blocking
#    advantage-weighted imitation of cached teacher completions. Test: does erase
#    still suppress with NO teacher pool (pure student GRPO)?
# 4. AdamW MOMENTUM BYPASS: we project g but the optimizer steps on moment m, which
#    banks hack-ward mass. cout≈0 while the update is still hack-ward → silent no-op.
# 5. held-out ↓ only via capability damage → enforce matched pass PER held-out mode.
# 6. confound: a "more pairs" win that is really k or τ_axis → one-knob sweeps.
# 7. OUTCOME-FLAG DEFLATION: exploited = passed ∧ ¬gt_correct falls when passed
#    falls; report P(exploit | attempt) via the code-structure detector (E flag).
# 8. rotation: early hack drop is temporary; model relearns hacks orthogonal to V.

# ── PRIORITY (reviewers + blog "what I'd want to see") ─────────────────
# The head-to-head exists (Table 1, n=2). The cheapest things that would change
# belief, in order:
# 1. n=3 matched seeds (blog #1; pueue queued). At n=2, SEM is not evaluable — the
#    spec's SEM falsification clause can't even be checked yet.
# 2. random-V control, ONE extra run paired with the existing erase run. A lone
#    erase-beats-vanilla win can't rule out projection-as-regularization; random-V
#    can. Cheapest insurance. (shuffled-V / nonhack-V later.)
# 3. cos null baseline (blog #3): two random unit vecs in r-dim with k kept axes
#    have E[cin] ≈ sqrt(k/r). "+0.28" only means something above THAT. Compute it.
# 4. Direct overlap audit: ‖relu(V@g)‖/‖g‖ energy on the live STUDENT grad; correlate
#    with per-batch exploited fraction; student vs teacher terms separately.
# 5. DROP route until erase is proven (route = erase + quarantine + deploy-ablation
#    shock). Then weak-detector leave-one-out.
# NB the headline is against ONE dominant mechanism: the teacher pool is degenerate,
# ~96% of rollouts fire just two correlated signatures (E=loophole_used, C=arbitrary_
# pass). Cross-mechanism (G2/G3) is the load-bearing generalization test, not these.
