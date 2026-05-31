# ── Extract the hack direction V ───────────────────────────────────────
# Source: extract_vhack_grad.py, pairs.py.
#
# Key identity (README "How it works"): the GRPO single-step update on a pair
# with advantages (adv_hack=+1, adv_clean=-1) equals, algebraically,
#     -adv_h·∇logp(hack) - adv_c·∇logp(clean)  =  ∇NLL(hack) - ∇NLL(clean)
# So we compute it via the simpler NLL path. The *meaning* is still "the GRPO
# gradient you'd get from a perfectly-labeled pair." We then SVD the per-module
# stack of these paired diffs; the top right singular vectors are the hack basis.
# Top-k (not just mean-diff) because hack signal is multi-modal across flavors
# (CHaRS, Abdullaev 2025): one global direction is brittle.

def completion_nll(model, prompt, completion):           # mean NLL on completion tokens only
    logits = model(tok(prompt+completion)).logits[:-1]
    return -log_softmax(logits)[completion_tokens].mean()
# REVIEW (both families): the GRPO=NLL-diff identity holds only at adv=±1, no
# ratio-clip, no length-norm. But this extraction is length-normalized (.mean()
# per completion) while the live Dr.GRPO loss (05) uses a FIXED denom
# G·max_new·pp (no length norm). So V is extracted under a different metric than
# it is applied in → biased toward SHORT-completion hacks. Match the denominators
# or flag it. (Also: the live advantage is group-relative & mixed, not a clean ±1.)

def extract_v_hack(model, wrappers, pairs, k, τ_axis, n_heldout):
    train = pairs[:-n_heldout]                            # last n_heldout reserved for validation
    G_hack, G_clean = defaultdict(list), defaultdict(list)

    # ── 1. Sample the labeled-pair GRPO gradient on δS, per module ──────
    for pair in train:
        for label, completion in [("hack", pair.hack), ("clean", pair.clean)]:
            model.zero_grad()
            ℒ = completion_nll(model, pair.prompt, completion)
            assert isfinite(ℒ)                            # fail fast; nan ⇒ broken pair, not skip
            ℒ.backward()
            for name, w in wrappers.items():
                bucket[label][name].append(w.δS.grad.clone())   # g ∈ ℝ^r per module

    # ── 2. Per-module SVD of paired diffs D = g_hack - g_clean ─────────
    V_hack, V_sv = {}, {}
    for name in wrappers:
        D = stack(G_hack[name]) - stack(G_clean[name])    # D ∈ ℝ^{n_pairs×r}; pairing cancels prompt noise
        U_d, S_d, Vh_d = svd(D)
        V = Vh_d[:k]                                       # V ∈ ℝ^{k×r}, rows orthonormal in ℝ^r

        # ── 3. Orient hack-ward by per-pair majority vote ──────────────
        # proj.py gates one-sided on ⟨g, v_i⟩>0, so +v_i MUST point hack-ward.
        # Majority vote (not sign(mean)) is outlier-robust (repeng convention).
        votes = sign( count_pos(D @ V.T) - n_pairs/2 )    # [k]
        V = V * votes[:, None]

        if τ_axis > 0:                                     # zero noisy axes: S_i/S_0 < τ_axis
            V = V * (S_d[:k]/S_d[0] >= τ_axis)[:, None]
        V_hack[name], V_sv[name] = V, S_d[:k]             # save S to enable load-time noise floor

    return V_hack, V_sv

# ── 4. Load-time post-process (README "Noise floor at load") ───────────
def postprocess_v_hack(V_hack, V_sv, k_use, drop_bottom_frac):
    V_hack = {n: V[:k_use] for n,V in V_hack.items()}     # slice top-k_use (k-ablation knob)
    # Global noise floor: pool every S_i across ALL modules, drop bottom frac by
    # quantile. A noisy module shouldn't be protected by its own "top direction."
    thresh = quantile(concat(V_sv.values()), drop_bottom_frac)
    for n in V_hack:
        keep = V_sv[n][:k_use] >= thresh
        V_hack[n] = V_hack[n][keep]                        # rows below floor dropped
    return {n: V for n,V in V_hack.items() if len(V) > 0}  # all-below modules filtered out

# Pairs: ~10..21 hand-crafted (prompt, hack_completion, clean_completion) triples
# in pairs.py, OR pool-derived from graded teacher rollouts (pairs_from_pool.py)
# — the latter lets us build V from a half-A-only hack set (the weak detector,
# see 07_experiment). docs: how_to_write_personas.md, how_to_rewrite_pairs.md.
