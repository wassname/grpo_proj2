# ── GRPO / Dr.GRPO loss on a mixed student+teacher pool ────────────────
# Source: train.py inner step (~1220-1334).
# Inner GRPO_step ported from lsdefine/simple_GRPO grpo_vllm_one.py:64-95.
# Unbiased normalization: Dr.GRPO, Liu et al. 2025, arXiv:2503.20783.

def grpo_step(model, prompt, teacher_pool, G_s, G_t, β, clip, unbiased):
    # ── Rollouts: G_s live student + G_t cached teacher (mixed pool) ───
    o_s = model.generate(prompt, n=G_s)                # on-policy student rollouts
    o_t = sample(teacher_pool[prompt], G_t)            # cached, FROZEN reward labels (reproducible)
    o   = o_s ⊕ o_t                                    # one group; both feed one group-relative adv
    is_student = [1]*G_s + [0]*G_t

    # ── Reward + advantage ─────────────────────────────────────────────
    R = [compute_reward(oi) for oi in o_s] + [cached_reward(oi) for oi in o_t]
    if max(R) - min(R) < ε: return  # zero-variance group ⇒ adv≡0 ⇒ pure waste (simple_GRPO bail)
    A = R - mean(R)                                    # Dr.GRPO unbiased: NO /σ_R
    #   classic GRPO would use (R-mean)/std(R); unbiased drops that + the 1/|o| length norm.

    # ── PPO-clip policy term (single inner step ⇒ ratio≡1) ─────────────
    logπ_old = per_token_logps(model(o), o).detach()   # frozen target
    logπ     = per_token_logps(model(o), o)            # with grad
    ρ = exp(logπ - logπ_old)                           # ≡1 here, but keep the clip form
    Lₚ = -min(ρ·A, clip(ρ, 1±clip)·A)                  # per token

    if β > 0:                                          # optional KL to free ref (δS=0 trick)
        logπ_ref = ref_logprobs(model, o)              # see 01_adapter
        Lₚ += β · (exp(logπ_ref-logπ) - (logπ_ref-logπ) - 1)   # K3 estimator

    # ── Reduce (unbiased: fixed denom, no per-response length norm) ────
    mask = (o.completion_tokens != pad)
    L = (Lₚ * mask).sum() / (G · max_new · prompts_per_step)   # unbiased
    #   classic: mean over tokens per response, then mean over responses.

    # ── Per-source split (diagnostic) ──────────────────────────────────
    # backward is linear and is_student+is_teacher=1 elementwise, so
    #   grad_s + grad_t == full-batch grad. Two backwards (~2× cost) buy us
    #   cin_s vs cin_t: does V light up MORE on teacher (hack) grads than student?
    # NB the zero+clone BETWEEN passes is load-bearing: after L_t.backward(),
    # δS.grad already holds g_s+g_t (grad accumulates), so skipping the reset
    # double-counts g_s. (GPT-5.4 review flagged the earlier pseudocode here.)
    zero_grad(); L_s = (Lₚ*mask*is_student).sum()/denom; L_s.backward(retain_graph=True); g_s = δS.grad.clone()
    zero_grad(); L_t = (Lₚ*mask*(1-is_student)).sum()/denom; L_t.backward();             g_t = δS.grad.clone()
    δS.grad ← g_s + g_t

# Why mixing teacher in: a weak/early student rarely hacks on its own, so the
# live GRPO gradient carries little hack signal to project. Cached teacher
# rollouts (graded once, frozen) inject reliable hack examples into the group.
# REVIEW (both families, deepest leak): for the TEACHER half this is NOT on-policy
# GRPO — it is advantage-weighted IMITATION of fixed completions. ratio≡1 holds for
# fresh student samples but is FORCED for teacher samples (logπ_old is recomputed on
# the current student, not the model that generated them). So "the live GRPO gradient
# V must overlap" may really be "push mass onto high-reward teacher completions", and
# erase may just block that imitation — nothing hack-specific. Decide what teacher_pool
# IS; for the clean causal test, isolate or remove it (see 07 priority).
