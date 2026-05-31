# Related work — gradient routing, projection, and confessions vs our method

Compiled 2026-05-31 from full-text reads. Local copies in this dir. The question
for every paper: **does it cheat / assume a hack oracle? how does it handle unknown
hacks at train and deploy?** Our no-cheat invariant: training never gated on
`gt_pass` or the full detector suite over student rollouts; the mask is a weak,
self-supervised persona direction; the quarantine is deleted at deploy.

## No-cheat scorecard

| paper | train-time signal | hack oracle? | deploy | unknown hacks |
|---|---|---|---|---|
| **TDGA** (Deng, [2605.25189](https://arxiv.org/abs/2605.25189)) | clean warmup only (SVD of clean SFT param-deltas) | **no** — references only *clean* data | none; projects onto clean subspace throughout | rejected by construction (any hack is off-clean), but only *delays* (rank-5/10 hack-free ~200 steps then drift); fixed subspace |
| **Gradient Routing** (Cloud, [2410.04332](https://arxiv.org/abs/2410.04332), Oct 2024) | user data-label mask (activation detach) | needs labels, works with **partial/ad-hoc** labels | ablate routed subregion (Expand-Route-Ablate) | **absorption** — unlabeled forget data localises to the routed region anyway (`:352,367`); "mechanistic supervision avoids Goodharting" (`:422`); MNIST split *needs* L1 (`:857,1115`) |
| **SGTM** (Shilov/Cloud, [2512.05648](https://arxiv.org/abs/2512.05648), Dec 2025) | noisy classifier label (parameter-gradient zero-mask on dedicated neurons/heads) | tolerates **label noise** explicitly | zero the dedicated params | self-reinforcing localisation; **leakage** measured (§4.3): 0.005-0.02 at 64M w/ 40% undiscovered, *shrinks with scale* |
| **Confessions** (OpenAI, [2512.08093](https://arxiv.org/abs/2512.08093)) | weak **LLM judge** on confession honesty (separate rewarded output) | **no perfect oracle** — judge may be as hackable as the main RM (`:59`) | inference-time monitor/reject | generalises OOD (admits untrained misbehaviour) **but does not reduce hacking** (`:265`) |

Load-bearing consequence: **none need a hack oracle.** Cloud's absorption + SGTM's
label-noise robustness *are* the "weak detector for A generalises to suppress
unlabeled B" result. The routing+absorption machinery is theirs; what is ours is
the *signal source* and *setting* (below).

## How each routes gradients (one-liner each)

- **Cloud (earlier):** `x = m*a + (1-m)*a.detach()` — activation-level stop-gradient
  mask; forward value unchanged, backward routed. L1 on the encoding to force the
  split on unnatural problems.
- **SGTM (later, the one to copy):** for a labeled forget example, zero the gradient
  to all params except a fixed dedicated subset of MLP neurons + attn heads.
  Forward keeps the region for everyone; only the backward is masked. No SVD.
- **TDGA:** `G_clean = U_clean diag(alpha) U_clean^T G`, step on the clean component
  only. Project *onto* clean, not *off* hack.
- **Confessions:** no masking. Extra output `y_c`; reward it with a judge; reinforce
  only confession tokens; "seal of confession" decouples it from task reward.

## Closest twins / comparison points (not yet quote-verified beyond the abstract)

From `search_for_more.md`:
- **GRIFT** ([2604.16242](https://arxiv.org/abs/2604.16242)) — gradient fingerprints
  to detect+reject reward hacking. Same level (gradient), same RLVR-loophole setting;
  their detector is a comparison point for our weak persona mask.
- **Rebound / Advantage Modification** ([2604.01476](https://arxiv.org/abs/2604.01476))
  — intervenes in the GRPO *advantage*; we intervene in the *gradient*. The exact contrast.
- **Spilling the Beans** ([2511.06626](https://arxiv.org/abs/2511.06626)) — SFT
  self-report generalises OOD (train on known errors -> admit hidden objectives).
  The generalisation story structurally matches ours.
- **Baker et al. monitoring** ([2503.11926](https://arxiv.org/abs/2503.11926)) —
  weak monitor catches strong model, BUT optimising against it breeds *obfuscated*
  reward hacking. This is the risk that kills reward-driven routing and argues for
  mask-only (mechanistic) routing.

## What is ours (the testable contribution)

Not the machinery (Cloud/SGTM own route+ablate+absorption), not project-onto-clean
(Deng), not reward-for-honesty (OpenAI, and we reject it — it invites Baker's
obfuscation and reintroduces a live judge over student rollouts).

Plausibly novel:
1. **Mask source = a self-supervised persona-contrast direction** (grad or act, in
   the SVD-of-W basis), ~10 pairs, no content classifier, no outcome label, no
   oracle. Cloud/SGTM route by document/domain labels.
2. **Setting = RL reward hacking** (they do pretrain/SFT content-domain unlearning).
3. **Scalability argument** — the mask is the model's own representation, so it
   should sharpen with capability; SGTM's own data (leakage shrinks with scale)
   supports rather than threatens this. Weak-to-strong alignment framing.

Caveat: SGTM's authors are Anthropic alignment, so RL reward hacking is plausibly on
their roadmap — lead with the persona-direction-as-mask as the distinctive claim,
and report honestly that the routing/absorption mechanism is inherited.

## Design implications carried into `docs/spec/20260531_routing_v2_distinct_basis.md`

- Impose the mask, never reward it (Baker; Cloud `:422`).
- Distinct basis + additive forward + detach-route (not hard MoE) so unflagged
  hacks pass through the quarantine and can be absorbed.
- Seed hard (flagged), absorb soft (unflagged) — SGTM's hybrid.
- Expect to add an L1 sparsity aid (Cloud shows it can be necessary).
- Leakage is bounded and shrinks with scale (SGTM §4.3) — the central reason the
  additive design is viable, and the central risk at our small scale.
