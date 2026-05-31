# ── AntiPaSTO adapter ──────────────────────────────────────────────────
# Train ONE per-module knob δS in the singular-value basis of each Linear.
# Source: antipasto.py.  Prior work: github.com/wassname/AntiPaSTO
#
# Why SVD basis: the extracted hack direction, the live gradient, and the
# projection all need to live in the same low-rank, weight-aligned coordinates.
# At δS=0 the adapter is bit-identical to the base model (no SVD round-trip on
# the main path -- W is never reconstructed), which gives us a free reference
# model: a no_grad forward with δS zeroed yields π_ref logprobs, no 2nd model.

TARGET = {"q_proj","k_proj","v_proj","o_proj",      # attention
          "in_proj_qkv","in_proj_z","in_proj_a","in_proj_b","out_proj",  # GatedDeltaNet
          "up_proj","gate_proj","down_proj"}        # MLP

def wrap(model):
    for name, lin in model.named_linears():          # lin.weight = W ∈ ℝ^{d_out×d_in}
        if name.split(".")[-1] not in TARGET: continue
        U, Σ, Vh = svd_cached(lin.W)                  # cache key = sha256(W) → fail-loud stale
        r = min(d_in, d_out)
        lin.U, lin.Vh ← freeze(U), freeze(Vh)         # buffers, native dtype; double as v_hack basis
        lin.δS      = Param(zeros(r))                  # main trainable knob   ∈ ℝ^r
        lin.δS_hack = Param(zeros(r))                  # routing quarantine    ∈ ℝ^r (Cloud 2024)
        lin.register_forward_hook(δ_hook)
    freeze(every param except δS, δS_hack)            # only 2 knobs per module train
    return wrappers   # dict[name → {layer, δS, δS_hack, r}]

# ── Forward hook: y_new = y + U diag(δS+δS_hack) Vh · x ────────────────
def δ_hook(lin, x, y):
    h = (Vh @ x)                                       # h ∈ ℝ^{...×r}   into singular basis
    h = h * (lin.δS + lin.δS_hack)                     # scale per singular axis
    return y + (U @ h)                                # back to ℝ^{...×d_out}
    # δS+δS_hack=0 at init ⇒ δ=0 ⇒ identical to base. The forward uses the SUM,
    # so a hack-ward update parked in δS_hack still moves the *training* model;
    # zeroing δS_hack at eval ablates that routed capability (see 03_project).

# ── Free reference model (KL term) ─────────────────────────────────────
def ref_logprobs(model, ids):
    with zeroed(δS, δS_hack), no_grad():              # W' = W + U·0·Vh = W exactly
        return per_token_logps(model(ids).logits, ids)
