# ── Gradient projection: erase / route the hack component ──────────────
# Source: proj.py.  Once per optimizer step, after δS.grad is populated,
# before opt.step().  The bet: subtract the hack-ward component of the live
# GRPO gradient (basis V from labeled pairs); the model descends what's left,
# so it can't move hack-ward even when the reward rewards it.

def project_one(g, V, gate_mode, preserve_magnitude, overshoot):
    c = V @ g                                       # g ∈ ℝ^r;  V ∈ ℝ^{k×r};  c per-axis hack coef
    cos_pre = ‖relu(c)‖ / ‖g‖                       # hack-ward fraction removed ∈ [0,1]
    if   gate_mode == "one_sided": c_use = relu(c)  # default: only zap hack-ward motion
    elif gate_mode == "no_gate":   c_use = c        # sign-agnostic; drives ‖Vg‖→0
    elif gate_mode == "reverse":   c_use = 2*c      # push away from hack: Vg' → -Vg
    removed = overshoot * (c_use @ V)               # ∈ span(V); overshoot=1 removes, 1.1 = 10% reversal
    g_proj  = g - removed
    if preserve_magnitude: g_proj *= ‖g‖ / ‖g_proj‖ # keep update size, change only direction
    cos_post = ‖relu(V @ g_proj)‖ / ‖g_proj‖        # residual hack-ward leak
    return g_proj, removed, cos_pre, cos_post

def project_all(wrappers, V_hack, measure_only, route):
    for name, w in wrappers.items():
        if name not in V_hack: continue             # below noise floor → no hack signal → skip
        g_proj, removed, cin, cout = project_one(w.δS.grad, V_hack[name], ...)
        if measure_only: continue                   # vanilla: measure cin/cout, leave g
        w.δS.grad ← g_proj                          # erase: descend the hack-free gradient
        if route: w.δS_hack.grad ← removed          # route: park hack part in deletable knob (Cloud 2024)
    return mean(cin), mean(cout), frac_fired

# arms:  none → measure_only  |  erase → write g_proj  |  route → erase + park in δS_hack (zero at eval)

# Gotchas
# - relu BEFORE aggregating, everywhere cosine appears (cin_s/cin_t in 06, health checks in
#   07): we remove relu(c)@V, so cin = ‖relu(c)‖/‖g‖. A signed c.sum() lets anti-hack axes
#   cancel hack-ward ones and under-reports the stripped mass.
# - cout≈0 under one_sided is an arithmetic identity, not a result. Only behavioural hack
#   rate at matched pass, beaten against negative controls, settles whether it worked.
# - AdamW steps on accumulated moment m, which banks hack-ward mass from pre-V / non-split
#   steps, so per-step g-projection can be a silent no-op (cout≈0 while the update still moves
#   hack-ward). Project the update, or purge Adam state in span(V). weight_decay on δS_hack
#   pulls it →0, leaking quarantined capability into δS.
# - preserve_magnitude is erase-then-boost: on a hack-dominated step it renormalises the noise
#   residue back to full ‖g‖. Ablate it; a smaller norm on hack-ward steps may be desired.
# - route ≠ erase across training: the route forward uses δS+δS_hack → different rollouts →
#   different δS trajectory, and ablating δS_hack at deploy can shock.
# - smoke (mechanics, not efficacy): removed ∈ span(V); ‖δS_hack‖>0 iff route.
