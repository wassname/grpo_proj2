"""Gradient projection: erase / route the hack component (pseudocode 03).

Once per optimizer step, after δS.grad is populated, before opt.step(). The bet:
subtract the hack-ward component of the live GRPO gradient (basis V from labeled
pairs) so the model descends what's left and cannot move hack-ward even when the
reward rewards it.

RELU-BEFORE-AGG everywhere a cosine appears: we remove relu(c)@V, so
cin = ‖relu(c)‖/‖g‖ (the hack-ward fraction stripped), NOT a signed sum where
anti-hack axes cancel. cout≈0 under one_sided is an arithmetic identity, not a
result -- only behavioural hack rate beaten against controls settles efficacy.
"""
from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor


def project_one(g: Float[Tensor, "r"], V: Float[Tensor, "k r"],
                gate_mode: str, preserve_magnitude: bool, overshoot: float):
    c = V @ g                                   # (k,) per-axis hack coef; c_i>0 => grad pushes hack-ward
    cin = torch.relu(c).norm() / g.norm().clamp_min(1e-12)
    if gate_mode == "one_sided":
        c_use = torch.relu(c)                   # default: only zap hack-ward motion
    elif gate_mode == "no_gate":
        c_use = c                               # sign-agnostic; drives ‖Vg‖->0
    elif gate_mode == "reverse":
        c_use = 2 * c                           # push away from hack: Vg' -> -Vg
    else:
        raise ValueError(gate_mode)
    removed = overshoot * (c_use @ V)           # ∈ span(V); overshoot=1 removes, 1.1 = 10% reversal
    g_proj = g - removed
    if preserve_magnitude:                      # keep update size, change only direction
        g_proj = g_proj * (g.norm() / g_proj.norm().clamp_min(1e-12))
    cout = torch.relu(V @ g_proj).norm() / g_proj.norm().clamp_min(1e-12)
    return g_proj, removed, cin, cout


def project_all(wrappers, V_hack, measure_only: bool, route: bool,
                gate_mode: str = "one_sided", overshoot: float = 1.0,
                preserve_magnitude: bool = False):
    """arms: none -> measure_only | erase -> write g_proj | route -> erase + park in δS_hack."""
    cins, couts, fired = [], [], 0
    for name, w in wrappers.items():
        if name not in V_hack:                  # below noise floor -> no hack signal -> skip
            continue
        g_proj, removed, cin, cout = project_one(
            w.δS.grad, V_hack[name], gate_mode, preserve_magnitude, overshoot)
        cins.append(cin); couts.append(cout)
        fired += int(cin > 1e-6)
        if measure_only:                        # vanilla: measure cin/cout, leave g
            continue
        w.δS.grad = g_proj                       # erase: descend the hack-free gradient
        if route:
            w.δS_hack.grad = removed             # park hack part in deletable knob (Cloud 2024)
    n = max(len(cins), 1)
    mean = lambda xs: torch.stack(xs).mean().item() if xs else 0.0
    return mean(cins), mean(couts), fired / n


def cos_overlap(grads: dict, V_hack) -> float:
    """‖relu(V@g)‖/‖g‖ aggregated over modules (cin_s / cin_t diagnostic, relu-before-agg)."""
    cs = []
    for name, g in grads.items():
        if name not in V_hack or g is None:
            continue
        cs.append(torch.relu(V_hack[name] @ g).norm() / g.norm().clamp_min(1e-12))
    return torch.stack(cs).mean().item() if cs else 0.0
