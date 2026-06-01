"""Parametrized-LoRA adapter (pseudocode 01).

Per target Linear, freeze a random projection A ∈ ℝ^{r×d_in} and train B ∈ ℝ^{d_out×r}
(plus a same-shape quarantine B_hack for the route arm). The added weight is

    δW = (B + B_hack) · A          # applied as  y += (A x) @ (B+B_hack)ᵀ

which is LINEAR in the trained knob B: ∂δW/∂B is the constant A. That is the one
property the gradient projection needs -- a hack direction V extracted once stays a
fixed weight-space direction, so we can project it out every step without re-deriving
it (cf. proj.py). At B=B_hack=0 the adapter is identity (W is never touched), so a
no_grad forward with the knobs zeroed is π_ref for free -- no 2nd model. B_hack has the
SAME shape as B, so the quarantine is capacity-matched (it cannot become a sink).

A is frozen and random (not the SVD basis): showing the projection works in an
arbitrary fixed basis is stronger than an SVD-aligned one, and it drops the whole
per-module SVD subsystem.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
from loguru import logger

TARGET = {"q_proj", "k_proj", "v_proj", "o_proj",
          "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",   # GatedDeltaNet
          "up_proj", "gate_proj", "down_proj"}


@dataclass
class Wrap:
    layer: nn.Linear   # carries frozen .A buffer and trained .B, .B_hack params
    r: int

    @property
    def B(self) -> nn.Parameter:
        return self.layer.B

    @property
    def B_hack(self) -> nn.Parameter:
        return self.layer.B_hack


def _lora_hook(lin: nn.Linear, x_in, y):
    x = x_in[0]                                  # (..., d_in)
    h = x @ lin.A.T                              # (..., r)   into the fixed random subspace
    return y + h @ (lin.B + lin.B_hack).T        # (..., d_out)   B=B_hack=0 -> identity


def wrap(model: nn.Module, r: int = 32) -> dict[str, Wrap]:
    wrappers: dict[str, Wrap] = {}
    for name, lin in model.named_modules():
        if not isinstance(lin, nn.Linear) or name.split(".")[-1] not in TARGET:
            continue
        d_out, d_in = lin.out_features, lin.in_features
        rk = min(r, d_in)                        # need rk ≤ d_in for orthonormal rows of A
        A = torch.empty(rk, d_in, dtype=torch.float32, device=lin.weight.device)
        nn.init.orthogonal_(A)                   # A Aᵀ = I_rk; QR (geqrf) needs fp32, cast after
        lin.register_buffer("A", A.to(lin.weight.dtype))  # frozen bf16 buffer for the forward matmul
        zeros = lambda: nn.Parameter(torch.zeros(d_out, rk, dtype=lin.weight.dtype, device=lin.weight.device))
        lin.register_parameter("B", zeros())     # zero-init -> δW=0 at start (π_ref) and ∇B≠0 (A≠0)
        lin.register_parameter("B_hack", zeros())
        lin.register_forward_hook(_lora_hook)
        wrappers[name] = Wrap(lin, rk)
    for p in model.parameters():
        p.requires_grad_(False)                  # freeze base; A is a buffer so already frozen
    for w in wrappers.values():
        w.B.requires_grad_(True)
        w.B_hack.requires_grad_(True)
    logger.info(f"wrapped {len(wrappers)} Linears with LoRA knobs (r in "
                f"[{min(w.r for w in wrappers.values())}, {max(w.r for w in wrappers.values())}])")
    return wrappers


@contextmanager
def zeroed(wrappers: dict[str, Wrap]):
    """Both knobs -> 0 (free reference model). Restores on exit."""
    saved = [(w.B.data.clone(), w.B_hack.data.clone()) for w in wrappers.values()]
    for w in wrappers.values():
        w.B.data.zero_()
        w.B_hack.data.zero_()
    try:
        yield
    finally:
        for w, (b, bh) in zip(wrappers.values(), saved):
            w.B.data.copy_(b)
            w.B_hack.data.copy_(bh)


@contextmanager
def ablate_quarantine(wrappers: dict[str, Wrap]):
    """B_hack -> 0 only (deploy-time ablation / extraction). No-op for erase (B_hack stays 0)."""
    saved = [w.B_hack.data.clone() for w in wrappers.values()]
    for w in wrappers.values():
        w.B_hack.data.zero_()
    try:
        yield
    finally:
        for w, bh in zip(wrappers.values(), saved):
            w.B_hack.data.copy_(bh)


def per_token_logps(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """logp of each next token. logits (b,s,V), ids (b,s) -> (b, s-1)."""
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    return logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def ref_logprobs(model: nn.Module, wrappers: dict[str, Wrap], ids: torch.Tensor) -> torch.Tensor:
    with zeroed(wrappers), torch.no_grad():
        return per_token_logps(model(ids).logits, ids)
