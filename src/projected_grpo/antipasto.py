"""SVD-basis adapter (pseudocode 01). Prior work: github.com/wassname/AntiPaSTO.

Train ONE per-module knob δS in the singular-value basis of each Linear. The
extracted hack direction V, the live gradient, and the projection all live in
these same low-rank weight-aligned coordinates (ℝ^r). At δS=δS_hack=0 the adapter
is bit-identical to the base model (W is never reconstructed on the main path),
so a no_grad forward with the knobs zeroed gives π_ref for free -- no 2nd model.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from safetensors.torch import load_file, save_file

TARGET = {"q_proj", "k_proj", "v_proj", "o_proj",
          "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",   # GatedDeltaNet
          "up_proj", "gate_proj", "down_proj"}
SVD_CACHE = Path("svd_cache")


@dataclass
class Wrap:
    layer: nn.Linear   # carries .U, .Vh buffers and .δS, .δS_hack params
    r: int

    @property
    def δS(self) -> nn.Parameter:
        return self.layer.δS

    @property
    def δS_hack(self) -> nn.Parameter:
        return self.layer.δS_hack


def svd_cached(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """U Σ Vh = W, reduced. Cache key = sha256(W) so a stale cache is impossible
    (different W -> different file). Native dtype in, fp32 SVD, cast back."""
    SVD_CACHE.mkdir(exist_ok=True)
    key = hashlib.sha256(W.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    path = SVD_CACHE / f"{key}.safetensors"
    if path.exists():
        t = load_file(path)
        return t["U"].to(W.dtype), t["S"].to(W.dtype), t["Vh"].to(W.dtype)
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    save_file({"U": U, "S": S, "Vh": Vh}, path)
    return U.to(W.dtype), S.to(W.dtype), Vh.to(W.dtype)


def _δ_hook(lin: nn.Linear, x_in, y):
    x = x_in[0]                              # (..., d_in)
    h = x @ lin.Vh.T                         # (..., r)   into singular basis
    h = h * (lin.δS + lin.δS_hack)           # scale per singular axis (forward uses the SUM)
    return y + h @ lin.U.T                   # (..., d_out)   δS+δS_hack=0 -> identity


def wrap(model: nn.Module) -> dict[str, Wrap]:
    wrappers: dict[str, Wrap] = {}
    for name, lin in model.named_modules():
        if not isinstance(lin, nn.Linear) or name.split(".")[-1] not in TARGET:
            continue
        U, _Σ, Vh = svd_cached(lin.weight)              # W ∈ ℝ^{d_out×d_in}
        r = min(lin.in_features, lin.out_features)
        lin.register_buffer("U", U)
        lin.register_buffer("Vh", Vh)
        lin.register_parameter("δS", nn.Parameter(torch.zeros(r, dtype=lin.weight.dtype)))
        lin.register_parameter("δS_hack", nn.Parameter(torch.zeros(r, dtype=lin.weight.dtype)))
        lin.register_forward_hook(_δ_hook)
        wrappers[name] = Wrap(lin, r)
    for p in model.parameters():
        p.requires_grad_(False)
    for w in wrappers.values():
        w.δS.requires_grad_(True)
        w.δS_hack.requires_grad_(True)
    logger.info(f"wrapped {len(wrappers)} Linears with δS knobs (r in "
                f"[{min(w.r for w in wrappers.values())}, {max(w.r for w in wrappers.values())}])")
    return wrappers


@contextmanager
def zeroed(wrappers: dict[str, Wrap]):
    """Both knobs -> 0 (free reference model). Restores on exit."""
    saved = [(w.δS.data.clone(), w.δS_hack.data.clone()) for w in wrappers.values()]
    for w in wrappers.values():
        w.δS.data.zero_()
        w.δS_hack.data.zero_()
    try:
        yield
    finally:
        for w, (s, sh) in zip(wrappers.values(), saved):
            w.δS.data.copy_(s)
            w.δS_hack.data.copy_(sh)


@contextmanager
def ablate_quarantine(wrappers: dict[str, Wrap]):
    """δS_hack -> 0 only (deploy-time ablation / extraction). No-op for erase (δS_hack stays 0)."""
    saved = [w.δS_hack.data.clone() for w in wrappers.values()]
    for w in wrappers.values():
        w.δS_hack.data.zero_()
    try:
        yield
    finally:
        for w, sh in zip(wrappers.values(), saved):
            w.δS_hack.data.copy_(sh)


def per_token_logps(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """logp of each next token. logits (b,s,V), ids (b,s) -> (b, s-1)."""
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    return logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def ref_logprobs(model: nn.Module, wrappers: dict[str, Wrap], ids: torch.Tensor) -> torch.Tensor:
    with zeroed(wrappers), torch.no_grad():
        return per_token_logps(model(ids).logits, ids)
