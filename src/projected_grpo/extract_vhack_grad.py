"""Extract the hack direction V from labeled pairs (pseudocode 02).

Key identity: the GRPO single-step update on a pair with advantages (+1 hack,
-1 clean) equals ∇NLL(hack) - ∇NLL(clean). So we compute it via the simpler NLL
path; the meaning is still "the GRPO gradient a perfectly-labeled pair emits". We
SVD the per-module stack of these paired diffs; the top right singular vectors
are the hack basis. Top-k (not mean-diff) because hack signal is multi-modal.

REVIEW caveat: this extraction is length-normalized (.mean() per completion)
while the live Dr.GRPO loss uses a fixed denom (no length norm) -> V is biased
toward short-completion hacks. Flagged, not fixed (matches pseudocode).

CLI: python -m projected_grpo.extract_vhack_grad --model=... --out-path=...
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import tyro
from loguru import logger
from safetensors.torch import load_file, save_file

from projected_grpo.antipasto import Wrap, per_token_logps, wrap
from projected_grpo.problems import Pair, default_pairs


def completion_nll(model, tok, prompt: str, completion: str) -> torch.Tensor:
    """Mean NLL on completion tokens only."""
    dev = next(model.parameters()).device           # full feeds the cuda model; ids must follow
    p_ids = tok(prompt, return_tensors="pt").input_ids
    full = tok(prompt + completion, return_tensors="pt").input_ids.to(dev)
    n_p = p_ids.shape[1]
    logp = per_token_logps(model(full).logits, full)        # (1, L-1)
    comp = logp[:, n_p - 1:]                                  # logp of completion tokens
    return -comp.mean()


def extract_v_hack(model, tok, wrappers: dict[str, Wrap], pairs: list[Pair],
                   k: int, τ_axis: float, n_heldout: int):
    train = pairs[:len(pairs) - n_heldout] if n_heldout else pairs
    G = {"hack": {n: [] for n in wrappers}, "clean": {n: [] for n in wrappers}}
    for pair in train:
        for label, completion in [("hack", pair.hack), ("clean", pair.clean)]:
            model.zero_grad()
            ℒ = completion_nll(model, tok, pair.prompt, completion)
            assert torch.isfinite(ℒ), f"non-finite NLL on {label} pair"   # fail fast, never skip
            ℒ.backward()
            for n, w in wrappers.items():
                G[label][n].append(w.δS.grad.detach().clone())

    V_hack, V_sv = {}, {}
    for n in wrappers:
        D = torch.stack(G["hack"][n]) - torch.stack(G["clean"][n])         # (n_pairs, r); pairing cancels prompt noise
        U_d, S_d, Vh_d = torch.linalg.svd(D.float(), full_matrices=False)
        V = Vh_d[:k]                                                       # (k, r), rows orthonormal in ℝ^r
        # orient hack-ward by per-pair majority vote (outlier-robust; proj gates on ⟨g,v_i⟩>0)
        n_pairs = D.shape[0]
        votes = torch.sign((D.float() @ V.T > 0).sum(0).float() - n_pairs / 2)  # (k,); D bf16, V fp32 from svd
        votes[votes == 0] = 1.0
        V = V * votes[:, None]
        if τ_axis > 0:                                                     # zero noisy axes: S_i/S_0 < τ_axis
            V = V * (S_d[:k] / S_d[0] >= τ_axis)[:, None].float()
        V_hack[n], V_sv[n] = V, S_d[:k]
    logger.info(f"extracted V from {len(train)} pairs over {len(wrappers)} modules, k={k}")
    return V_hack, V_sv


def postprocess_v_hack(V_hack, V_sv, k_use: int, drop_bottom_frac: float):
    """Slice top-k_use, then a GLOBAL noise floor: pool every S_i across all modules,
    drop rows below the drop_bottom_frac quantile. Modules left empty are filtered out."""
    V_hack = {n: V[:k_use] for n, V in V_hack.items()}
    V_sv = {n: S[:k_use] for n, S in V_sv.items()}
    if drop_bottom_frac > 0:
        thresh = torch.quantile(torch.cat(list(V_sv.values())), drop_bottom_frac)
        out = {}
        for n, V in V_hack.items():
            keep = V_sv[n] >= thresh
            if keep.any():
                out[n] = V[keep]
        return out
    return V_hack


def save_vhack(path: str, V_hack, V_sv):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    flat = {f"V/{n}": V.contiguous() for n, V in V_hack.items()}
    flat.update({f"S/{n}": S.contiguous() for n, S in V_sv.items()})
    save_file(flat, path)


def load_vhack(path: str):
    t = load_file(path)
    V_hack = {k[2:]: v for k, v in t.items() if k.startswith("V/")}
    V_sv = {k[2:]: v for k, v in t.items() if k.startswith("S/")}
    return V_hack, V_sv


def load_pairs_json(path: str) -> list[Pair]:
    return [Pair(**d) for d in json.loads(Path(path).read_text())]


def resolve_or_extract(model, tok, wrappers, v_hack_path: str, pairs_path: str | None,
                       k: int, τ_axis: float, n_heldout: int):
    """Load cached V if the path exists, else auto-extract on the current model and
    cache it (R5: auto-extract on cache-miss, cache-hit on rerun)."""
    if Path(v_hack_path).exists():
        logger.info(f"v_hack cache HIT: {v_hack_path}")
        return load_vhack(v_hack_path)
    logger.info(f"v_hack cache MISS: extracting -> {v_hack_path}")
    pairs = load_pairs_json(pairs_path) if pairs_path else default_pairs()
    V_hack, V_sv = extract_v_hack(model, tok, wrappers, pairs, k, τ_axis, n_heldout)
    save_vhack(v_hack_path, V_hack, V_sv)
    return V_hack, V_sv


def main(model: str, out_path: str, train_grads_path: str = "",
         dtype: str = "fp32", pairs_from_pool: str = "", n_heldout: int = 0,
         top_k: int = 4, τ_axis: float = 0.0):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModelForCausalLM.from_pretrained(model, dtype=dt)
    net.eval()
    wrappers = wrap(net)
    pairs = load_pairs_json(pairs_from_pool) if pairs_from_pool else default_pairs()
    V_hack, V_sv = extract_v_hack(net, tok, wrappers, pairs, top_k, τ_axis, n_heldout)
    save_vhack(out_path, V_hack, V_sv)
    if train_grads_path:
        Path(train_grads_path).parent.mkdir(parents=True, exist_ok=True)
        save_file({f"S/{n}": S for n, S in V_sv.items()}, train_grads_path)
    logger.info(f"saved v_hack -> {out_path} ({len(V_hack)} modules)")


if __name__ == "__main__":
    tyro.cli(main)
