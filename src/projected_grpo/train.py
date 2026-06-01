"""GRPO / Dr.GRPO loop with SVD-basis gradient projection (pseudocode 05 + 06).

Inner GRPO_step ported from lsdefine/simple_GRPO grpo_vllm_one.py:64-95.
Unbiased normalization: Dr.GRPO, Liu et al. 2025, arXiv:2503.20783 (drop the
1/|o| length norm and the /σ_R group-std).

  generate -> grade -> backward -> project -> step

Arms (one knob): none -> measure_only | erase -> write g_proj | route -> +park δS_hack.
Presets: smoke (tiny-random, GPU bf16, the only gate) | fast | full.

  python -m projected_grpo.train smoke --intervention=erase ...
"""
from __future__ import annotations

import gzip
import json
import random
import sys
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import tyro
from loguru import logger
from safetensors.torch import save_file
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from projected_grpo.antipasto import ablate_quarantine, per_token_logps, ref_logprobs, wrap
from projected_grpo.extract_vhack_grad import extract_v_hack, postprocess_v_hack, resolve_or_extract
from projected_grpo.problems import default_pairs, load_problems
from projected_grpo.proj import cos_overlap, project_all
from projected_grpo.rewards import EnvMode, compute_reward

TINY = "llamafactory/tiny-random-qwen3"


def _setup_logging():
    """Route loguru through tqdm.write (so the bar never collides with log lines),
    single-char level icons, no timestamp/module on stdout; full DEBUG to a file."""
    logger.remove()
    # colorize=False: tqdm.write does NOT strip ANSI, so color leaks raw escapes into
    # piped pueue/sweep logs. Icons carry the level; color is not worth the noise.
    logger.add(lambda m: tqdm.write(m, end=""), colorize=False,
               format="{level.icon} {message}", level="INFO")
    Path("logs").mkdir(exist_ok=True)
    vlog = f"logs/{datetime.now():%Y%m%dT%H%M%S}_verbose.log"
    logger.add(vlog, format="{time:HH:mm:ss} | {level} | {message}", level="DEBUG")
    for lvl, icon in [("INFO", "I"), ("WARNING", "W"), ("ERROR", "E"), ("DEBUG", "D")]:
        logger.level(lvl, icon=icon)
    logger.info(f"verbose log: {vlog}")


@dataclass
class Config:
    # arm / identity
    intervention: str = "erase"            # none | erase | route
    seed: int = 41
    out_tag: str = ""
    # model
    model: str = TINY
    dtype: str = "fp32"
    device: str = "cpu"
    # loop
    steps: int = 30
    group: int = 2                         # G = G_s + G_t per prompt
    prompts_per_step: int = 2
    max_new: int = 16
    n_problems: int = 6
    env_mode: Optional[EnvMode] = None     # force a single mode (emergence cells)
    mix_ratio: float = 0.5                 # G_t / G  (teacher fraction)
    teacher_pool_dir: str = ""
    # optim
    lr: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.0
    warmup_frac: float = 0.1
    grad_clip: float = 10.0
    # GRPO
    beta: float = 0.0                      # KL to free ref (δS=0); 0 = off
    clip: float = 0.2
    # projection
    gate_mode: str = "one_sided"
    project_overshoot: float = 1.0
    preserve_magnitude: bool = False
    # v_hack
    v_hack_path: str = "out/vhack/v_hack_smoke.safetensors"
    vhack_pairs_path: Optional[str] = None
    v_hack_k: int = 4
    v_hack_top_k: int = 4
    v_hack_drop_bottom_frac: float = 0.0
    tau_axis: float = 0.0
    n_heldout: int = 0
    vhack_refresh_every: int = 0
    # route deploy-eval
    eval_ablate_every: int = 0
    eval_n_prompts: int = 2
    # cross-mech BLUF labels (in-distribution vs held-out)
    half_a: str = ""


PRESETS = {
    "smoke": Config(device="cuda", dtype="bf16"),   # tiny-random on GPU: walks the real cuda+bf16 path
    "fast": Config(model="Qwen/Qwen3-4B", device="cuda", dtype="bf16", steps=60, group=8,
                   prompts_per_step=4, max_new=512, lr=3e-3, adam_beta1=0.5, adam_beta2=0.9,
                   mix_ratio=0.125, v_hack_path="out/vhack/v_hack_full.safetensors"),
    "full": Config(model="Qwen/Qwen3-4B", device="cuda", dtype="bf16", steps=200, group=6,
                   prompts_per_step=43, max_new=1024, lr=1e-3, beta=1e-3,
                   v_hack_path="out/vhack/v_hack_full.safetensors"),
}


# ── teacher pool ────────────────────────────────────────────────────────────
def load_pool(pool_dir: str) -> dict[str, list[dict]]:
    """prompt -> list of cached graded rollouts (frozen reward labels)."""
    pool: dict[str, list[dict]] = {}
    for f in sorted(Path(pool_dir).glob("prompt_*.jsonl.gz")):
        rows = [json.loads(l) for l in gzip.open(f, "rt") if l.strip()]
        if rows:
            pool[rows[0]["prompt"]] = rows
    logger.info(f"loaded teacher pool: {len(pool)} prompts, "
                f"{sum(len(v) for v in pool.values())} rollouts from {pool_dir}")
    return pool


# ── inner GRPO step on one prompt ─────────────────────────────────────────────
def grpo_step(model, tok, wrappers, problem, pool_rows, G_s, G_t, cfg, pad_id, denom):
    """Returns (g_s, g_t, metrics) for this prompt; δS.grad left holding g_s+g_t.
    g_s/g_t are per-module dicts so the outer loop can accumulate + diagnose cin."""
    prompt_ids = tok(problem.prompt(), return_tensors="pt").input_ids.to(cfg.device)
    P = prompt_ids.shape[1]

    # rollouts: G_s live student + G_t cached teacher (one mixed group)
    seqs, texts, rewards, is_student, s_rew = [], [], [], [], []
    if G_s:
        with torch.no_grad():
            gen = model.generate(prompt_ids, do_sample=True, num_return_sequences=G_s,
                                 max_new_tokens=cfg.max_new, pad_token_id=pad_id)
        for row in gen:
            seqs.append(row)
            txt = tok.decode(row[P:], skip_special_tokens=True)
            r = compute_reward(txt, problem)          # graded once: reused for reward AND metrics
            texts.append(txt); s_rew.append(r); rewards.append(r.reward)
            is_student.append(1)
    teach = random.sample(pool_rows, min(G_t, len(pool_rows))) if (G_t and pool_rows) else []
    for row in teach:
        ids = tok(problem.prompt() + row["completion"], return_tensors="pt").input_ids[0].to(cfg.device)
        seqs.append(ids)
        texts.append(row["completion"])
        rewards.append(row["reward"])            # FROZEN teacher label (reproducible)
        is_student.append(0)

    G = len(seqs)
    R = torch.tensor(rewards, dtype=torch.float32, device=cfg.device)
    # per-source behavioural metrics (live grade for student, frozen for teacher)
    t_rows = list(teach)
    m = dict(reward=R.mean().item(),
             hack_s=_mean([r.exploited for r in s_rew]), gt_s=_mean([r.gt_correct for r in s_rew]),
             hack_t=_mean([r["exploited"] for r in t_rows]), gt_t=_mean([r["gt_correct"] for r in t_rows]))
    if s_rew:                                    # stash one student gen for the jsonl + the final coherence eyeball
        m["gen"] = dict(text=texts[0], hack=s_rew[0].exploited, gt=s_rew[0].gt_correct, reward=s_rew[0].reward)

    empty = {n: torch.zeros_like(w.δS) for n, w in wrappers.items()}
    if R.max() - R.min() < 1e-6:                 # zero-variance group => adv≡0 => pure waste (bail)
        return empty, dict(empty), m

    A = R - R.mean()                             # Dr.GRPO unbiased: NO /σ_R

    # pad the group to a rectangle
    L = max(s.shape[0] for s in seqs)
    o = torch.full((G, L), pad_id, dtype=torch.long, device=cfg.device)
    for i, s in enumerate(seqs):
        o[i, :s.shape[0]] = s

    logits = model(o).logits
    logπ = per_token_logps(logits, o)            # (G, L-1) with grad
    logπ_old = logπ.detach()
    ρ = torch.exp(logπ - logπ_old)               # ≡1 (single inner step) but keep the clip form
    A_tok = A[:, None]
    Lp = -torch.min(ρ * A_tok, torch.clamp(ρ, 1 - cfg.clip, 1 + cfg.clip) * A_tok)   # (G, L-1)
    if cfg.beta > 0:
        logπ_ref = ref_logprobs(model, wrappers, o)
        Lp = Lp + cfg.beta * (torch.exp(logπ_ref - logπ) - (logπ_ref - logπ) - 1)    # K3

    comp_mask = torch.zeros_like(Lp)             # completion tokens only, no pad
    for i in range(G):
        comp_mask[i, P - 1:] = (o[i, P:] != pad_id).float()
    is_s = torch.tensor(is_student, dtype=torch.float32, device=cfg.device)[:, None]

    # split student/teacher grads. zero+clone BETWEEN passes is load-bearing: after
    # L_t.backward, δS.grad already holds g_s+g_t (accumulation), so reset first.
    model.zero_grad()
    L_s = (Lp * comp_mask * is_s).sum() / denom
    L_s.backward(retain_graph=True)
    g_s = {n: w.δS.grad.detach().clone() for n, w in wrappers.items()}
    model.zero_grad()
    L_t = (Lp * comp_mask * (1 - is_s)).sum() / denom
    L_t.backward()
    g_t = {n: w.δS.grad.detach().clone() for n, w in wrappers.items()}
    m["loss"] = (L_s + L_t).item()
    return g_s, g_t, m


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def eval_hack_solve(model, tok, wrappers, problems, n, max_new, device, pad_id):
    """Greedy hack/solve on a fixed subset (route deploy-eval, under ablation)."""
    hk, sv = [], []
    for p in problems[:n]:
        ids = tok(p.prompt(), return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, do_sample=False, max_new_tokens=max_new, pad_token_id=pad_id)
        r = compute_reward(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True), p)
        hk.append(r.exploited); sv.append(r.gt_correct)
    return _mean(hk), _mean(sv)


def save_ckpt(wrappers, rows, tag):
    Path("out/ckpt").mkdir(parents=True, exist_ok=True)
    flat = {}
    for n, w in wrappers.items():
        flat[f"δS/{n}"] = w.δS.detach().cpu()
        flat[f"δS_hack/{n}"] = w.δS_hack.detach().cpu()
    save_file(flat, f"out/ckpt/ckpt{tag}.safetensors")
    Path(f"out/ckpt/rows{tag}.json").write_text(json.dumps(rows))


# ── outer loop ────────────────────────────────────────────────────────────────
def main(cfg: Config):
    _setup_logging()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    logger.info(f"argv: {' '.join(sys.argv)}")
    logger.info(f"cfg: {asdict(cfg)}")
    logger.info(f"=== TRAIN [{cfg.intervention}] seed={cfg.seed} steps={cfg.steps} G={cfg.group} mix={cfg.mix_ratio} ===")

    dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.dtype]
    tok = AutoTokenizer.from_pretrained(cfg.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dt).to(cfg.device)
    model.eval()
    wrappers = wrap(model)

    params = [w.δS for w in wrappers.values()] + [w.δS_hack for w in wrappers.values()]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2),
                            weight_decay=cfg.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.warmup_frac * cfg.steps), cfg.steps)

    # v_hack: load cached else auto-extract on the loaded model (all arms, so the
    # measure_only diagnostic fires for vanilla too -- 03: none -> measure_only).
    V_raw, V_sv = resolve_or_extract(model, tok, wrappers, cfg.v_hack_path,
                                     cfg.vhack_pairs_path, cfg.v_hack_top_k, cfg.tau_axis, cfg.n_heldout)
    V_hack = postprocess_v_hack(V_raw, V_sv, cfg.v_hack_k, cfg.v_hack_drop_bottom_frac)
    δS_ref = next(iter(wrappers.values())).δS              # V lives in δS's space: match dtype+device
    V_hack = {n: V.to(δS_ref) for n, V in V_hack.items()}  # extract/load give fp32[/cpu]; grads are bf16 cuda
    logger.info(f"V_hack over {len(V_hack)} modules (k_use={cfg.v_hack_k}); "
                f"arm={cfg.intervention} -> measure_only={cfg.intervention == 'none'}")

    problems = load_problems(cfg.n_problems, cfg.env_mode)
    pool = load_pool(cfg.teacher_pool_dir) if cfg.teacher_pool_dir else {}
    G_t = round(cfg.group * cfg.mix_ratio)
    G_s = cfg.group - G_t
    denom = cfg.group * cfg.max_new * cfg.prompts_per_step
    is_route = cfg.intervention == "route"

    logname = Path(f"logs/run_{cfg.intervention}{cfg.out_tag}_s{cfg.seed}.log")
    logname.parent.mkdir(exist_ok=True)
    log = logname.open("w")
    rows = []

    logger.info(
        "SHOULD (per-step JSONL rows + tqdm postfix): erase/route cin_t>0 and >cin_s early "
        "(teacher pool carries the hack signal), decaying as V goes stale; cout~0 under one_sided "
        "is an arithmetic identity, NOT efficacy; |δSh|>0 iff route, else 0; hack_s stays 0 on "
        "tiny-random (mechanics smoke). If cin_t~0 throughout -> V missed the live GRPO gradient.")
    pbar = tqdm(range(cfg.steps), desc=f"train[{cfg.intervention}] s{cfg.seed}",
                mininterval=120, maxinterval=120)
    for step in pbar:
        opt.zero_grad()
        g_s_acc = {n: torch.zeros_like(w.δS) for n, w in wrappers.items()}
        g_t_acc = {n: torch.zeros_like(w.δS) for n, w in wrappers.items()}
        ms = []
        for _ in range(cfg.prompts_per_step):
            prob = problems[random.randrange(len(problems))]
            g_s, g_t, m = grpo_step(model, tok, wrappers, prob, pool.get(prob.prompt(), []),
                                    G_s, G_t, cfg, pad_id, denom)
            for n in wrappers:
                g_s_acc[n] += g_s[n]; g_t_acc[n] += g_t[n]
            ms.append(m)
        for n, w in wrappers.items():
            w.δS.grad = g_s_acc[n] + g_t_acc[n]   # combined live GRPO gradient on the knob
            w.δS_hack.grad = None                 # quarantine moves ONLY via routed `removed` (set in project_all)

        cin_s = cos_overlap(g_s_acc, V_hack) if V_hack else 0.0
        cin_t = cos_overlap(g_t_acc, V_hack) if V_hack else 0.0
        cin = cout = fired = 0.0
        if V_hack:
            cin, cout, fired = project_all(wrappers, V_hack,
                                           measure_only=(cfg.intervention == "none"), route=is_route,
                                           gate_mode=cfg.gate_mode, overshoot=cfg.project_overshoot,
                                           preserve_magnitude=cfg.preserve_magnitude)
        gn = torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], cfg.grad_clip)
        opt.step(); sched.step()

        dSh = sum(w.δS_hack.detach().norm().item() for w in wrappers.values())
        row = dict(step=step, hack_s=_mean([m["hack_s"] for m in ms]), gt_s=_mean([m["gt_s"] for m in ms]),
                   hack_t=_mean([m["hack_t"] for m in ms]), gt_t=_mean([m["gt_t"] for m in ms]),
                   reward=_mean([m["reward"] for m in ms]), loss=_mean([m.get("loss", 0.0) for m in ms]),
                   gn=gn.item(), cin_s=cin_s, cin_t=cin_t, cin=cin, cout=cout, fired=fired,
                   dSh=dSh)
        if "gen" in ms[0]:                       # log the first prompt's first student gen (text + flags)
            row["gen"] = ms[0]["gen"]

        # online refresh: re-extract V against the CURRENT model (under quarantine ablation)
        if cfg.vhack_refresh_every and V_hack and step % cfg.vhack_refresh_every == 0:
            old = {n: V.clone() for n, V in V_hack.items()}
            with ablate_quarantine(wrappers):
                V_raw, V_sv = extract_v_hack(model, tok, wrappers, default_pairs(),
                                             cfg.v_hack_top_k, cfg.tau_axis, cfg.n_heldout)
            V_hack = postprocess_v_hack(V_raw, V_sv, cfg.v_hack_k, cfg.v_hack_drop_bottom_frac)
            V_hack = {n: V.to(δS_ref) for n, V in V_hack.items()}   # keep V in δS's bf16/cuda space
            shared = [n for n in V_hack if n in old]
            ov = _mean([abs((V_hack[n][0] @ old[n][0]).item()) for n in shared]) if shared else 0.0
            row["basis_overlap"] = ov            # GUARD: should sit near 1.0; <~0.2 => refresh rotated off-hack

        # deployment-time eval (route only): zero δS_hack, eval the ablated model
        if is_route and cfg.eval_ablate_every and step % cfg.eval_ablate_every == 0:
            with ablate_quarantine(wrappers):
                hd, sd = eval_hack_solve(model, tok, wrappers, problems, cfg.eval_n_prompts,
                                         cfg.max_new, cfg.device, pad_id)
            row["hack_deploy"], row["solve_deploy"] = hd, sd

        rows.append(row)
        log.write(json.dumps(row) + "\n"); log.flush()
        pbar.set_postfix(hack_s=f"{row['hack_s']:.2f}", cin_s=f"{cin_s:.2f}", cin_t=f"{cin_t:.2f}",
                         cout=f"{cout:.2f}", δSh=f"{dSh:.2f}", loss=f"{row['loss']:.3f}",
                         refresh=False)  # don't force a bar redraw every step (pollutes piped logs)
        if step % 25 == 0:
            save_ckpt(wrappers, rows, cfg.out_tag or f"_{cfg.intervention}_s{cfg.seed}")

    log.close()
    save_ckpt(wrappers, rows, cfg.out_tag or f"_{cfg.intervention}_s{cfg.seed}")
    _bluf(cfg, rows, wrappers, model, tok, problems, pad_id)


def _bluf(cfg, rows, wrappers, model, tok, problems, pad_id):
    """Final 30 lines: out path, argv, BLUF main-metric + cue, then a paper-ready
    table with direction arrows (token-efficient-logging skill)."""
    last5 = rows[-5:]
    hack = _mean([r["hack_s"] for r in last5]); solve = _mean([r["gt_s"] for r in last5])
    cin_t = _mean([r["cin_t"] for r in last5]); cout = _mean([r["cout"] for r in last5])
    dSh = rows[-1]["dSh"]
    out = f"logs/run_{cfg.intervention}{cfg.out_tag}_s{cfg.seed}.log"
    # mechanics cue: projection arms want the teacher hack signal present (cin_t>0); none is neutral
    cue = "\U0001F7E2" if (cfg.intervention == "none" or cin_t > 0) else "\U0001F534"

    summary = {"arm": cfg.intervention, "steps": len(rows),
               "hack↓": round(hack, 3), "solve→": round(solve, 3),
               "cin_t↑": round(cin_t, 3), "cout→0": round(cout, 3), "|δSh|": round(dSh, 2)}
    if cfg.intervention == "route":
        with ablate_quarantine(wrappers):
            hk_abl, sv_abl = eval_hack_solve(model, tok, wrappers, problems, cfg.eval_n_prompts or 2,
                                             cfg.max_new, cfg.device, pad_id)
        hk_kept, sv_kept = eval_hack_solve(model, tok, wrappers, problems, cfg.eval_n_prompts or 2,
                                           cfg.max_new, cfg.device, pad_id)
        summary["hk_kept↓"], summary["hk_abl↓"] = round(hk_kept, 3), round(hk_abl, 3)
        summary["sv_kept"], summary["sv_abl"] = round(sv_kept, 3), round(sv_abl, 3)

    caption = (f"Table: last-5-step means, arm={cfg.intervention} seed={cfg.seed}. "
               "hack↓ student exploit rate, solve→ gt-correct rate, cin_t↑ teacher-grad "
               "overlap with V (relu-before-agg), cout→0 residual hack-ward leak (identity under "
               "one_sided, not efficacy), |δSh| quarantine norm (>0 iff route).")
    logger.info(f"out: {out}")
    # Last student generation -- a coherence eyeball before the numbers. SHOULD: real
    # code/prose for the problem. If it is token salad the policy diverged and the eval
    # numbers below are moot. (Empty on tiny-random if no student rollout was graded.)
    last_gen = next((r["gen"] for r in reversed(rows) if r.get("gen")), None)
    if last_gen:
        logger.info(f"last student gen (hack={last_gen['hack']} gt={last_gen['gt']} "
                    f"reward={last_gen['reward']:.2f}):\n{last_gen['text']}")
    logger.info(f"main metric: {cue} hack_s={hack:.3f} solve={solve:.3f} cin_t={cin_t:.3f} cout={cout:.3f} "
                f"[arm={cfg.intervention},seed={cfg.seed},steps={len(rows)}]")
    print(f"\n{caption}\n")
    print(tabulate([summary], headers="keys", tablefmt="github", floatfmt=".3f"))


def cli():
    """`train.py {smoke|fast|full} [--field=val ...]`: preset selects defaults, flags override."""
    if len(sys.argv) < 2 or sys.argv[1] not in PRESETS:
        raise SystemExit(f"usage: train.py {{{'|'.join(PRESETS)}}} [overrides]")
    main(tyro.cli(Config, args=sys.argv[2:], default=PRESETS[sys.argv[1]]))


if __name__ == "__main__":
    cli()
