# ── Outer training loop ────────────────────────────────────────────────
# Source: train.py main(). Ties together 01..05. Reads top-to-bottom.

def main(cfg):                                          # cfg: Smoke | Fast | Full preset
    model = load(cfg.model)
    wrappers = wrap(model)                              # 01_adapter: SVD, freeze, attach δS knobs
    opt   = AdamW([δS, δS_hack], lr=cfg.lr, β=(cfg.adam_β1, cfg.adam_β2), wd=cfg.weight_decay)
    sched = warmup_cosine(opt, warmup=cfg.warmup_frac * cfg.steps)

    # v_hack: load cached, else auto-extract on the already-loaded model (cheap).
    # vanilla arm (intervention="none") uses V=None and skips projection entirely.
    V_hack = None if cfg.intervention=="none" else postprocess_v_hack(
        *resolve_or_extract(model, wrappers, cfg), cfg.v_hack_k, cfg.v_hack_drop_bottom_frac)

    problems     = load_problems(cfg.n_problems, cfg.env_mode)   # hint injected per mode
    teacher_pool = load_pool(cfg.teacher_pool_dir)               # prompt → cached graded rollouts
    G_t = round(cfg.group * cfg.mix_ratio);  G_s = cfg.group - G_t

    for step in range(cfg.steps):
        opt.zero_grad();  g_s, g_t = {}, {}
        # ── accumulate grad over P prompts (raises chance ≥1 group has reward variance)
        for _ in range(cfg.prompts_per_step):
            prob = problems[rng.randint()]
            grpo_step(model, prob, teacher_pool, G_s, G_t, cfg.beta, cfg.clip, cfg.unbiased)
            accumulate(g_s, g_t)                        # 05: split student/teacher grads
        δS.grad ← g_s + g_t                             # combined live GRPO gradient on the knob

        # ── intervene (03_project) ─────────────────────────────────────
        # cos() = ‖relu(V@g)‖/‖g‖, RELU-BEFORE-AGG (see 03): the hack-ward fraction,
        # not a signed sum where anti-hack axes cancel. cin_s on student grad, cin_t
        # on teacher grad; discriminator wants cin_t > cin_s.
        cin_s = cos(g_s, V_hack);  cin_t = cos(g_t, V_hack)
        cin, cout, fired = project_all(wrappers, V_hack,
                                       measure_only=(cfg.intervention=="none"),
                                       route=(cfg.intervention=="route"),
                                       gate_mode=cfg.gate_mode, overshoot=cfg.project_overshoot)

        gn = clip_grad_norm([δS, δS_hack], cfg.grad_clip)
        opt.step();  sched.step()

        # ── online refresh: re-extract V against the CURRENT model ──────
        # V goes stale fast (cin_t decays ~0.27→0.07 by step 10). Re-extract every
        # N steps to track where the student is being pulled NOW. For route, ablate
        # the quarantine during extraction (else V rotates off-hack: the routed
        # capability has left the observable main path).
        if cfg.vhack_refresh_every and step % cfg.vhack_refresh_every == 0:
            with ablate_quarantine(wrappers):          # δS_hack→0 during extract (no-op for erase)
                V_new = postprocess_v_hack(*extract_v_hack(model, wrappers, PAIRS, ...), ...)
            # GUARD (blog): log basis_overlap_with_prev = mean|⟨V_new_i, V_old_i⟩|.
            # Should sit near 1.0. <~0.2 ⇒ the re-extraction ROTATED the basis off-hack
            # (for route, this is the symptom of extracting through a live quarantine) —
            # refresh is then HARMFUL, not helpful. Don't refresh blind.
            V_hack = V_new

        # ── deployment-time eval (route only) ──────────────────────────
        # The TRAINING hack curve looks vanilla for route (forward still hacks);
        # routing's benefit only shows once the quarantine is ablated. So every N
        # steps: zero δS_hack, eval hack/solve on a fixed prompt subset → the
        # hack_deploy/solve_deploy series the dynamics plot uses.
        if cfg.intervention=="route" and step % cfg.eval_ablate_every == 0:
            with ablate_quarantine(wrappers):
                hack_deploy, solve_deploy = eval_hack_solve(model, eval_prompts)

        log_row(step, rew, hack_s, hack_t, gt_s, gt_t, loss, gn, cin_s, cin_t, cin, cout, fired)
        if step % 25 == 0: save_ckpt(δS, δS_hack, rows)

    bluf(...)   # final: kept-vs-ablated hack/solve, per-mechanism breakdown (substrate)

# Presets (single source of truth on the Config dataclasses):
#   smoke: tiny-random-qwen3, CPU, 30 steps, G=2  → `just smoke`, the only gate
#   fast : Qwen3-4B, 60 steps, G=8, aggressive Adam (~15 min iteration)
#   full : Qwen3-4B, 200 steps, G=6, pp=43 (paper's 256 gens/step), β=1e-3
