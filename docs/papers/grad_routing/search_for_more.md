  Closest to your gradient-level method

  Directional Alignment Mitigates Reward Hacking in RL for LMs — Deng, Huang, Ozkara, Li, Thrampoulidis, Li, Park (2026) · https://arxiv.org/abs/2605.25189

  ▎ "We analyze this drift through dominant singular directions of parameter updates and show that reward-hacking runs exhibit substantially larger directional change than clean runs. Motivated by this observation, we introduce 
  ▎ trusted-direction projection, which constrains gradients to remain within a clean reference subspace."

  This is your near-twin: singular directions of updates + gradient projection. Theirs keeps gradients inside a clean subspace; yours subtracts a hack subspace. This is the baseline to differentiate from.

  Detecting and Suppressing Reward Hacking with Gradient Fingerprints (GRIFT) — Wang, Pham, Yin, Wang, Chen, Durrett, Ye (2026) · https://arxiv.org/abs/2604.16242

  ▎ "GRIFT computes gradients of the CoT conditioned on the prompt and compresses them into a compact representation, which is then used to assess whether the CoT reflects reward hacking behavior... integrating GRIFT into the rejection 
  ▎ fine-tuning pipeline... reduces reward hacking."

  Same level (gradient), same setting (RLVR loopholes). They detect+reject; you project. Their detector is a comparison point for your weak detector.

  When Reward Hacking Rebounds — Rui Wu, Ruixiang Tang (2026) · https://arxiv.org/abs/2604.01476

  ▎ "we propose Advantage Modification, which integrates shortcut concept scores into GRPO advantage computation to penalize hacking rollouts before policy updates."

  This is your "Rebound / advantage-level" contrast paper, now confirmed. Note the contrast is exact: they intervene in the GRPO advantage; you intervene in the gradient.

  Gradient-routing lineage (your "route" arm)

  Beyond Data Filtering: Knowledge Localization for Capability Removal (SGTM) — Shilov, Cloud, Gema, Goldman-Wetzler, Panickssery, Sleight, Jones, Anil (2025) · https://arxiv.org/abs/2512.05648

  ▎ "SGTM zero-masks selected gradients such that target domain examples only update their dedicated parameters... SGTM exhibits strong robustness to adversarial fine-tuning, requiring seven times more fine-tuning steps to reach 
  ▎ baseline performance on the forget set."

  Direct gradient-routing follow-up by the original authors, with the label-noise robustness framing that matches your weak-detector test.

  Confessions cluster (weak-detector-generalizes thesis)

  Spilling the Beans: Teaching LLMs to Self-Report Their Hidden Objectives — Chloe Li, Mary Phuong, Daniel Tan (2025) · https://arxiv.org/abs/2511.06626

  ▎ "the admission of factual errors in simple question-answering settings generalizes out-of-distribution (OOD) to the admission of hidden misaligned objectives in adversarial agentic settings."

  The SFT sibling to confessions, and its generalization claim is structurally your story: train on known case A, generalize to unknown case B.

  Monitoring Reasoning Models for Misbehavior... — Baker, Huizinga, Gao, Dou, Guan, Madry, Zaremba, Pachocki, Farhi (2025) · https://arxiv.org/abs/2503.11926

  ▎ "even weaker models like GPT-4o can effectively monitor stronger ones. However... with too much optimization, agents learn obfuscated reward hacking, hiding their intent within the CoT."

  The weak-monitor-catches-strong-model result, plus the warning that optimizing against the detector breeds obfuscation, which is a risk your gradient projection should be checked against.

  Caveat: the earlier subagents also named ~15 more (PCGrad, LEACE, MO-GRPO, the Schoen scheming evals, etc.) that I have not quote-verified. Want me to fetch quotes for those too, or drop these six into a docs/related_work.md with the
  links?