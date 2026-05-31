# AGENTS.md — projected_grpo

**This is novel ML research.** Not in your training data. Extrapolate carefully and without overconfidence.
**This is reasearch code** We want readable, fail fast, pseudocode like code
**Editing**: Plan, Implement, Smoke test (see setup-repo skill), subagent review, commit


see @README.md for project overview

## Project in one paragraph

Test whether SVD-basis gradient projection against an extracted hack-direction
reduces reward-hack rate in GRPO on Nanda's LeetCode benchmark. Differs from
Rebound (Wu & Tang 2026) by intervening at the *gradient* level rather than the
*advantage* level. Differs from AntiPaSTO (the user's prior work) by using
unpaired GRPO rollouts rather than paired-preference contrast.

Hypothesis
> We can find and the "reward hacking direciton", and reduce RL from learning reward hacking
> Specifically we can get the reward hacking directions by contrasting G_hack and G_not, from GRPO upates on hacky and nonhacky completions.
> Then during normal GRPO training we can erase the direction from the gradients collected on each learnable parameter in a low rank adapter, and this will reduce the reward hacking rate.

Motovation: 
We want to take the tool AI labs already use, and make them better for aligment using scalable self-supervised methods. Here the hope is that intervening in the gradient itself, rather than in the reward, can stop the student picking up the hack.

Inherit global rules from `~/.claude/CLAUDE.md`.


## Things the user has had to explain many times:

- We cannot cheat and use all reward hacks to stop hacks. During deployment there are known hacks and unknown hacks. We want to make an alignment toolslabs want to use. So it's ok to have a weak eward hack detector than can detect hack type A but not B, then use the gradient from A to try to stop the learning of B, and this mimicks the generalisation to unknown hacks that happens at deployment.
- do not overconfidentaly diagnoses. if you cant think of 3+ plausible hypothesis - including bugs, subtle failures, and you being wrong about concepts - then you have lost perspective and narrow vision
- I'd often afk so dont stop and ask me a question you know the likely answer, or I've already indicated or asked for, or where there is only on answer "waiting for your go ahead". I'd rather you just commit and go ahead

## Extra instructions:

- When you queue a job, follow with `pueue follow | tail` in bg so you are woken on fail or finish
- for every task, it's a goal of subgoal, be clear on UAT (user acceptance test), use you task list to track the goal and non trivial UAT, and it's not finished untill you have 1) collected the evidence 2) sanity checked it with a fresh eyes subagent 3) given the user the link to it's location
- say less, just answer my questions, and concisly address the top point unless I ask for more - otherwise there is too much for me to review and read

## Files

- Read [docs/spec.md](spec.md) for the preregistered plan.
- Read [docs/brainstorm/extracted_prefs.md](docs/brainstorm/extracted_prefs.md) for design rationale.
- New sweep arms get recipes in [justfile](justfile) with `# H:` hypothesis comments.
- `just smoke` before any real run (~1-2 min, beartype on, real pipeline on tiny inputs).
- Real runs go through `pueue` on the 96GB GPU box. Label each job with `why:` and `resolve:`.
- Head [docs/RESEARCH_JOURNAL.md](docs/RESEARCH_JOURNAL.md) for latest results.
- No `tests/` dir; `smoke` is the correctness gate.

On persona pairs
- ./docs/personas/how_to_rewrite_pairs.md
- ./docs/personas/how_to_write_personas.md
- ./docs/personas/personas_kept.md

On concepts such as "what are contrastive pairs" or "why SVD space" grep
- ./docs/vendor/AntiPaSTO_concepts/README.md

For the original paper
- LessWrong post: ./docs/papers/2025_lw_ariahw_steering-rl-training-benchmarking-interventions.md
- Code: ./docs/vendor/rl-rewardhacking