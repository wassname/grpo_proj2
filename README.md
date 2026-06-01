# projected_grpo

Motovation: Can we erase or route reward hacking using a "cheat direction"?

Hypothesis: a weak detector that only knows some hacks generalises to suppress
the unknown ones, which is the situation any real deployment is in.

Experiment: We can take contrastive (hack, clean) prompts to extract a "cheat direction" in the model's activation space.Then during GRPO we erase or route "cheat" gradients. 

In early experiment this worked, but the model quickly routed around it, so we take SGTM's approach of absorbing the cheat direction in a non-adverserial manner.

## How?

Like [SGTM](https://arxiv.org/abs/2512.05648) (Selective Gradient Masking, which
localises unwanted capabilities into deletable parameters during pretraining),
but for RL reward-hacking. When GRPO teaches a coding model to cheat, we route
the "cheat direction" out of each gradient into a throwaway adapter we delete at
deployment, or just erase it. SGTM decides what to route using a per-example
classifier label. We have no such label: instead we route by a "cheat direction"
extracted from contrastive (hack, clean) prompts (see `problems.py`).

Environment: We reuse the simple [ariahw/rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking),
extended from one to four distinct loophole hacks. Researchers may find this useful as a toy environment. But to avoid 65 hours experiments we use teacher examples to bootstrap it, resulting in 30 minutes experiments.


Full story: [blog draft](docs/blog/20260529_gradient_projection_vs_reward_hacking_LW_draft.md).

Try it: `uv sync && just smoke`. 

Building on the code: [AGENTS.md](AGENTS.md).
