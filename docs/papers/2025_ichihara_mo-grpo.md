<!-- source: https://r.jina.ai/https://arxiv.org/html/2509.22047v2 fetched: 2026-05-23 -->

Title: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems

URL Source: https://arxiv.org/html/2509.22047v2

Markdown Content:
Yuki Ichihara 1 Yuu Jinnai 2 Tetsuro Morimura 2 Mitsuki Sakamoto 2

Ryota Mitsuhashi 2 Eiji Uchibe 3

1 Nara Institute of Science and Technology 2 CyberAgent 

3 Advanced Telecommunications Research Institute International

###### Abstract

Group Relative Policy Optimization (GRPO) has been shown to be an effective algorithm when an accurate reward model is available. However, such a highly reliable reward model is not available in many real-world tasks. In this paper, we particularly focus on multi-objective settings, in which we identify that GRPO is vulnerable to reward hacking, optimizing only one of the objectives at the cost of the others. To address this issue, we propose MO-GRPO, an extension of GRPO with a simple normalization method to reweight the reward functions automatically according to the variances of their values. We first show analytically that MO-GRPO ensures that all reward functions contribute evenly to the loss function while preserving the order of preferences, eliminating the need for manual tuning of the reward functions’ scales. Then, we evaluate MO-GRPO experimentally in four domains: (i) the multi-armed bandits problem, (ii) simulated control task (Mo-Gymnasium), (iii) machine translation tasks on the WMT benchmark (En-Ja, En-Zh), and (iv) instruction following task. MO-GRPO achieves stable learning by evenly distributing correlations among the components of rewards, outperforming GRPO, showing MO-GRPO to be a promising algorithm for multi-objective reinforcement learning problems.

Instruction Translate the following English into easily readable Japanese.\nOver the past decade, our lives have changed through technology, with many working from home, …jReadability \uparrow BLEURT \uparrow
GRPO Over the past ten years, our lives have changed a lot because of technology. …\mathbf{0.99}0.57
MO-GRPO 過去の10年で、技術の進化により、多くの人がホームワークから仕事をしているようになり…(Translation: In the past decade, technological advances have enabled many people to work from home work…)0.40 0.69

Table 1: (Machine translation) Generation examples of GRPO and MO-GRPO by Llama (Llama-3.2-3B-Instruct). GRPO optimizes only the Japanese readability score (jReadability) by avoiding using difficult Japanese words, eventually stops using any Japanese characters, ignoring the translation accuracy score (BLEURT), resulting in generating non-Japanese text, which defeats the purpose of the translation. On the other hand, MO-GRPO evenly optimizes both objectives, achieving improvement on both objectives as intended. Generation examples from other LLMs and the results of outputs during training are shown in Appendix[D](https://arxiv.org/html/2509.22047v2#A4 "Appendix D Reward hacking Examples ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 

## 1 Introduction

Reward hacking is a phenomenon in which an agent overfits to a misspecified reward model, failing to optimize for the true intended objective Skalse et al. ([2022](https://arxiv.org/html/2509.22047v2#bib.bib43 "Defining and Characterizing Reward Gaming")); Gao et al. ([2023](https://arxiv.org/html/2509.22047v2#bib.bib44 "Scaling Laws for Reward Model Overoptimization")); Rafailov et al. ([2024](https://arxiv.org/html/2509.22047v2#bib.bib68 "Scaling laws for reward model overoptimization in direct alignment algorithms")). Recent work proposes Group Relative Policy Optimization (GRPO; Shao et al.[2024](https://arxiv.org/html/2509.22047v2#bib.bib1 "Deepseekmath: Pushing the Limits of Mathematical Reasoning in Open Language Models"); Liu et al.[2025](https://arxiv.org/html/2509.22047v2#bib.bib2 "Understanding r1-zero-like training: A critical perspective")) to enhance the reasoning capability of Large Language Models (LLMs; Liu et al.[2024](https://arxiv.org/html/2509.22047v2#bib.bib66 "Deepseek-v3 technical report"); Yang et al.[2025](https://arxiv.org/html/2509.22047v2#bib.bib64 "Qwen3 technical report")), demonstrating that the method can significantly improve the performance of the agent using a reward model with high accuracy. However, it is difficult to obtain high-accuracy reward models in many real-world tasks.

In this work, we evaluate GRPO’s performance in a specific scenario where we only have access to under-specified reward models that do not accurately represent the task’s objective on their own. In particular, we study a practical scenario where we specify the intended behavior of the agent using multiple reward models. The scenario involves a real-world machine translation task. The objective is to generate texts that are both (1) consistent with the content in the original language and (2) easy to read in the target language.

Our analysis shows that the loss function of the GRPO leads to optimizing the reward functions with higher variances while ignoring the lower ones, which may result in an undesirable policy. We empirically evaluate GRPO on multi-objective reinforcement learning problems, where we also observe reward hacking behavior of GRPO that it tends to ignore rewards with lower variances and only optimizes the ones with higher variances, resulting in an unintended behavior (e.g., hacking the readability objective while ignoring the translation consistency; Table[1](https://arxiv.org/html/2509.22047v2#S0.T1 "Table 1 ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")).

To resolve this problem, we propose GRPO with a simple automated normalization method to the objectives, which we call MO-GRPO (Multi-Objective Group Relative Policy Optimization). MO-GRPO normalizes the advantage functions for each objective so that their variances are scaled evenly. This ensures that all reward functions contribute equally to updating the policy, regardless of their variance scales (Theorem[2](https://arxiv.org/html/2509.22047v2#Thmtheorem2 "Theorem 2 (Correlation between a reward function and advantage function with MO-GRPO). ‣ 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")). In this way, it prevents any objectives from being ignored in the training process, mitigating the reward hacking behavior of GRPO on multiple objectives. Our normalization technique maintains the original preference ordering under positive affine transformations of reward scales, even in cases where GRPO fails to preserve this ordering.

We evaluate MO-GRPO experimentally on multi-armed bandit, simulated control Felten et al. ([2023](https://arxiv.org/html/2509.22047v2#bib.bib59 "A toolkit for reliable benchmarking and research in multi-objective reinforcement learning")), machine translation Kocmi et al. ([2024](https://arxiv.org/html/2509.22047v2#bib.bib25 "Findings of the WMT24 general machine translation shared task: the LLM era is here but MT is not solved yet")) problems, and instruction following task with multiple objectives. The result shows that MO-GRPO successfully mitigates reward hacking in the problem settings where GRPO incurs reward hacking (Table[1](https://arxiv.org/html/2509.22047v2#S0.T1 "Table 1 ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")), resulting in a policy that is desirable for the given task.

## 2 Related Works

Reward hacking is one of the known problems in reinforcement learning (Amodei et al., [2016](https://arxiv.org/html/2509.22047v2#bib.bib40 "Concrete Problems in AI Safety"); Ziegler et al., [2020](https://arxiv.org/html/2509.22047v2#bib.bib41 "Fine-tuning language models from human preferences"); Stiennon et al., [2020](https://arxiv.org/html/2509.22047v2#bib.bib42 "Learning to summarize with human feedback"); Skalse et al., [2022](https://arxiv.org/html/2509.22047v2#bib.bib43 "Defining and Characterizing Reward Gaming"); Gao et al., [2023](https://arxiv.org/html/2509.22047v2#bib.bib44 "Scaling Laws for Reward Model Overoptimization")). In the context of Large Language Models (LLMs), reward hacking often becomes a problem in the alignment process Rafailov et al. ([2024](https://arxiv.org/html/2509.22047v2#bib.bib68 "Scaling laws for reward model overoptimization in direct alignment algorithms")) where the LLMs are optimized to generate outputs that maximize the score of a reward model trained using human feedback (e.g., RLHF; (Ziegler et al., [2020](https://arxiv.org/html/2509.22047v2#bib.bib41 "Fine-tuning language models from human preferences"); Stiennon et al., [2020](https://arxiv.org/html/2509.22047v2#bib.bib42 "Learning to summarize with human feedback"))). Prior work shows that if the reward model is inaccurate, the LLM can learn to generate sentences that increase the score from the reward model, even if it does not improve the true intended objective of the training. For example, it might produce outputs that are very long, overly polite, or stylistically pleasing to the reward model, even if they are factually incorrect or unhelpful, because such attributes are spuriously correlated with high reward scores Gao et al. ([2023](https://arxiv.org/html/2509.22047v2#bib.bib44 "Scaling Laws for Reward Model Overoptimization")). This over-optimization on a proxy reward can lead to a decrease in the true objective and alignment of the model’s outputs Pan et al. ([2022](https://arxiv.org/html/2509.22047v2#bib.bib45 "The effects of reward misspecification: mapping and mitigating misaligned models")); Gleave et al. ([2020](https://arxiv.org/html/2509.22047v2#bib.bib46 "Adversarial policies: attacking deep reinforcement learning")); Kim et al. ([2025](https://arxiv.org/html/2509.22047v2#bib.bib62 "Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization")). There are several fine-tuning methods for multi-objective problems that can be used to solve this problem Zhou et al. ([2024](https://arxiv.org/html/2509.22047v2#bib.bib70 "Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization")); Li et al. ([2025](https://arxiv.org/html/2509.22047v2#bib.bib71 "Self-Improvement Towards Pareto Optimality: Mitigating Preference Conflicts in Multi-Objective Alignment")). However, they face some problems. The problem of Zhou et al. ([2024](https://arxiv.org/html/2509.22047v2#bib.bib70 "Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization")) is that, as the ratio of conflicts in the training data increase, the Pareto Fronts gradually move downwards, showing significant performance decreases. Li et al. ([2025](https://arxiv.org/html/2509.22047v2#bib.bib71 "Self-Improvement Towards Pareto Optimality: Mitigating Preference Conflicts in Multi-Objective Alignment")) can resolve this problem, but the proposed method involves response sampling, refinement, filtering, and fine-tuning, which could introduce computational overhead compared to simpler methods.

## 3 Group Relative Policy Optimization (GRPO)

GRPO is a reinforcement learning algorithm Shao et al. ([2024](https://arxiv.org/html/2509.22047v2#bib.bib1 "Deepseekmath: Pushing the Limits of Mathematical Reasoning in Open Language Models")) which is usually used for online and on-policy learning. For a given state, a policy generates multiple outputs and learns to generate outputs with higher relative reward scores compared to the rest of the outputs. p_{\mathcal{Q}} is the distribution over the initial state (prompt) (q\sim p_{\mathcal{Q}}). The policy \pi_{\theta}\left(\cdot\mid q\right) outputs the action (sentence) o_{g} based on the initial state q from the action space. Formally, let R_{i} be the i-th reward function, which is a mapping from a prompt-output pair to a scalar value, and assume that there are K reward functions. For each prompt q, GRPO samples a group of outputs \mathbf{o}=\left\{o_{1},o_{2},...,o_{G}\right\} from the old policy \pi_{\theta_{\text{old}}} and then optimizes the policy model by maximizing the following objective:

\displaystyle\mathcal{J}(\pi_{\theta})\displaystyle=\mathbb{E}\Biggl[\sum_{g=1}^{G}\frac{1}{G}\frac{1}{|o_{g}|}\frac{\pi_{\theta}\bigl(o_{g}\mid q\bigr)}{\pi_{\theta_{\mathrm{old}}}\bigl(o_{g}\mid q\bigr)}A_{g}\Biggr]
\displaystyle-\beta\text{KL}(\pi_{\theta},\pi_{\theta_{\text{ref}}})(1)

where \beta is a hyperparameter, and KL is Kullback–Leibler (KL) divergence. Since we omit symbols such as the threshold \epsilon and \mathrm{min} operation for simplicity, we note formal expressions in the Appendix[B](https://arxiv.org/html/2509.22047v2#A2 "Appendix B Formal Formulation of GRPO ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems").

A_{g} represents the normalized advantage value of the sentence o_{g} using K reward models:

A_{g}=\frac{\sum_{i=1}^{K}R_{i}(q,o_{g})-\mathrm{mean}_{\mathbf{o}}\bigl(\sum_{i=1}^{K}R_{i}(q,\mathbf{o})\bigr)}{\mathrm{std}_{\mathbf{o}}\bigl(\sum_{i=1}^{K}R_{i}(q,\mathbf{o})\bigr)}.(2)

The advantage value A_{g} is computed without normalizing the scale of the reward functions (Eq.[2](https://arxiv.org/html/2509.22047v2#S3.E2 "In 3 Group Relative Policy Optimization (GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")). Consequently, when rewards differ in scale or variance, the advantage value can be dominated by the value from a high variance reward function.

###### Theorem 1(Correlation between reward function and advantage function with GRPO).

Assume the G\rightarrow\infty. The correlation coefficient between an individual reward function R_{i} and the advantage A_{g} is the ratio of R_{i}’s standard deviation \sigma_{i} to the standard deviation of the total reward \sigma.

\operatorname{Corr}(R_{i}(q,o_{g}),A_{g})=\frac{\sigma_{i}^{2}+X}{\sigma\sigma_{i}}(3)

where X=\sum_{j\neq i}\operatorname{Cov}(R_{i},R_{j}), \operatorname{Cov}(\cdot,\cdot) is covariance.

The proof is in Appendix[E.1](https://arxiv.org/html/2509.22047v2#A5.SS1 "E.1 Proof of Theorem 1 ‣ Appendix E Proof of Theorem and Proposition ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems").

Theorem[1](https://arxiv.org/html/2509.22047v2#Thmtheorem1 "Theorem 1 (Correlation between reward function and advantage function with GRPO). ‣ 3 Group Relative Policy Optimization (GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows that the advantage function in GRPO is more strongly correlated with reward components that exhibit higher variance. This shows that GRPO learns to optimize reward functions with higher variances than the lower ones and may lead to unintended behavior, which we show empirically in Section[5](https://arxiv.org/html/2509.22047v2#S5 "5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems").

![Image 1: Refer to caption](https://arxiv.org/html/2509.22047v2/x1.png)

Figure 1: (Simulated experiment) Comparison of the advantage values of GRPO and MO-GRPO on a toy example with two reward functions with different sizes of variances (1 and 5). The advantage values of GRPO (left figure) are dominated by the high variation reward (R_{2}), indicating that the algorithm is sensitive to the relative scales of the rewards. In contrast, the advantage values of MO-GRPO (right figure) are invariant with the scale of the reward models, which shows that MO-GRPO is an easy-to-use algorithm for multi-objective learning tasks that does not require manual tuning of the reward models to avoid reward hacking. 

## 4 Multi-Objective GRPO (MO-GRPO)

To solve the problem that GRPO is affected by the scale of the variance of the reward functions, we propose Multi-Objective GRPO (MO-GRPO). By computing a separate advantage function for each reward, our framework adjusts for differences in reward variance and enables more stable learning.

\displaystyle\mathcal{J}(\pi_{\theta})\displaystyle=\mathbb{E}\Biggl[\sum_{g=1}^{G}\frac{1}{G}\frac{1}{|o_{g}|}\frac{\pi_{\theta}\bigl(o_{g}\mid q\bigr)}{\pi_{\theta_{\mathrm{old}}}\bigl(o_{g}\mid q\bigr)}\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}A_{g}^{\mathrm{MO}}\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\Biggr]
\displaystyle-\beta\text{KL}(\pi_{\theta},\pi_{\theta_{\text{ref}}})(4)

where A_{g}^{\mathrm{MO}} is defined as follows:

\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}A_{g}^{\mathrm{MO}}\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}=\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\sum^{K}_{i=1}\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\frac{R_{i}(q,o_{g})-\mathrm{mean}_{\mathbf{o}}\bigl(R_{i}(q,\mathbf{o})\bigr)}{\mathrm{std}_{\mathbf{o}}\bigl(R_{i}(q,\mathbf{o})\bigr)}.(5)

Note that MO-GRPO rescales the reward individually, then aggregating over the reward functions, whereas vanilla GRPO rescales it after all the reward values are aggregated into a single value (Equation[2](https://arxiv.org/html/2509.22047v2#S3.E2 "In 3 Group Relative Policy Optimization (GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")). Thus, MO-GRPO ensures a consistent correlation between each advantage function and its corresponding reward function.

###### Theorem 2(Correlation between a reward function and advantage function with MO-GRPO).

Assume that the number of samples G\to\infty. The correlation of the advantage functions A_{g}^{\mathrm{MO}} with each reward function R_{i} for any o_{g} remains constant.

\displaystyle\operatorname{Corr}(R_{i}(q,o_{g}),A_{g}^{\mathrm{MO}})\displaystyle=\frac{1+Z}{\sqrt{K+Y}}(6)

where Y=\sum_{j=1}^{K}\sum_{l\neq j}\frac{\operatorname{Cov}\left(R_{l},R_{j}\right)}{\sigma_{l}\sigma_{j}} and Z=\sum_{j\neq i}\frac{\operatorname{Cov}\left(R_{i},R_{j}\right)}{\sigma_{i}\sigma_{j}}, \operatorname{Cov}(\cdot,\cdot) is covariance.

The proof is in Appendix[E.2](https://arxiv.org/html/2509.22047v2#A5.SS2 "E.2 Proof of Theorem 2 ‣ Appendix E Proof of Theorem and Proposition ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems").

Theorem[2](https://arxiv.org/html/2509.22047v2#Thmtheorem2 "Theorem 2 (Correlation between a reward function and advantage function with MO-GRPO). ‣ 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows that the advantage function in MO-GRPO is roughly equal to \frac{1}{\sqrt{K}} for every reward function R_{i}, with some effect of correlation between the reward functions. Specifically, if all the reward functions are uncorrelated with each other, then the correlation of the reward and the advantage is exactly \frac{1}{\sqrt{K}} for all the reward functions:

###### Corollary 1(Correlation between a reward function and advantage function with MO-GRPO under certain assumptions).

Assume that the K reward functions R_{i} are mutually uncorrelated and the number of samples G\to\infty. The correlation of the advantage functions A_{g}^{\mathrm{MO}} with each reward function R_{i} for any o_{g} remains constant.

\operatorname{Corr}(R_{i}(q,o_{g}),A_{g}^{\mathrm{MO}})=\frac{1}{\sqrt{K}}(7)

The proof follows immediately from Theorem [2](https://arxiv.org/html/2509.22047v2#Thmtheorem2 "Theorem 2 (Correlation between a reward function and advantage function with MO-GRPO). ‣ 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). The result shows that the reward functions of the MO-GRPO have roughly the same amount of influence on the policy update regardless of their variances. Thus, MO-GRPO does not ignore reward functions with lower variances, which could lead to unintended behavior. We show this property empirically in Section[5](https://arxiv.org/html/2509.22047v2#S5 "5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems").

#### Example.

Here, we demonstrate how MO-GRPO preserves the sensitivity of each reward function within the advantage function. We consider two reward functions and three outputs \color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}R_{1}:[0.1,0.5,0.9]and R_{2}:[0.15,0.13,0.05]. A case of GRPO, reward mean is 0.61 and standard deviation is 0.29, therefore, the advantage functions are [-1.38,0.43,\mathbf{0.95}]. On the other hand a case of MO-GRPO, reward means are [\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}0.5\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0},\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}0.11\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}] and standard deviation: [\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}0.33\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0},\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}0.04\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}], therefore advantage function are [\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}-1.22\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}+0.93\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0},\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}0.0\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}+0.46\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0},\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}1.22\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}-1.39\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}]. From this, it can be said that MO-GRPO successfully reflects the superiority of R_{2} in the advantage function. This can be demonstrated by the following Theorem[2](https://arxiv.org/html/2509.22047v2#Thmtheorem2 "Theorem 2 (Correlation between a reward function and advantage function with MO-GRPO). ‣ 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems").

#### Simulated experiment.

Figure[1](https://arxiv.org/html/2509.22047v2#S3.F1 "Figure 1 ‣ 3 Group Relative Policy Optimization (GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows the comparison of the advantage functions of GRPO (Eq.[2](https://arxiv.org/html/2509.22047v2#S3.E2 "In 3 Group Relative Policy Optimization (GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")) and MO-GRPO (Eq.[5](https://arxiv.org/html/2509.22047v2#S4.E5 "In 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")) with two reward functions with low and high variances. The reward functions return a value sampled from a normal distribution R_{1}\sim\mathcal{N}\left(1,1^{2}\right) and R_{2}\sim\mathcal{N}\left(1,5^{2}\right). The advantage value by GRPO is significantly influenced by R_{2} while the effect of R_{1} is negligible. This motivates the agent to maximize the value of R_{2} even at the cost of losing R_{1}. Conversely, the advantage value calculated by MO-GRPO successfully considers both reward functions, even when their variances differ significantly. This indicates that, when using MO-GRPO, one does not need to manually adjust the scale of the reward models for a given task to prevent the agent from performing unbalanced optimization.

#### Invariance to positive affine transformation.

An additional advantage of MO-GRPO is its invariance under positive affine transformations of reward functions.

###### Proposition 1(Affine Invariance of MO-GRPO Advantage).

Let o_{a} and o_{b} be two possible outputs, and let \mathcal{R}=\{R_{i}\}_{i=1}^{K} be a set of reward functions. Consider a transformed set \mathcal{R}^{\prime}=\{R^{\prime}_{i}=a_{i}R_{i}+b_{i}\}_{i=1}^{K} with a_{i}>0. Then, the preference ordering induced by MO-GRPO is invariant under such positive affine transformations:

\displaystyle A^{\mathrm{MO}}_{a}\geq A^{\mathrm{MO}}_{b}\iff A^{\mathrm{MO^{\prime}}}_{a}\geq A^{\mathrm{MO^{\prime}}}_{b}

where A^{\mathrm{MO^{\prime}}}_{a}=\sum^{K}_{i=1}\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\frac{R^{\prime}_{i}(q,o_{a})\;-\;\mathrm{mean}_{\mathbf{o}}\bigl(R^{\prime}_{i}(q,\mathbf{o})\bigr)}{\mathrm{std}_{\mathbf{o}}\bigl(R^{\prime}_{i}(q,\mathbf{o})\bigr)}.

The proof of Proposition[1](https://arxiv.org/html/2509.22047v2#Thmproposition1 "Proposition 1 (Affine Invariance of MO-GRPO Advantage). ‣ Invariance to positive affine transformation. ‣ 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") is in Appendix[E.3](https://arxiv.org/html/2509.22047v2#A5.SS3 "E.3 Proof of Proposition 1 ‣ Appendix E Proof of Theorem and Proposition ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). Meanwhile, Theorem[2](https://arxiv.org/html/2509.22047v2#Thmtheorem2 "Theorem 2 (Correlation between a reward function and advantage function with MO-GRPO). ‣ 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows that MO-GRPO does not require engineering efforts to normalize the scale of the reward functions relative to other reward functions, Proposition[1](https://arxiv.org/html/2509.22047v2#Thmproposition1 "Proposition 1 (Affine Invariance of MO-GRPO Advantage). ‣ Invariance to positive affine transformation. ‣ 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows that it does not need to normalize the absolute scale of the reward functions. This makes MO-GRPO a practically useful algorithm for real-world problems where the scale of the reward models is unclear (e.g., neural models) or instance-dependent. Conversely, this property does not hold with GRPO.

###### Proposition 2.

The preference ordering induced by GRPO (and Dr .GRPO) is not invariant under positive affine transformations.

The proof is in Appendix[E.4](https://arxiv.org/html/2509.22047v2#A5.SS4 "E.4 Proof of Proposition 2 ‣ Appendix E Proof of Theorem and Proposition ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems").

#### Summary.

Unlike GRPO (Theorem[1](https://arxiv.org/html/2509.22047v2#Thmtheorem1 "Theorem 1 (Correlation between reward function and advantage function with GRPO). ‣ 3 Group Relative Policy Optimization (GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")), the advantage function of MO-GRPO is built keeping the correlation with each reward function constant (Theorem[2](https://arxiv.org/html/2509.22047v2#Thmtheorem2 "Theorem 2 (Correlation between a reward function and advantage function with MO-GRPO). ‣ 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")). In addition, it is invariant with the rescaling of the reward functions with positive affine transformation (Proposition[1](https://arxiv.org/html/2509.22047v2#Thmproposition1 "Proposition 1 (Affine Invariance of MO-GRPO Advantage). ‣ Invariance to positive affine transformation. ‣ 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")).

These properties make MO-GRPO an easy-to-use algorithm. It can use off-the-shelf reward functions without requiring manual tuning of the reward values to fit them to target tasks.

![Image 2: Refer to caption](https://arxiv.org/html/2509.22047v2/x2.png)

Figure 2: (Multi-armed bandit) This figure illustrates the average rewards obtained by the sum of the three reward functions: GRPO, MO-GRPO, and Dr. GRPO. As the figure shows, MO-GRPO finds a better policy faster than GRPO and Dr. GRPO.

![Image 3: Refer to caption](https://arxiv.org/html/2509.22047v2/x3.png)

![Image 4: Refer to caption](https://arxiv.org/html/2509.22047v2/x4.png)

![Image 5: Refer to caption](https://arxiv.org/html/2509.22047v2/x5.png)

Figure 3: (Multi-arm bandit) Comparison of the three reward functions with varying variances (10, 1, and 0.1) obtained by GRPO, Dr. GRPO, and MO-GRPO. While GRPO and Dr. GRPO fail or are slow to learn the reward functions with lower variances (R_{2} and R_{3}), MO-GRPO successfully optimizes all three reward functions regardless of the scale of the variances.

## 5 Experiment

We conduct experiments on four tasks: (1) multi-armed bandit, (2) simulated control, (3) machine translation, and (4) instruction following task. We compare three methods: GRPO, Dr. GRPO, and MO-GRPO.

### 5.1 Multi-Armed Bandit

We first conduct experiments on a simple multi-armed bandit environment to observe the behavior of GRPO and MO-GRPO in a controlled environment. We set the number of arms (actions) k to 50, and there are three stochastic reward functions R_{1},R_{2}, and R_{3}. The episode length is fixed to 5000 steps. The expected return of the arm \mu_{k} is chosen at random from a normal distribution of \mathcal{N}(0,1) at the beginning of the episode and is fixed throughout the episode. The three reward functions output the reward value of \mu_{k} plus additional stochastic noise to it as follows:

\displaystyle R_{1}(k)\displaystyle\sim\mathcal{N}(\mu_{k},10^{2})
\displaystyle R_{2}(k)\displaystyle\sim\mathcal{N}(\mu_{k},1^{2})-0.1R_{1}(k)
\displaystyle R_{3}(k)\displaystyle\sim\mathcal{N}(\mu_{k},0.1^{2})

R_{1}(k) outputs a value sampled from a normal distribution of mean \mu_{k} and standard deviation of 10. R_{2}(k) is a sum of a value sampled from a normal distribution of mean \mu_{k} and standard deviation of 1, minus the value of 0.1\times R_{1}. Thus, R_{2} is negatively correlated with R_{1}, making it harder to learn. R_{3} is a value sampled from a normal distribution of mean \mu and standard deviation of 0.1. These reward functions are designed to create a challenging optimization landscape where a high-variance reward function (R_{1}) could dominate the learning signal, potentially overshadowing the gradients from lower-variance and/or negatively correlated reward functions (R_{2} and R_{3}).

We set the number of actions GRPO, Dr. GRPO, and MO-GRPO samples G to 8, and we use a neural network with 3 hidden layers for policy. We conduct experiments with five different random seeds.

Figure[2](https://arxiv.org/html/2509.22047v2#S4.F2 "Figure 2 ‣ Summary. ‣ 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows the comparison of the three algorithms on the sum of the three reward functions. The result shows that MO-GRPO achieves a better policy faster than the others. Figure[3](https://arxiv.org/html/2509.22047v2#S4.F3 "Figure 3 ‣ Summary. ‣ 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows the breakdown of the three reward functions, showing that GRPO and Dr. GRPO fail to learn low variance reward functions (R_{2} and R_{3}), resulting in suboptimal policies.

The experimental result on a multi-armed bandit problem shows that MO-GRPO is a promising approach for tasks where the reward functions have different scales of variance.

### 5.2 Simulated Control

![Image 6: Refer to caption](https://arxiv.org/html/2509.22047v2/x6.png)

Figure 4: Simulated control task we use for the experiment. Two‑joint arms with a 6-state vector (\sin, \cos of joint angles and their angular velocities) select among 9 discrete actions to reach four targets within a 50‑step episode. Each reward function is defined as R_{i}=1-4\lVert p_{\text{arm}}-p_{\text{target},i}\rVert_{2}^{2}. The optimal control in this environment is to keep swinging the arm at a constant speed.

To evaluate the method on a simulated control task, we utilize the MO-Reacher (mo-reacher-v5) control benchmark from the mo_gymnasium framework Felten et al.([2023](https://arxiv.org/html/2509.22047v2#bib.bib59 "A toolkit for reliable benchmarking and research in multi-objective reinforcement learning")), as in Figure[4](https://arxiv.org/html/2509.22047v2#S5.F4 "Figure 4 ‣ 5.2 Simulated Control ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). In this task, a policy controls a two-joint robotic arm to simultaneously reach four distinct targets within 50 steps, which is an episode. The state is represented by a 6-dimensional vector composed of the \sin and \cos of the two joint angles and their angular velocities, while the policy selects from a discrete action space of nine. The objective of the task is to maximize the sum of the 4 reward functions, where each component R_{i} is a function of the squared Euclidean distance from the arm position to the target position: R_{i}=1-4\cdot\left\|p_{\text{arm }}-p_{\text{target },i}\right\|_{2}^{2}. We conduct this experiment with five different random seeds.

Table[2](https://arxiv.org/html/2509.22047v2#S5.T2 "Table 2 ‣ 5.2 Simulated Control ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows the average total reward obtained per step. As can be seen, MO-GRPO can obtain a higher reward value per step than GRPO, demonstrating its effectiveness, even when applied to conventional reinforcement learning tasks. For a supplementary explanation, both methods show large values for R_{1} and R_{3} because GRPO and MO-GRPO learned policies that emphasized rewards for R_{1} and R_{3} once each, while producing very negative values for R_{2} and R_{4} out of five seed iterations. This can also be confirmed by looking at the standard deviations of R_{2} and R_{4}.

Method Total Reward\uparrow Each Reward\uparrow
R_{1}R_{2}R_{3}R_{4}
GRPO 1.29\!\pm\!0.24 0.39\!\pm\!0.11 0.24\!\pm\!0.15 0.44\!\pm\!0.08 0.20\!\pm\!0.17
Dr. GRPO 1.10\!\pm\!0.20 0.33\!\pm\!0.06 0.22\!\pm\!0.05 0.32\!\pm\!0.07 0.22\!\pm\!0.06
MO-GRPO\mathbf{1.48\!\pm\!0.26}\mathbf{0.45\!\pm\!0.04}\mathbf{0.29\!\pm\!0.18}\mathbf{0.46\!\pm\!0.08}\mathbf{0.28\!\pm\!0.18}

Table 2: (Simulated control) MO‑Reacher results. The total reward shows the sum of the four reward values (R_{1}, R_{2}, R_{3}, and R_{4}) per step. The optimal policy achieves the total reward of around 1.76. 

![Image 7: Refer to caption](https://arxiv.org/html/2509.22047v2/x7.png)

Figure 5: (Machine translation) The training process of GRPO on the WMT En-Ja dataset uses BLEURT and jReadability as the reward functions by Sarashina (sarashina2.2-3b-instruct-v0.1). As the results show, GRPO overfits jReadability at the expense of BLEURT performance. As the results show, the standard deviation of jReadability is always more than BLEURT. As shown in Appendix[K](https://arxiv.org/html/2509.22047v2#A11 "Appendix K Training Process of GRPO and MO-GRPO with BLEURT and jReadability ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), the same phenomenon is observed in other LLMs. 

In the ablation study, we modified the environment by setting the standard deviation of R_{1} to 2. This verifies how MO-RPO behaves when encountering reward functions with high variance.

Method Total Reward\uparrow Each Reward\uparrow
R_{1}R_{2}R_{3}R_{4}
GRPO 1.14\!\pm\!0.21\mathbf{0.37\!\pm\!0.05}0.21\!\pm\!0.06 0.35\!\pm\!0.10 0.21\!\pm\!0.05
Dr. GRPO 1.10\!\pm\!0.16 0.35\!\pm\!0.02 0.20\!\pm\!0.07 0.34\!\pm\!0.09 0.21\!\pm\!0.04
MO-GRPO\mathbf{1.40\!\pm\!0.25}0.36\!\pm\!0.09\mathbf{0.33\!\pm\!0.05}\mathbf{0.40\!\pm\!0.05}\mathbf{0.30\!\pm\!0.07}

Table 3: (Simulated control) MO‑Reacher by setting the standard deviation of only the first reward function R_{1} to 2 results. The total reward shows the sum of the four reward values (R_{1}, R_{2}, R_{3}, and R_{4}) per step. The optimal policy achieves the total reward of around 1.76. 

Table[3](https://arxiv.org/html/2509.22047v2#S5.T3 "Table 3 ‣ 5.2 Simulated Control ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows the average reward values for each reward function per step with the standard deviation of R_{1}, set to 2. MO-GRPO has lower values than GRPO for R_{1} (high-standard-deviation reward function), but improves upon GRPO for the others. This suggests that, unlike GRPO, MO-GRPO can learn stably even when the standard deviations of each reward function diverge. It can be seen that Dr. GRPO exhibits behavior similar to that of GRPO in this setting. The behavior of GRPO and MO-GRPO is shown in Appendix[I](https://arxiv.org/html/2509.22047v2#A9 "Appendix I Supplementry Results of mo-reacher-v5 ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems").

WMT24 (En–Ja)
Base Model Method BLEURT†\uparrow jReadability†\uparrow GPT‑Eval⋆\uparrow
Sarashina Base Model 0.66 0.70 50.0%
GRPO 0.62 0.86 66.0%
Dr. GRPO 0.67 0.73 68.4%
MO-GRPO (ours)0.69 0.76 76.8%
Qwen Base Model 0.67 0.66 50.0%
GRPO 0.65 0.74 53.0%
Dr. GRPO 0.67 0.66 69.6%
MO-GRPO (ours)0.67 0.67 88.8%
Llama Base Model 0.65 0.67 50.0%
GRPO 0.60 0.90 35.6%
Dr. GRPO 0.63 0.77 42.6%
MO-GRPO (ours)0.66 0.69 68.8%
WMT24 (En–Zh)
Base Model Method BLEURT†\uparrow TRank†\downarrow GPT‑Eval⋆\uparrow
Qwen Base Model 0.73-2.73 50.0%
GRPO 0.71-3.20 53.9%
Dr. GRPO 0.73-2.98 62.4%
MO-GRPO (ours)0.73-2.85 68.1%
Llama Base Model 0.69-2.44 50.0%
GRPO 0.62-2.98 33.2%
Dr. GRPO 0.60-3.15 28.3%
MO-GRPO (ours)0.71-2.55 71.7%

Table 4: (Machine translation) Translation quality on WMT24 (higher is better). †BLEURT, jReadability, and TRank are training objectives and thus susceptible to over‑fitting. ⋆GPT‑Eval (against Base Model) is not optimized during training; we therefore regard it as the _primary metric_. Across all three base models, our MO-GRPO improves GPT‑Eval while avoiding excessive optimization of the training‑objective metrics.

### 5.3 Machine Translation

We evaluate the performance of MO-GRPO on machine translation with two objective functions, translation accuracy and readability. Readability is one of the important objectives in real-world text generation tasks Hasebe and Lee ([2015](https://arxiv.org/html/2509.22047v2#bib.bib24 "Introducing a readability evaluation system for Japanese language education")); Trokhymovych et al.([2024](https://arxiv.org/html/2509.22047v2#bib.bib77 "An Open Multilingual System for Scoring Readability of Wikipedia")). It measures the accessibility of the text for a diverse audience, including children and non-native speakers, and is critical for communicating vital information during emergencies such as natural disasters.

We conduct experiments on English to Japanese (En-Ja) and English to Chinese (En-Zh). We use the WMT-21, WMT-22, and WMT-23 datasets for training (Akhbardeh et al., [2021](https://arxiv.org/html/2509.22047v2#bib.bib60 "Findings of the 2021 Conference on Machine Translation (WMT21)"); Freitag et al., [2022](https://arxiv.org/html/2509.22047v2#bib.bib27 "Results of WMT22 metrics shared task: stop using BLEU – neural metrics are better and more robust"), [2023](https://arxiv.org/html/2509.22047v2#bib.bib26 "Results of WMT23 metrics shared task: metrics might be guilty but references are not innocent")), and evaluate on the WMT-24 test set (Kocmi et al., [2024](https://arxiv.org/html/2509.22047v2#bib.bib25 "Findings of the WMT24 general machine translation shared task: the LLM era is here but MT is not solved yet")).

First, we perform the En-Ja translation task in WMT datasets using Sarashina (sarashina2.2-3b-instruct-v0.1), Qwen (Qwen2.5-3B-Instruct) Yang et al.([2025](https://arxiv.org/html/2509.22047v2#bib.bib64 "Qwen3 technical report")), and Llama (Llama-3.2-3B-Instruct) Grattafiori et al.([2024](https://arxiv.org/html/2509.22047v2#bib.bib65 "The Llama 3 Herd of Models")) as the base models. For the reward (objective) functions, we adopt (i) BLEURT(Sellam et al., [2020](https://arxiv.org/html/2509.22047v2#bib.bib61 "BLEURT: Learning Robust Metrics for Text Generation")) and (ii) jReadability(Hasebe and Lee, [2015](https://arxiv.org/html/2509.22047v2#bib.bib24 "Introducing a readability evaluation system for Japanese language education")) to measure readability in Japanese. To evaluate the overall generation quality, we use LLM-as-a-Judge Zheng et al.([2023](https://arxiv.org/html/2509.22047v2#bib.bib72 "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena")) with GPT-4o-mini (GPT-Eval) so that both the translation accuracy and readability are considered.

Table[4](https://arxiv.org/html/2509.22047v2#S5.T4 "Table 4 ‣ 5.2 Simulated Control ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows that, compared to the base model score, GRPO achieved a high jReadability score but at the cost of degrading the BLEURT score. This result leads to the worst win rate score against the base model in three methods. In contrast, MO-GRPO almost successfully improved both metrics compared to the base model’s score, achieving in BLEURT and jReadability scores, preventing overfitting to jReadability, and MO-GRPO also achieves the highest win rate score. For supplementary, Dr. GRPO also shows higher values for both metrics compared to the base model, but not as high as MO-GRPO with respect to GPT-Eval (Table[11](https://arxiv.org/html/2509.22047v2#A6.T11 "Table 11 ‣ Appendix F Experiment Settings in WMT ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")).1 1 1 Dr. GRPO in this experiment is implemented using trl=0.16.1. This version of Dr. GRPO has no exclusions regarding sentence length normalization, only in the form of removing the standard deviation of the advantage function.

In detail, MO-GRPO with Sarashina achieves BLEURT score of 0.69. For comparison, when Sarashina is trained with GRPO solely on BLEURT, the score reached 0.70. This close score suggests that MO-GRPO effectively learns to optimize BLEURT without sacrificing other objectives. Furthermore, as shown in the output examples (Table[10](https://arxiv.org/html/2509.22047v2#A4.T10 "Table 10 ‣ Appendix D Reward hacking Examples ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")), GRPO exhibits behavior analogous to reward hacking on jReadability; on the other hand, it is not observed with MO-GRPO, and the training process of MO-GRPO with Sarashina is shown in Figure[6](https://arxiv.org/html/2509.22047v2#S5.F6 "Figure 6 ‣ 5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), which suggests MO-GRPO avoids overfitting of jReadability and prevents degradation of BLEURT, unlike GRPO (Figure[5](https://arxiv.org/html/2509.22047v2#S5.F5 "Figure 5 ‣ 5.2 Simulated Control ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")).

![Image 8: Refer to caption](https://arxiv.org/html/2509.22047v2/x8.png)

Figure 6: (Machine translation) The training process of MO-GRPO on the WMT En-Ja dataset uses BLEURT and jReadability as the reward functions by Sarashina. Unlike GRPO (Figure[5](https://arxiv.org/html/2509.22047v2#S5.F5 "Figure 5 ‣ 5.2 Simulated Control ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")), the figure shows MO-GRPO avoids overfitting of jReadability and prevents deterioration of BLEURT. We also see stability in the results in other LLMs (Appendix[K](https://arxiv.org/html/2509.22047v2#A11 "Appendix K Training Process of GRPO and MO-GRPO with BLEURT and jReadability ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")).

Next, we examine the details of the results from other language models such as Qwen and Llama. The relatively limited improvement of MO-GRPO in Qwen’s experiments is likely attributable to Qwen not being a language model specialized for Japanese. However, Qwen outputs (Table[9](https://arxiv.org/html/2509.22047v2#A4.T9 "Table 9 ‣ Appendix D Reward hacking Examples ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")) are also confirmed reward hacking behavior from GRPO (such behavior is also not observed with MO-GRPO). Llama outputs (Table[1](https://arxiv.org/html/2509.22047v2#S0.T1 "Table 1 ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")) show that GRPO engaged in reward hacking by outputting English text instead of a Japanese translation, thereby improving the jReadability. This phenomenon again did not occur with MO-GRPO. As discussed in Appendix[K](https://arxiv.org/html/2509.22047v2#A11 "Appendix K Training Process of GRPO and MO-GRPO with BLEURT and jReadability ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), GRPO with Llama exhibits the strongest overfitting to jReadability, which can be explained by the higher variance of this metric under Llama.

Second, we conduct En-Zh translation task in WMT datasets using Qwen and Llama as base models. For the reward functions, we adopt (i) BLEURT and (ii) TRank(Trokhymovych et al., [2024](https://arxiv.org/html/2509.22047v2#bib.bib77 "An Open Multilingual System for Scoring Readability of Wikipedia")), which can evaluate the readability of text across multiple languages, including Chinese. Since TRank scores are higher for more difficult texts, multiply the score by -1 in this experiment setting during the training (Table[4](https://arxiv.org/html/2509.22047v2#S5.T4 "Table 4 ‣ 5.2 Simulated Control ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows the true TRank score, i.e., the score obtained without applying the -1 multiplication during evaluation.).

As shown in Table[4](https://arxiv.org/html/2509.22047v2#S5.T4 "Table 4 ‣ 5.2 Simulated Control ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), MO-GRPO also appropriately treats the two reward functions across Qwen and Llama, similar to the En-Ja task. Therefore, it consistently achieves higher GPT-Eval win rates than GRPO and Dr. GRPO. GPT-Eval indicates that GRPO and Dr. GRPO are improving with Qwen but exhibit clear reward hacking with Llama. Both methods overfit to TRank at the expense of BLEURT. Interestingly, the trained models output non-Chinese text even though the task is Chinese translation (similarly to what we observed in En-Ja translation task in Table[1](https://arxiv.org/html/2509.22047v2#S0.T1 "Table 1 ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")).

Table[5](https://arxiv.org/html/2509.22047v2#S5.T5 "Table 5 ‣ 5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") further quantifies this phenomenon: using langdetect,2 2 2 https://pypi.org/project/langdetect/. We find that GRPO and Dr. GRPO frequently produce non-Chinese generations, exploiting TRank and this overly yields lower GPT-Eval scores than the base model.

Method Non-Chinese (%)(w/o Penalty) \downarrow Non-Chinese (%)(w/ Penalty) \downarrow Base Model (Llama)14.7%14.7%GRPO 68.7%1.2%Dr. GRPO 70.7%1.2%MO-GRPO 5.6%0.6%

Table 5: The probability of non-Chinese outputs (Non-Chinese) in machine translation. The reference set contains 851 Chinese sentences (by langdetect). Non-Chinese=1-\#\text{Chinese}/851. w/o Penalty: TRank only. w/ Penalty: TRank with penalty for non-Chinese outputs. MO-GRPO consistently maintains proper outputs under both settings.

#### Ablation study.

Instead of using MO-GRPO, one may solve the reward hacking by patching the reward function so that it cannot be hacked. We improve the TRank reward function by giving a huge penalty (score=10) if the generated output is identified as non-Chinese by langdetect. In this way, we can prevent the model to learn to generate non-Chinese texts. Table[5](https://arxiv.org/html/2509.22047v2#S5.T5 "Table 5 ‣ 5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows that adding a penalty to TRank reduces the probability of non-Chinese outputs (Non-Chinese) for all methods; MO-GRPO still has the lowest probability. Additionally, Table[6](https://arxiv.org/html/2509.22047v2#S5.T6 "Table 6 ‣ Ablation study. ‣ 5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows the TRank with penalties under the same experimental settings as Table[4](https://arxiv.org/html/2509.22047v2#S5.T4 "Table 4 ‣ 5.2 Simulated Control ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). This shows that MO-GRPO outperforms other methods in terms of GPT-Eval, and in this setting as well, GRPO and Dr. GRPO are trained with a focus on TRank penalties, which fluctuate more than BLEURT. This shows that MO-GRPO is on par with GRPO even if the reward functions are reasonably designed (e.g., problem settings in which GRPO achieves improvement).

Method BLEURT\uparrow TRank w/ Penalty\downarrow GPT-Eval\uparrow Base Model 0.69 1.39 50%GRPO 0.70-0.66 71.5%Dr. GRPO 0.70-0.64 69.6%MO-GRPO 0.71-0.47 74.0%

Table 6: (Machine translation) Translation quality on WMT24 En-Zh (higher is better) with Llama. TRank w/ Penalty penalty non-Chinese outputs. MO-GRPO improves GPT‑Eval while avoiding excessive optimization of the training‑objective metrics.

### 5.4 Instruction Following Task

In this section, we conduct an experiment in AlpacaFarm (training dataset: tatsu-lab/alpaca, eval dataset: tatsu-lab/alpaca_eval) Dubois et al.([2023](https://arxiv.org/html/2509.22047v2#bib.bib73 "AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback")) to evaluate the performance of MO-GRPO for the generic instruction following task using Qwen and Llama as base models. We use RM-Mistral-7B Dong et al.([2023](https://arxiv.org/html/2509.22047v2#bib.bib74 "RAFT: Reward ranked Finetuning for Generative Foundation Model Alignment")) and the Length reward function. The Length reward function (R_{\text{Len}}) gives a higher reward on the outputs closer to the length of the reference text so that it mitigates the length bias problem Shen et al.([2023](https://arxiv.org/html/2509.22047v2#bib.bib76 "Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback")); Singhal et al.([2024](https://arxiv.org/html/2509.22047v2#bib.bib75 "A Long Way to Go: Investigating Length Correlations in RLHF")). Length reward function is defined as follows:

R_{\text{Len}}=\begin{aligned} &\begin{cases}\dfrac{L}{0.9L_{\text{ref}}},&L<0.9L_{\text{ref}},\\
1,&0.9L_{\text{ref}}\leq L\leq 1.1L_{\text{ref}},\\
\dfrac{1.1L_{\text{ref}}}{L},&L>1.1L_{\text{ref}}.\end{cases}\end{aligned}

where L_{\text{ref}} is the reference text length, L is the output text length. Given that RM-Mistral-7B tends to prefer longer outputs Shen et al.([2023](https://arxiv.org/html/2509.22047v2#bib.bib76 "Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback")); Singhal et al.([2024](https://arxiv.org/html/2509.22047v2#bib.bib75 "A Long Way to Go: Investigating Length Correlations in RLHF")), the Length reward function is an adversarial objective to it, making the optimization more challenging.

AlpacaFarm Base Model Method RM-Mistral-7B\uparrow Length\uparrow Qwen Base Model 5.55 0.42 GRPO 5.81 0.36 Dr. GRPO 6.24 0.34 MO-GRPO (ours)5.51 0.44 Llama Base Model 5.26 0.42 GRPO 5.56 0.37 Dr. GRPO 5.90 0.34 MO-GRPO (ours)5.28 0.42

Table 7: (AlpacaFarm) Since RM-Mistral and Length have conflicting objectives, the correct answer here is to prevent it from being derived from the base model. GRPO and Dr. GRPO have learned to prioritize RM-Mistral, resulting in a significant sacrifice of Length, but MO-GRPO retains both reward functions almost entirely.

Table[7](https://arxiv.org/html/2509.22047v2#S5.T7 "Table 7 ‣ 5.4 Instruction Following Task ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows that GRPO and Dr. GRPO optimize RM-Mistral while decreasing the Length of both Llama and Qwen. In contrast, MO-GRPO attempts to maintain the values of both rewards. In such adversarial cases where both reward functions are important, the optimal behavior is to remain close to the base model, such as MO-GRPO.

## 6 Conclusion

We conducted an investigation into the theoretical and empirical properties of handling multiple reward functions with GRPO. Our analysis revealed a previously unreported vulnerability. The advantage function of GRPO is biased toward reward functions with high variance. This makes the algorithm susceptible to reward-hacking behaviors in multi-objective settings. To address this weakness, we proposed Multi-Objective GRPO (MO-GRPO), an extension of GRPO that uses a simple normalization method to automatically reweight reward functions according to their value variances. MO-GRPO treats each reward function value equitably while preserving preference orderings under rescalings. Comprehensive experiments confirmed the practical benefits of this mechanism. We experimentally evaluate MO-GRPO in four domains: (i) the multi-armed bandits problem, (ii) the simulated control task (Mo-Gymnasium), (iii) machine translation tasks on the WMT benchmark (En-Ja, En-Zh), and (iv) instruction following task. MO-GRPO consistently avoids reward hacking and shows improvements in task-specific metrics (e.g., BLEURT, jReadability) and learning stability.

## References

*   F. Akhbardeh, A. Arkhangorodsky, M. Biesialska, O. Bojar, R. Chatterjee, V. Chaudhary, M. R. Costa-jussa, C. España-Bonet, A. Fan, C. Federmann, M. Freitag, Y. Graham, R. Grundkiewicz, B. Haddow, L. Harter, K. Heafield, C. Homan, M. Huck, K. Amponsah-Kaakyire, J. Kasai, D. Khashabi, K. Knight, T. Kocmi, P. Koehn, N. Lourie, C. Monz, M. Morishita, M. Nagata, A. Nagesh, T. Nakazawa, M. Negri, S. Pal, A. A. Tapo, M. Turchi, V. Vydrin, and M. Zampieri (2021)Findings of the 2021 Conference on Machine Translation (WMT21). In Proceedings of the Sixth Conference on Machine Translation, L. Barrault, O. Bojar, F. Bougares, R. Chatterjee, M. R. Costa-jussa, C. Federmann, M. Fishel, A. Fraser, M. Freitag, Y. Graham, R. Grundkiewicz, P. Guzman, B. Haddow, M. Huck, A. J. Yepes, P. Koehn, T. Kocmi, A. Martins, M. Morishita, and C. Monz (Eds.), Online,  pp.1–88. External Links: [Link](https://aclanthology.org/2021.wmt-1.1/)Cited by: [§5.3](https://arxiv.org/html/2509.22047v2#S5.SS3.p2.1 "5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   Concrete Problems in AI Safety. arXiv preprint arXiv:1606.06565. Cited by: [§2](https://arxiv.org/html/2509.22047v2#S2.p1.1 "2 Related Works ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   H. Dong, W. Xiong, D. Goyal, Y. Zhang, W. Chow, R. Pan, S. Diao, J. Zhang, K. SHUM, and T. Zhang (2023)RAFT: Reward ranked Finetuning for Generative Foundation Model Alignment. Transactions on Machine Learning Research. Note: External Links: ISSN 2835-8856, [Link](https://openreview.net/forum?id=m7p5O7zblY)Cited by: [Table 13](https://arxiv.org/html/2509.22047v2#A7.T13.2.9.1 "In Appendix G Reproducibility Statement ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§5.4](https://arxiv.org/html/2509.22047v2#S5.SS4.p1.1 "5.4 Instruction Following Task ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   Y. Dubois, C. X. Li, R. Taori, T. Zhang, I. Gulrajani, J. Ba, C. Guestrin, P. S. Liang, and T. B. Hashimoto (2023)AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback. In Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), Vol. 36,  pp.30039–30069. External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2023/file/5fc47800ee5b30b8777fdd30abcaaf3b-Paper-Conference.pdf)Cited by: [Table 13](https://arxiv.org/html/2509.22047v2#A7.T13.2.8.1 "In Appendix G Reproducibility Statement ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§5.4](https://arxiv.org/html/2509.22047v2#S5.SS4.p1.1 "5.4 Instruction Following Task ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   F. Felten, L. N. Alegre, A. Nowe, A. Bazzan, E. G. Talbi, G. Danoy, and B. C. da Silva (2023)A toolkit for reliable benchmarking and research in multi-objective reinforcement learning. In Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), Vol. 36,  pp.23671–23700. External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2023/file/4aa8891583f07ae200ba07843954caeb-Paper-Datasets_and_Benchmarks.pdf)Cited by: [§1](https://arxiv.org/html/2509.22047v2#S1.p5.1 "1 Introduction ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§5.2](https://arxiv.org/html/2509.22047v2#S5.SS2.p1.4 "5.2 Simulated Control ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   X. Feng, B. Han, Z. Zhou, J. Fan, J. Yao, K. H. Li, D. Yu, and M. Ng (2025)DyPO: Dynamic Policy Optimization for Multi-Turn Interactive Reasoning. In ICML 2025 Workshop on Programmatic Representations for Agent Learning, External Links: [Link](https://openreview.net/forum?id=OWDBiMKYdo)Cited by: [Appendix A](https://arxiv.org/html/2509.22047v2#A1.p1.1 "Appendix A Related Work: Variants of GRPO ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   M. Freitag, N. Mathur, C. Lo, E. Avramidis, R. Rei, B. Thompson, T. Kocmi, F. Blain, D. Deutsch, C. Stewart, C. Zerva, S. Castilho, A. Lavie, and G. Foster (2023)Results of WMT23 metrics shared task: metrics might be guilty but references are not innocent. In Proceedings of the Eighth Conference on Machine Translation, P. Koehn, B. Haddow, T. Kocmi, and C. Monz (Eds.), Singapore,  pp.578–628. External Links: [Link](https://aclanthology.org/2023.wmt-1.51/), [Document](https://dx.doi.org/10.18653/v1/2023.wmt-1.51)Cited by: [§5.3](https://arxiv.org/html/2509.22047v2#S5.SS3.p2.1 "5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   M. Freitag, R. Rei, N. Mathur, C. Lo, C. Stewart, E. Avramidis, T. Kocmi, G. Foster, A. Lavie, and A. F. T. Martins (2022)Results of WMT22 metrics shared task: stop using BLEU – neural metrics are better and more robust. In Proceedings of the Seventh Conference on Machine Translation (WMT), P. Koehn, L. Barrault, O. Bojar, F. Bougares, R. Chatterjee, M. R. Costa-jussà, C. Federmann, M. Fishel, A. Fraser, M. Freitag, Y. Graham, R. Grundkiewicz, P. Guzman, B. Haddow, M. Huck, A. Jimeno Yepes, T. Kocmi, A. Martins, M. Morishita, C. Monz, M. Nagata, T. Nakazawa, M. Negri, A. Névéol, M. Neves, M. Popel, M. Turchi, and M. Zampieri (Eds.), Abu Dhabi, United Arab Emirates (Hybrid),  pp.46–68. External Links: [Link](https://aclanthology.org/2022.wmt-1.2/)Cited by: [§5.3](https://arxiv.org/html/2509.22047v2#S5.SS3.p2.1 "5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   L. Gao, J. Schulman, and J. Hilton (2023)Scaling Laws for Reward Model Overoptimization. In Proceedings of the 40th International Conference on Machine Learning, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett (Eds.), Proceedings of Machine Learning Research, Vol. 202,  pp.10835–10866. External Links: [Link](https://proceedings.mlr.press/v202/gao23h.html)Cited by: [§1](https://arxiv.org/html/2509.22047v2#S1.p1.1 "1 Introduction ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§2](https://arxiv.org/html/2509.22047v2#S2.p1.1 "2 Related Works ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   A. Gleave, M. Dennis, C. Wild, N. Kant, S. Levine, and S. Russell (2020)Adversarial policies: attacking deep reinforcement learning. In International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=HJgEMpVFwB)Cited by: [§2](https://arxiv.org/html/2509.22047v2#S2.p1.1 "2 Related Works ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. (2024)The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783. Cited by: [Table 13](https://arxiv.org/html/2509.22047v2#A7.T13.2.6.1 "In Appendix G Reproducibility Statement ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§5.3](https://arxiv.org/html/2509.22047v2#S5.SS3.p3.1 "5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   Y. Hasebe and J. Lee (2015)Introducing a readability evaluation system for Japanese language education. In Proceedings of the 6th international conference on computer assisted systems for teaching & learning Japanese,  pp.19–22. Cited by: [Table 13](https://arxiv.org/html/2509.22047v2#A7.T13.2.7.1 "In Appendix G Reproducibility Statement ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§5.3](https://arxiv.org/html/2509.22047v2#S5.SS3.p1.1 "5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§5.3](https://arxiv.org/html/2509.22047v2#S5.SS3.p3.1 "5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   S. Kim, D. Kang, T. Kwon, H. Chae, D. Lee, and J. Yeo (2025)Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar (Eds.), Vienna, Austria,  pp.13252–13280. External Links: [Link](https://aclanthology.org/2025.acl-long.649/), ISBN 979-8-89176-251-0 Cited by: [§2](https://arxiv.org/html/2509.22047v2#S2.p1.1 "2 Related Works ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   T. Kocmi, E. Avramidis, R. Bawden, O. Bojar, A. Dvorkovich, C. Federmann, M. Fishel, M. Freitag, T. Gowda, R. Grundkiewicz, B. Haddow, M. Karpinska, P. Koehn, B. Marie, C. Monz, K. Murray, M. Nagata, M. Popel, M. Popović, M. Shmatova, S. Steingrímsson, and V. Zouhar (2024)Findings of the WMT24 general machine translation shared task: the LLM era is here but MT is not solved yet. In Proceedings of the Ninth Conference on Machine Translation, B. Haddow, T. Kocmi, P. Koehn, and C. Monz (Eds.), Miami, Florida, USA,  pp.1–46. External Links: [Link](https://aclanthology.org/2024.wmt-1.1/), [Document](https://dx.doi.org/10.18653/v1/2024.wmt-1.1)Cited by: [Table 13](https://arxiv.org/html/2509.22047v2#A7.T13.2.2.1 "In Appendix G Reproducibility Statement ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§1](https://arxiv.org/html/2509.22047v2#S1.p5.1 "1 Introduction ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§5.3](https://arxiv.org/html/2509.22047v2#S5.SS3.p2.1 "5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   M. Li, Y. Zhang, W. Wang, W. Shi, Z. Liu, F. Feng, and T. Chua (2025)Self-Improvement Towards Pareto Optimality: Mitigating Preference Conflicts in Multi-Objective Alignment. In Findings of the Association for Computational Linguistics: ACL 2025, W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar (Eds.), Vienna, Austria,  pp.11010–11031. External Links: [Link](https://aclanthology.org/2025.findings-acl.574/), ISBN 979-8-89176-256-5 Cited by: [§2](https://arxiv.org/html/2509.22047v2#S2.p1.1 "2 Related Works ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. (2024)Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437. Cited by: [§1](https://arxiv.org/html/2509.22047v2#S1.p1.1 "1 Introduction ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, and M. Lin (2025)Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783. Cited by: [§1](https://arxiv.org/html/2509.22047v2#S1.p1.1 "1 Introduction ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   A. Pan, K. Bhatia, and J. Steinhardt (2022)The effects of reward misspecification: mapping and mitigating misaligned models. In International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=JYtwGwIL7ye)Cited by: [§2](https://arxiv.org/html/2509.22047v2#S2.p1.1 "2 Related Works ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   R. Rafailov, Y. Chittepu, R. Park, H. Sikchi, J. Hejna, W. B. Knox, C. Finn, and S. Niekum (2024)Scaling laws for reward model overoptimization in direct alignment algorithms. In Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (Eds.), Vol. 37,  pp.126207–126242. External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/e45caa3d5273d105b8d045e748636957-Paper-Conference.pdf)Cited by: [§1](https://arxiv.org/html/2509.22047v2#S1.p1.1 "1 Introduction ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§2](https://arxiv.org/html/2509.22047v2#S2.p1.1 "2 Related Works ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   A. Rastogi, A. Q. Jiang, A. Lo, G. Berrada, G. Lample, J. Rute, J. Barmentlo, K. Yadav, K. Khandelwal, K. R. Chandu, et al. (2025)Magistral. arXiv preprint arXiv:2506.10910. Cited by: [Appendix A](https://arxiv.org/html/2509.22047v2#A1.p1.1 "Appendix A Related Work: Variants of GRPO ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   T. Sellam, D. Das, and A. Parikh (2020)BLEURT: Learning Robust Metrics for Text Generation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault (Eds.), Online,  pp.7881–7892. External Links: [Link](https://aclanthology.org/2020.acl-main.704/), [Document](https://dx.doi.org/10.18653/v1/2020.acl-main.704)Cited by: [Table 13](https://arxiv.org/html/2509.22047v2#A7.T13.2.3.1 "In Appendix G Reproducibility Statement ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§5.3](https://arxiv.org/html/2509.22047v2#S5.SS3.p3.1 "5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. (2024)Deepseekmath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300. Cited by: [§1](https://arxiv.org/html/2509.22047v2#S1.p1.1 "1 Introduction ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§3](https://arxiv.org/html/2509.22047v2#S3.p1.11 "3 Group Relative Policy Optimization (GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   W. Shen, R. Zheng, W. Zhan, J. Zhao, S. Dou, T. Gui, Q. Zhang, and X. Huang (2023)Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback. In Findings of the Association for Computational Linguistics: EMNLP 2023,  pp.2859–2873. Cited by: [§5.4](https://arxiv.org/html/2509.22047v2#S5.SS4.p1.1 "5.4 Instruction Following Task ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§5.4](https://arxiv.org/html/2509.22047v2#S5.SS4.p1.3 "5.4 Instruction Following Task ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   P. Singhal, T. Goyal, J. Xu, and G. Durrett (2024)A Long Way to Go: Investigating Length Correlations in RLHF. In First Conference on Language Modeling, External Links: [Link](https://openreview.net/forum?id=G8LaO1P0xv)Cited by: [§5.4](https://arxiv.org/html/2509.22047v2#S5.SS4.p1.1 "5.4 Instruction Following Task ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§5.4](https://arxiv.org/html/2509.22047v2#S5.SS4.p1.3 "5.4 Instruction Following Task ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   J. Skalse, N. Howe, D. Krasheninnikov, and D. Krueger (2022)Defining and Characterizing Reward Gaming. In Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35,  pp.9460–9471. External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/3d719fee332caa23d5038b8a90e81796-Paper-Conference.pdf)Cited by: [§1](https://arxiv.org/html/2509.22047v2#S1.p1.1 "1 Introduction ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§2](https://arxiv.org/html/2509.22047v2#S2.p1.1 "2 Related Works ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   N. Stiennon, L. Ouyang, J. Wu, D. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. F. Christiano (2020)Learning to summarize with human feedback. In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33,  pp.3008–3021. External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf)Cited by: [§2](https://arxiv.org/html/2509.22047v2#S2.p1.1 "2 Related Works ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   M. Trokhymovych, I. Sen, and M. Gerlach (2024)An Open Multilingual System for Scoring Readability of Wikipedia. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L. Ku, A. Martins, and V. Srikumar (Eds.), Bangkok, Thailand,  pp.6296–6311. External Links: [Link](https://aclanthology.org/2024.acl-long.342/), [Document](https://dx.doi.org/10.18653/v1/2024.acl-long.342)Cited by: [Table 13](https://arxiv.org/html/2509.22047v2#A7.T13.2.10.1 "In Appendix G Reproducibility Statement ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§5.3](https://arxiv.org/html/2509.22047v2#S5.SS3.p1.1 "5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§5.3](https://arxiv.org/html/2509.22047v2#S5.SS3.p7.2 "5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   C. Xiao, M. Zhang, and Y. Cao (2025)BNPO: Beta Normalization Policy Optimization. arXiv preprint arXiv:2506.02864. Cited by: [Appendix A](https://arxiv.org/html/2509.22047v2#A1.p1.1 "Appendix A Related Work: Variants of GRPO ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. (2025)Qwen3 technical report. arXiv preprint arXiv:2505.09388. Cited by: [Table 13](https://arxiv.org/html/2509.22047v2#A7.T13.2.5.1 "In Appendix G Reproducibility Statement ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§1](https://arxiv.org/html/2509.22047v2#S1.p1.1 "1 Introduction ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), [§5.3](https://arxiv.org/html/2509.22047v2#S5.SS3.p3.1 "5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   C. Zheng, S. Liu, M. Li, X. Chen, B. Yu, C. Gao, K. Dang, Y. Liu, R. Men, A. Yang, J. Zhou, and J. Lin (2025)Group Sequence Policy Optimization. arXiv preprint arXiv:2507.18071. Cited by: [Appendix A](https://arxiv.org/html/2509.22047v2#A1.p1.1 "Appendix A Related Work: Variants of GRPO ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   L. Zheng, W. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica (2023)Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), Vol. 36,  pp.46595–46623. External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf)Cited by: [§5.3](https://arxiv.org/html/2509.22047v2#S5.SS3.p3.1 "5.3 Machine Translation ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   Z. Zhou, J. Liu, J. Shao, X. Yue, C. Yang, W. Ouyang, and Y. Qiao (2024)Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization. In Findings of the Association for Computational Linguistics: ACL 2024, L. Ku, A. Martins, and V. Srikumar (Eds.), Bangkok, Thailand,  pp.10586–10613. External Links: [Link](https://aclanthology.org/2024.findings-acl.630/), [Document](https://dx.doi.org/10.18653/v1/2024.findings-acl.630)Cited by: [§2](https://arxiv.org/html/2509.22047v2#S2.p1.1 "2 Related Works ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 
*   D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving (2020)Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593. Cited by: [§2](https://arxiv.org/html/2509.22047v2#S2.p1.1 "2 Related Works ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"). 

## Appendix A Related Work: Variants of GRPO

Several algorithms have been proposed to improve GRPO. Xiao et al.([2025](https://arxiv.org/html/2509.22047v2#bib.bib28 "BNPO: Beta Normalization Policy Optimization")) uses adaptive reward normalization with a Beta distribution to improve training stability and precision, outperforming REINFORCE and GRPO in reasoning tasks, and it dynamically normalizes rewards, enhancing policy optimization. In addition, LLMs have been developed that apply an improved method of GRPO that performs length normalization sequentially and normalizes the advantage function batch by batch Rastogi et al.([2025](https://arxiv.org/html/2509.22047v2#bib.bib58 "Magistral")). Dynamic Policy Optimization Feng et al.([2025](https://arxiv.org/html/2509.22047v2#bib.bib63 "DyPO: Dynamic Policy Optimization for Multi-Turn Interactive Reasoning")) is an extension of GRPO that enables large language models to perform adaptive, multi-turn reasoning in dynamic environments. In experiments, DyPO outperformed existing methods consistently in interactive decision-making and reasoning. Zheng et al.([2025](https://arxiv.org/html/2509.22047v2#bib.bib5 "Group Sequence Policy Optimization")) proposed the new GRPO method to use sequence likelihood for importance sampling.

Our contribution is distinct from these studies as we focus on low-resource settings where a reliable reward model is not available. MO-GRPO is orthogonal to these ideas and can be combined with these algorithms.

## Appendix B Formal Formulation of GRPO

The formal formulation of GRPO is as follows:

\displaystyle J_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{\,q,\{o_{g}\}\sim\pi_{\theta_{\mathrm{ref}}}}\Bigg[\frac{1}{G}\sum_{g=1}^{G}\frac{1}{|o_{g}|}(8)
\displaystyle\min\!\Bigg(\frac{\pi_{\theta}(o_{g}\mid q)}{\pi_{\theta_{\mathrm{old}}}(o_{g}\mid q)}\,A_{g},\,(9)
\displaystyle\operatorname{clip}\!\Big(1-\epsilon,\,1+\epsilon,\,\frac{\pi_{\theta}(o_{g}\mid q)}{\pi_{\theta_{\mathrm{old}}}(o_{g}\mid q)}\Big)\,A_{g}\Bigg)\Bigg](10)
\displaystyle\qquad\qquad-\beta\,\text{KL}\!\left(\pi_{\theta}\,\middle\|\,\pi_{\theta_{\mathrm{ref}}}\right).(11)

where \epsilon is a threshold parameter.

## Appendix C Correlation Analysis of Dr. GRPO

In Dr. GRPO, the advantage function is defined as:

A^{\text{Dr}}_{g}=\sum_{i=1}^{K}R_{i}(q,o_{g})\;-\;\mathrm{mean}_{\mathbf{o}}\bigl(\sum_{i=1}^{K}R_{i}(q,\mathbf{o})\bigr).(12)

###### Theorem 3(Correlation each reward function and advantage function with Dr. GRPO).

Assume that G\rightarrow\infty. The correlation coefficient between an individual reward function R_{i} and the advantage A is the ratio of the standard deviation of R_{i} to the standard deviation of the total reward.

\displaystyle\operatorname{Corr}(R_{i},A^{\text{Dr}}_{g})=(13)
\displaystyle\frac{\sigma_{i}^{2}+X}{\sqrt{\sigma_{i}^{2}\left(\sum\limits_{j=1}^{K}\sigma_{j}^{2}+\sum\limits_{j\neq l}\sum\limits_{l\neq j}\operatorname{Cov}\left(R_{j},R_{l}\right)\right)}}(14)

where X=\sum_{j\neq i}\operatorname{Cov}(R_{i},R_{j}).

###### Proof.

From here on, for simplicity, we omit the notation for prompt q and optional output o_{g} (e.g., R_{i}(q,o_{g})\rightarrow R_{i}). We assume the number of samples G\to\infty, which allows the sample statistics to approximate the true population parameters. Let R_{1},\dots,R_{i} be K reward functions. We assume they are uncorrelated, such that \operatorname{Cov}[R_{i},R_{j}]=0 for all i\neq j. Let \mu_{i}=\mathbb{E}[R_{i}] and \sigma_{i}^{2}=\operatorname{Var}[R_{i}] denote the mean and variance of the i-th reward, respectively.

A^{\text{Dr}}\;:=\;{\mathbf{R}-\mu},\qquad\mu=\mathbb{E}[\mathbf{R}].(15)

\displaystyle\operatorname{Cov}(R_{i},A^{\text{Dr}})=\operatorname{Cov}\left(R_{i},\mathbf{R}-\mathbb{E}[\mathbf{R}]\right)(16)
\displaystyle=\operatorname{Cov}(R_{i},\mathbf{R})(17)
\displaystyle=\operatorname{Cov}\left(R_{i},\sum_{j=1}^{K}R_{j}\right)(18)
\displaystyle=\sum_{j=1}^{K}\operatorname{Cov}(R_{i},R_{j})(19)
\displaystyle=\left(\operatorname{Cov}(R_{i},R_{i})+\underbrace{\sum_{j\neq i}\operatorname{Cov}(R_{i},R_{j})}_{X}\right)(20)
\displaystyle=(\operatorname{Var}[R_{i}]+X)={\sigma_{i}^{2}}+X(21)

The correlation coefficient between the i-th reward and the advantage is:

\displaystyle\operatorname{Corr}(R_{i},A^{\text{Dr}})\displaystyle=\frac{\operatorname{Cov}\left(R_{i},A^{\text{Dr}}\right)}{\sqrt{\operatorname{Var}\left[R_{i}\right]\operatorname{Var}\left[A^{\text{Dr}}\right]}}(22)

Finally, we can get the correlation between each reward function and advantage function with Dr. GRPO.

\displaystyle\operatorname{Corr}(R_{i},A^{\text{Dr}})=\frac{\operatorname{Cov}(R_{i},A^{\text{Dr}})}{\sqrt{\operatorname{Var}[R_{i}])\operatorname{Var}[A^{\text{Dr}}]}}(23)
\displaystyle=\frac{\sigma_{i}^{2}+X}{\sqrt{\sigma_{i}^{2}\cdot\operatorname{Var}[A^{\text{Dr}}]}}(24)
\displaystyle=\frac{\sigma_{i}^{2}+X}{\sqrt{\sigma_{i}^{2}\left(\sum\limits_{j=1}^{K}\sigma_{j}^{2}+\sum\limits_{j\neq l}\sum\limits_{l\neq j}\operatorname{Cov}\left(R_{j},R_{l}\right)\right)}}(25)

∎

If there is no correlation between rewards, the following applies:

\displaystyle\operatorname{Corr}(R_{i},A^{\text{Dr}})\displaystyle=\frac{\sigma_{i}}{\sigma}(26)

In other words, Dr. GRPO implies that learning is biased toward rewards with large variances.

## Appendix D Reward hacking Examples

We show the results of the outputs during training (Table[8](https://arxiv.org/html/2509.22047v2#A4.T8 "Table 8 ‣ Appendix D Reward hacking Examples ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")), and the other LLMs cause the reward hacking behavior (Table[9](https://arxiv.org/html/2509.22047v2#A4.T9 "Table 9 ‣ Appendix D Reward hacking Examples ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") and Table[10](https://arxiv.org/html/2509.22047v2#A4.T10 "Table 10 ‣ Appendix D Reward hacking Examples ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")). In Table[9](https://arxiv.org/html/2509.22047v2#A4.T9 "Table 9 ‣ Appendix D Reward hacking Examples ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") and Table[10](https://arxiv.org/html/2509.22047v2#A4.T10 "Table 10 ‣ Appendix D Reward hacking Examples ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), it can be seen that both Qwen and Sarashina use more non-Japanese languages in GRPO to increase the jReadabillity score (reward hacking).

Instruction Translate the following English into easily readable Japanese.\nOver the past decade, our lives have changed through technology, with many working from home, …jReadability \uparrow BLEURT \uparrow
GRPO (\frac{1}{10}T)過去の10年で、技術の進化により、多くの人が家で働いており …0.33 0.69
GRPO (\frac{1}{3}T)Over the past ten years, our lives have changed a lot because of technology. …\mathbf{0.99}0.57
GRPO (T)Over the past ten years, our lives have changed a lot because of technology. …\mathbf{0.94}0.57
MO-GRPO (T)過去の10年で、技術の進化により、多くの人がホームワークから仕事をしているようになり…0.40 0.69

Table 8: (Machine translation) Generation examples of GRPO and MO-GRPO by Llama (Llama-3.2-3B-Instruct). T is the total steps. GRPO optimizes only the Japanese readability score (jReadability) by avoiding using difficult Japanese words, eventually stops using any Japanese characters, ignoring the translation accuracy score (BLEURT), resulting in generating non-Japanese text, which defeats the purpose of the translation. On the other hand, MO-GRPO evenly optimizes both objectives, achieving improvement on both objectives as intended. 

Method Output jReadability\uparrow BLEURT\uparrow
Input"People Swimming in the Swimming Pool" from 2022 is one Vicente Siso artwork that will display at Tierra del Sol Gallery beginning Jan. 13. (photo courtesy of Vicente Siso)––
GRPO"People Swimming in the Swimming Pool" 2022年はビクセン・シソオーワークスがティラードールギャラリーで1月13日から展示します。（ビクセン・シソオーフォト提供）1.0 0.68
MO-GRPO 2022年の「泳ぎの人たち」は、1月13日からティラードールギャラリーでVICENTE SISOの作品が展示されます。（VICENTE SISOの写真提供)0.86 0.77

Table 9: Generation examples of GRPO and MO-GRPO overoptimizing for a single reward function (readability score) at the cost of the other (translation accuracy) with Qwen. GRPO exploits the problem of the jReadability score that it significantly increases when non-Japanese characters are used, resulting in generating non-Japanese characters, which defeats the purpose of the translation task. On the other hand, MO-GRPO evenly optimizes both objectives, achieving improvement on both.

Method Output jReadability\uparrow BLEURT\uparrow
Input Siso’s depictions of land, water center new gallery exhibition––
GRPO「シスコが描いた土地と水の新しい展覧会がギャラリーで始まります」(This translation conveys the main idea of the English text, focusing on the subject Siso (which could be a misspelling or a pseudonym for an artist like Claude Monet, known for his water lily paintings), the action depictions, and the new exhibition at a gallery, making it suitable for elementary school children and easy to understand.)0.77 0.61
MO-GRPO「シスコの陸と海を描いた新しい展覧会」0.57 0.77

Table 10: Generation examples of GRPO and MO-GRPO overoptimizing for a single reward function (readability score) at the cost of the other (translation accuracy) with Sarashina. GRPO exploits the problem of the jReadability score that it significantly increases when non-Japanese characters are used, resulting in generating non-Japanese characters, which defeats the purpose of the translation task. On the other hand, MO-GRPO evenly optimizes both objectives, achieving improvement on both.

## Appendix E Proof of Theorem and Proposition

### E.1 Proof of Theorem[1](https://arxiv.org/html/2509.22047v2#Thmtheorem1 "Theorem 1 (Correlation between reward function and advantage function with GRPO). ‣ 3 Group Relative Policy Optimization (GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")

From here on, for simplicity, we omit the notation for prompt q and optional output o_{g} (e.g., R_{i}(q,o_{g})\rightarrow R_{i}). We assume the number of samples G\to\infty, which allows the sample statistics to approximate the true population parameters. Let R_{1},\dots,R_{i} be K reward functions. Let \mu_{i}=\mathbb{E}[R_{i}] and \sigma_{i}^{2}=\operatorname{Var}[R_{i}] denote the mean and variance of the i-th reward, respectively.

\displaystyle\operatorname{Cov}(R_{i},A)=\operatorname{Cov}\left(R_{i},\frac{\mathbf{R}-\mathbb{E}[\mathbf{R}]}{\sigma}\right)(27)
\displaystyle=\frac{1}{\sigma}\operatorname{Cov}(R_{i},\mathbf{R})(28)
\displaystyle=\frac{1}{\sigma}\operatorname{Cov}\left(R_{i},\sum_{j=1}^{K}R_{j}\right)(29)
\displaystyle=\frac{1}{\sigma}\sum_{j=1}^{K}\operatorname{Cov}(R_{i},R_{j})(30)
\displaystyle=\frac{1}{\sigma}\left(\operatorname{Cov}(R_{i},R_{i})+\sum_{j\neq i}\operatorname{Cov}(R_{i},R_{j})\right)(31)
\displaystyle=\frac{1}{\sigma}(\operatorname{Var}[R_{i}]+X)=\frac{\sigma_{i}^{2}+X}{\sigma}(32)

Finally, we can get the correlation between each reward function and advantage function with GRPO.

\displaystyle\operatorname{Corr}(R_{i},A)\displaystyle=\frac{\operatorname{Cov}(R_{i},A)}{\sqrt{\operatorname{Var}[R_{i}]\operatorname{Var}[A]}}(33)
\displaystyle=\frac{\frac{\sigma_{i}^{2}+X}{\sigma}}{\sqrt{\sigma_{i}^{2}\cdot 1}}(34)
\displaystyle=\frac{\sigma_{i}^{2}+X}{\sigma\sigma_{i}}(35)

The intuitive understanding of this proposition is that when a reward function R_{i} has a negative correlation with other reward functions, X becomes negative, thereby reducing the influence of R_{i} on the advantage function.

### E.2 Proof of Theorem[2](https://arxiv.org/html/2509.22047v2#Thmtheorem2 "Theorem 2 (Correlation between a reward function and advantage function with MO-GRPO). ‣ 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")

From here on, for simplicity, we omit the notation for prompt q and optional output o_{g} (e.g., R_{i}(q,o_{g})\rightarrow R_{i}). We assume the number of samples G\to\infty, which allows the sample statistics to approximate the true population parameters. Let R_{1},\dots,R_{i} be K reward functions. Let \mu_{i}=\mathbb{E}[R_{i}] and \sigma_{i}^{2}=\operatorname{Var}[R_{i}] denote the mean and variance of the i-th reward, respectively.

A^{\text{MO}}\;:=\;\sum^{K}_{j=1}\color[rgb]{0,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@gray@stroke{0}\pgfsys@color@gray@fill{0}\frac{R_{j}\;-\;\mathrm{mean}\bigl(R_{j}\bigl)}{\mathrm{std}(R_{j}\bigr)}.(36)

We first calculate \operatorname{Var}[A^{\text{MO}}].

\displaystyle\operatorname{Var}\left(A^{\mathrm{MO}}\right)\displaystyle=\sum_{j=1}^{K}1+\sum_{l\neq j}\frac{\operatorname{Cov}\left(R_{l},R_{j}\right)}{\sigma_{l}\sigma_{j}}(37)
\displaystyle=K+\underbrace{\sum_{j=1}^{K}\sum_{l\neq j}^{K}\frac{\operatorname{Cov}\left(R_{l},R_{j}\right)}{\sigma_{l}\sigma_{j}}}_{Y}(38)

The corresponding correlation is:

\displaystyle\operatorname{Corr}(R_{i},A^{\text{MO}})\displaystyle=\frac{\operatorname{Cov}(R_{i},\sum_{j=1}^{K}\frac{R_{j}-\mu_{j}}{\sigma_{j}})}{\sigma_{i}\sqrt{K+Y}}(39)
\displaystyle=\frac{\sum_{j=1}^{K}\frac{1}{\sigma_{j}}\operatorname{Cov}\left(R_{i},R_{j}\right)}{\sigma_{i}\sqrt{K+Y}}(40)
\displaystyle=\frac{\sigma_{i}\sum_{j=K}\frac{\operatorname{Cov}\left(R_{i},R_{j}\right)}{\sigma_{i}\sigma_{j}}}{\sigma_{i}\sqrt{K+Y}}(41)
\displaystyle=\frac{1+\sum_{j\neq i}\frac{\operatorname{Cov}\left(R_{i},R_{j}\right)}{\sigma_{i}\sigma_{j}}}{\sqrt{K+Y}}(42)
\displaystyle=\frac{1+Z}{\sqrt{K+Y}}(43)

### E.3 Proof of Proposition[1](https://arxiv.org/html/2509.22047v2#Thmproposition1 "Proposition 1 (Affine Invariance of MO-GRPO Advantage). ‣ Invariance to positive affine transformation. ‣ 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")

For simplicity, we know \mu_{i} and \sigma_{i}, the true mean and standard deviation of R_{i} over a group of outputs \mathbf{o}. The mean \mu^{\prime}_{i} and standard deviation \sigma^{\prime}_{i} of the transformed reward R^{\prime}_{i} are:

\displaystyle\mu^{\prime}_{i}\displaystyle=\mathbb{E}[a_{i}R_{i}+b_{i}]=a_{i}\mu_{i}+b_{i}
\displaystyle\sigma^{\prime}_{i}\displaystyle=\mathrm{std}(a_{i}R_{i}+b_{i})=a_{i}\sigma_{i}\quad(\text{since }a_{i}>0)

The i-th advantage function calculated using the transformed reward R^{\prime}_{i} for any o is:

\displaystyle\frac{R^{\prime}_{i}(o)-\mu^{\prime}_{i}}{\sigma^{\prime}_{i}}\displaystyle=\frac{(a_{i}R_{i}(o)+b_{i})-(a_{i}\mu_{i}+b_{i})}{a_{i}\sigma_{i}}(45)
\displaystyle=\frac{a_{i}(R_{i}(o)-\mu_{i})}{a_{i}\sigma_{i}}(46)
\displaystyle=\frac{R_{i}(o)-\mu_{i}}{\sigma_{i}}(47)

Since each advantage function is invariant, their sum A^{\text{MO}} is also invariant.

### E.4 Proof of Proposition[2](https://arxiv.org/html/2509.22047v2#Thmproposition2 "Proposition 2. ‣ Invariance to positive affine transformation. ‣ 4 Multi-Objective GRPO (MO-GRPO) ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")

The advantage function of GRPO’s mean and standard deviation of rewards between groups remains unchanged, allowing us to focus on the value of the rewards. For simplicity, we consider two reward functions and two outputs, o_{a} and o_{b}. Assume a trade-off scenario R_{1}(o_{a})>R_{1}(o_{b}) and R_{2}(o_{a})<R_{2}(o_{b}) and R_{1}(o_{b})+R_{2}(o_{b})<R_{1}(o_{a})+R_{2}(o_{a}). We consider a scaling \mathcal{R}^{\prime} where R^{\prime}_{i}=a_{i}R_{i} (i.e., b_{i}=0) and derive the condition under which the preference ordering is reversed:

\displaystyle A_{a}>A_{b}
\displaystyle\Rightarrow A^{\prime}_{a}<A^{\prime}_{b}.

where A^{\prime}_{a}=\frac{\sum_{i=1}^{K}R^{\prime}_{i}(q,o_{g})\;-\;\mathrm{mean}\bigl(\sum_{i=1}^{K}R^{\prime}_{i}(q,\mathbf{o})\bigr)}{\mathrm{std}\bigl(\sum_{i=1}^{K}R^{\prime}_{i}(q,\mathbf{o})\bigr)}. This reduces to:

\displaystyle a_{1}R_{1}(o_{b})+a_{2}R_{2}(o_{b})\displaystyle>a_{1}R_{1}(o_{a})+a_{2}R_{2}(o_{a})(48)
\displaystyle\Rightarrow\frac{a_{2}}{a_{1}}\displaystyle>\frac{R_{1}(o_{a})-R_{1}(o_{b})}{R_{2}(o_{b})-R_{2}(o_{a})}(49)

Such a_{1} and a_{2} exist, so GRPO does not hold.

## Appendix F Experiment Settings in WMT

Table[11](https://arxiv.org/html/2509.22047v2#A6.T11 "Table 11 ‣ Appendix F Experiment Settings in WMT ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows the prompt to evaluate on gpt-4o-mini, and Table[12](https://arxiv.org/html/2509.22047v2#A6.T12 "Table 12 ‣ Appendix F Experiment Settings in WMT ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") shows the parameter settings applied in the experiment.

Table 11: Prompt used for GPT‑4o‑mini‑based evaluation.

Parameter
temperature 0.7
learning rate 2e-6
adam beta1 0.9
adam beta2 0.99
weight decay 0.1
gradient accumulation steps 4
num generations 8
num train epochs 3
beta 0.04

Table 12: Parameter Setting of the Experiment in WMT for GRPO, Dr. GRPO, and MO-GRPO.

## Appendix G Reproducibility Statement

The experiments are conducted using an NVIDIA A100 GPU with 80 GB VRAM.

All the code of the experiments will be open-sourced on publication. The datasets and models used in the experiments ar e publicly available (Table[13](https://arxiv.org/html/2509.22047v2#A7.T13 "Table 13 ‣ Appendix G Reproducibility Statement ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")) except for GPT-4o-mini used for evaluation.

Name Reference
WMT Kocmi et al.([2024](https://arxiv.org/html/2509.22047v2#bib.bib25 "Findings of the WMT24 general machine translation shared task: the LLM era is here but MT is not solved yet"))[https://github.com/wmt-conference](https://github.com/wmt-conference)
BLEURT Sellam et al.([2020](https://arxiv.org/html/2509.22047v2#bib.bib61 "BLEURT: Learning Robust Metrics for Text Generation"))[https://huggingface.co/lucadiliello/BLEURT-20](https://huggingface.co/lucadiliello/BLEURT-20)
Sarashina[https://huggingface.co/sbintuitions/sarashina2.2-3b-instruct-v0.1](https://huggingface.co/sbintuitions/sarashina2.2-3b-instruct-v0.1)
Qwen Yang et al.([2025](https://arxiv.org/html/2509.22047v2#bib.bib64 "Qwen3 technical report"))[https://huggingface.co/Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
Llama Grattafiori et al.([2024](https://arxiv.org/html/2509.22047v2#bib.bib65 "The Llama 3 Herd of Models"))[https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
jReadability Hasebe and Lee ([2015](https://arxiv.org/html/2509.22047v2#bib.bib24 "Introducing a readability evaluation system for Japanese language education"))[https://github.com/joshdavham/jreadability](https://github.com/joshdavham/jreadability)
Alapca Dubois et al.([2023](https://arxiv.org/html/2509.22047v2#bib.bib73 "AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback"))[https://huggingface.co/datasets/tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
RM-Mistral-7B Dong et al.([2023](https://arxiv.org/html/2509.22047v2#bib.bib74 "RAFT: Reward ranked Finetuning for Generative Foundation Model Alignment"))[https://huggingface.co/weqweasdas/RM-Mistral-7B](https://huggingface.co/weqweasdas/RM-Mistral-7B)
TRank Trokhymovych et al.([2024](https://arxiv.org/html/2509.22047v2#bib.bib77 "An Open Multilingual System for Scoring Readability of Wikipedia"))[https://huggingface.co/trokhymovych/TRank_readability](https://huggingface.co/trokhymovych/TRank_readability)

Table 13: List of datasets and models used in the experiments.

## Appendix H Supplementary Result

We have shown cases where some reward function is learned, but we will conduct supplementary experiments to confirm whether the proposed method maintains the performance of GRPO when the reward function is not over-optimized.

Method BLEURT\uparrow jReadability\uparrow
GRPO 0.76 0.72
MO-GRPO (ours)0.76 0.72

Table 14: Translation quality on WMT23 De-En, training dataset is WMT-21, 22.

Method BLEURT\uparrow jReadability\uparrow
GRPO 0.78 0.68
MO-GRPO (ours)0.78 0.70

Table 15: Translation quality on WMT23 Ru-En, training dataset is WMT-21, 22.

## Appendix I Supplementry Results of mo-reacher-v5

We show the results of GRPO (Figure[7](https://arxiv.org/html/2509.22047v2#A9.F7 "Figure 7 ‣ Appendix I Supplementry Results of mo-reacher-v5 ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")) and MO-GRPO (Figure[8](https://arxiv.org/html/2509.22047v2#A9.F8 "Figure 8 ‣ Appendix I Supplementry Results of mo-reacher-v5 ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")) in mo-reacher-v5.

![Image 9: Refer to caption](https://arxiv.org/html/2509.22047v2/img/img_mujoco/grpo1.png)

![Image 10: Refer to caption](https://arxiv.org/html/2509.22047v2/img/img_mujoco/grpo2.png)

![Image 11: Refer to caption](https://arxiv.org/html/2509.22047v2/img/img_mujoco/grpo3.png)

![Image 12: Refer to caption](https://arxiv.org/html/2509.22047v2/img/img_mujoco/grpo4.png)

![Image 13: Refer to caption](https://arxiv.org/html/2509.22047v2/img/img_mujoco/grpo5.png)

![Image 14: Refer to caption](https://arxiv.org/html/2509.22047v2/img/img_mujoco/grpo6.png)

![Image 15: Refer to caption](https://arxiv.org/html/2509.22047v2/img/img_mujoco/grpo7.png)

Figure 7: This shows the results of GRPO in mo-reacher-v5. GRPO’s learned policy does not swing the reacher once, but rather stops in the right half, close to reward function 1 R_{1}.

![Image 16: Refer to caption](https://arxiv.org/html/2509.22047v2/img/img_mujoco/mogrpo1.png)

![Image 17: Refer to caption](https://arxiv.org/html/2509.22047v2/img/img_mujoco/mogrpo2.png)

![Image 18: Refer to caption](https://arxiv.org/html/2509.22047v2/img/img_mujoco/mogrpo3.png)

![Image 19: Refer to caption](https://arxiv.org/html/2509.22047v2/img/img_mujoco/mogrpo4.png)

![Image 20: Refer to caption](https://arxiv.org/html/2509.22047v2/img/img_mujoco/mogrpo5.png)

![Image 21: Refer to caption](https://arxiv.org/html/2509.22047v2/img/img_mujoco/mogrpo6.png)

![Image 22: Refer to caption](https://arxiv.org/html/2509.22047v2/img/img_mujoco/mogrpo7.png)

Figure 8: This shows the results of MO-GRPO in mo-reacher-v5. The policy learned by MO-GRPO successfully completes one swing of the reacher.

## Appendix J Practical Implementation

1 def MO_GRPO(reward_1,reward_2):

2 combined_scores=[]

3 standard_deviation_reward_1=np.std(reward_1)+1 e-6

4 standard_deviation_reward_2=np.std(reward_2)+1 e-6

5 reward_1_norm=(reward_1-np.mean(reward_1))/standard_deviation_reward_1

6 reward_2_norm=(reward_2-np.mean(reward_2))/standard_deviation_reward_2

7

8 for i in range(len(group_samples)):

9 combined_score=(

10(reward_1_norm[i]+

11 reward_2_norm[i])/np.sqrt(2)

12)

13 combined_scores.append(combined_score)

## Appendix K Training Process of GRPO and MO-GRPO with BLEURT and jReadability

We show the training process of MO-GRPO with BLEURT and jReadability by Llama and Qwen (Figure[9](https://arxiv.org/html/2509.22047v2#A11.F9 "Figure 9 ‣ Appendix K Training Process of GRPO and MO-GRPO with BLEURT and jReadability ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")).

![Image 23: Refer to caption](https://arxiv.org/html/2509.22047v2/x9.png)

![Image 24: Refer to caption](https://arxiv.org/html/2509.22047v2/x10.png)

Figure 9: (Machine translation) The training process of MO-GRPO on the WMT En-Ja dataset uses BLEURT and jReadability as the reward functions by Llama, and Qwen. Unlike GRPO (Figure[5](https://arxiv.org/html/2509.22047v2#S5.F5 "Figure 5 ‣ 5.2 Simulated Control ‣ 5 Experiment ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), Figure[10](https://arxiv.org/html/2509.22047v2#A11.F10 "Figure 10 ‣ Appendix K Training Process of GRPO and MO-GRPO with BLEURT and jReadability ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems"), and Figure[11](https://arxiv.org/html/2509.22047v2#A11.F11 "Figure 11 ‣ Appendix K Training Process of GRPO and MO-GRPO with BLEURT and jReadability ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")), the figure shows MO-GRPO avoids overfitting of jReadability and prevents deterioration of BLEURT in all language models.

We show the training process of GRPO with BLEURT and jReadability by Llama and Qwen (Figure[10](https://arxiv.org/html/2509.22047v2#A11.F10 "Figure 10 ‣ Appendix K Training Process of GRPO and MO-GRPO with BLEURT and jReadability ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems") and Figure[11](https://arxiv.org/html/2509.22047v2#A11.F11 "Figure 11 ‣ Appendix K Training Process of GRPO and MO-GRPO with BLEURT and jReadability ‣ MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems")).

![Image 25: Refer to caption](https://arxiv.org/html/2509.22047v2/x11.png)

Figure 10: The training process of GRPO on the WMT En-Ja dataset uses BLEURT and jReadability as the reward functions by Llama. As the results show, GRPO overfits jReadability at the expense of BLEURT performance. As the results show, the standard deviation of jReadability is always more than BLEURT.

![Image 26: Refer to caption](https://arxiv.org/html/2509.22047v2/x12.png)

Figure 11: The training process of GRPO on the WMT En-Ja dataset uses BLEURT and jReadability as the reward functions by Qwen. As the results show, GRPO overfits jReadability at the expense of BLEURT performance. As the results show, the standard deviation of jReadability is always more than BLEURT.
