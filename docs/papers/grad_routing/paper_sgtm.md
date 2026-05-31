Title: Knowledge Localization for Capability Removal in LLMs

URL Source: https://arxiv.org/html/2512.05648

Published Time: Mon, 08 Dec 2025 01:32:44 GMT

Markdown Content:
## Beyond Data Filtering: 

Knowledge Localization for 

Capability Removal in LLMs

###### Abstract

Large Language Models increasingly possess capabilities that carry dual-use risks. While data filtering has emerged as a pretraining-time mitigation, it faces significant challenges: labeling whether data is harmful is expensive at scale, and given improving sample efficiency with larger models, even small amounts of mislabeled content could give rise to dangerous capabilities. To address risks associated with mislabeled harmful content, prior work proposed Gradient Routing(cloud2024gradient) – a technique that localizes target knowledge into a dedicated subset of model parameters so they can later be removed. We explore an improved variant of Gradient Routing, which we call Selective GradienT Masking (SGTM), with particular focus on evaluating its robustness to label noise. SGTM zero-masks selected gradients such that target domain examples only update their dedicated parameters. We test SGTM’s effectiveness in two applications: removing knowledge of one language from a model trained on a bilingual synthetic dataset, and removing biology knowledge from a model trained on English Wikipedia. In both cases SGTM provides better retain/forget trade-off in the presence of labeling errors compared to both data filtering and a previously proposed instantiation of Gradient Routing. Unlike shallow unlearning approaches that can be quickly undone through fine-tuning, SGTM exhibits strong robustness to adversarial fine-tuning, requiring seven times more fine-tuning steps to reach baseline performance on the forget set compared to a finetuning-based unlearning method (RMU). Our results suggest SGTM provides a promising pretraining-time complement to existing safety mitigations, particularly in settings where label noise is unavoidable.3 3 3 Code available at https://github.com/safety-research/selective-gradient-masking

Igor Shilov 1,3

Alex Cloud†,2, Aryo Pradipta Gema†,1,4, Jacob Goldman-Wetzler†,2, Nina Panickssery†,2, Henry Sleight†,5

Erik Jones∗,2, Cem Anil∗,2

1 Anthropic Fellows Program, 2 Anthropic, 3 Imperial College London, 4 University of Edinburgh, 5 Constellation

∗Equal advising 

†Names appear alphabetically

## 1 Introduction

![Image 1: Refer to caption](https://arxiv.org/html/2512.05648v1/x1.png)

(a) 

![Image 2: Refer to caption](https://arxiv.org/html/2512.05648v1/x2.png)

(b) 

Figure 1: SGTM shows better retain/forget trade-off when removing biology knowledge from a model trained on Wikipedia compared to data filtering. We compare Selective GradienT Masking (SGTM) with two data filtering baselines: weak (removing only articles classified as Biology) and strict (also removing Medicine & Health, Chemistry, and Earth & Environment articles). The y-axis shows forget loss (Biology), where higher values indicate stronger removal of biology knowledge. The x-axis shows retain loss (non-biology topics), where lower values indicate better general capability retention. Each line represents the progress of one training run, each point a checkpoint at equal intervals. Stars show final checkpoints. Dashed lines show equal compute expenditure in FLOPs (not shown on right). On general knowledge (left) and biology-adjacent knowledge (right) SGTM yields higher forget loss at any given retain loss value. SGTM incurs a compute efficiency penalty, requiring more compute to achieve the same retain loss value.

As LLMs grow more capable, concerns are being raised over their potential misuse – ranging from software exploits to dangerous chemical, biological, radiological, and nuclear (CBRN) applications(urbina2022dual; kang2024exploiting). Post-training mitigations, such as refusal training or output classifiers, are improving, yet continue to face challenges from determined adversaries(andriushchenko2024does; mckenzie2025stack). This motivates interventions earlier in the training pipeline, to prevent models from acquiring certain capabilities in the first place.

In response, recent research studied data filtering, a common pretraining-time approach, aiming to exclude harmful or restricted content before it can be learned(obrien2025deepignorance; anthropic2025pretraining; maini2025safety). Achieving comprehensive and precise filtering at scale is challenging: acquiring high-quality labels is expensive at scale(anwar2024foundational), undesired content is often embedded within benign documents(dodge2021documenting), and many concepts are entangled between harmful and beneficial use cases(pannu2025dual). Compounding this, recent evidence suggests that as models and datasets scale, even small absolute quantities of maliciously crafted data can be sufficient to influence model behavior(souly2025poisoning). This leads to an inevitable trade-off: developers must either accept false negatives (retaining dangerous content), or remove data useful for general capabilities(obrien2025deepignorance).

Recent research proposed localizing target knowledge to a subset of the model’s parameters, which can later be erased to remove knowledge from the model. Methods include Gradient Routing(cloud2024gradient), which achieves localization by modifying gradients during pretraining, and Redirection for Erasing Memory(schoepf2025redirection), applied post-training. Both methods outperform data filtering in terms of removal performance in the presence of labeling errors.

We explore an improved variant of Gradient Routing, which we call Selective GradienT Masking (SGTM). Following the general Gradient Routing framework, SGTM assigns a dedicated subset of model parameters to the target domain (e.g., CBRN). Specifically, it allocates certain MLP neurons and attention heads in each transformer block. During training, SGTM selectively zero-masks gradients from examples representing the target domain such that they only update the dedicated portion of the network. After training, it removes the undesired capabilities by zeroing out the dedicated portion of the network, leaving the rest of the model’s knowledge mostly intact. Compared to the Gradient Routing variant proposed by cloud2024gradient, which masks activation gradients on a subset of layers, SGTM’s parameter-gradient masking is less disruptive to retain performance and, when applied across all layers, more effectively restricts information flow from forget data into non-forget parameters (See Appendix[A](https://arxiv.org/html/2512.05648v1#A1 "Appendix A Gradient Routing Variants ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")).

We first demonstrate SGTM’s robustness to label noise in a controlled synthetic setup, aiming to remove the knowledge of a language from a model trained on the bilingual TinyStories dataset(eldan2023tinystories). We show that SGTM outperforms the original Gradient Routing variant on both retain and forget performance (Figure[3](https://arxiv.org/html/2512.05648v1#S4.F3 "Figure 3 ‣ 4.1 Experimental Setup ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")), and to provide a better retain/forget trade-off in the presence of labeling errors compared to data filtering (Figure[4](https://arxiv.org/html/2512.05648v1#S4.F4 "Figure 4 ‣ 4.2 Results ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), left). We also quantify the rate of data leakage from forget data into non-forget parameters across multiple model scales, finding that leakage decreases as models grow larger (Figure[5](https://arxiv.org/html/2512.05648v1#S4.F5 "Figure 5 ‣ 4.3 Leakage ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")). This scaling trend suggests that SGTM becomes increasingly effective at localizing target knowledge in larger models, providing stronger protection against mislabeled or undiscovered forget samples as scale increases. Finally, we study the localization mechanistically using gradient norms, which serve as a proxy for identifying which parts of the network are being updated during training. We find that unlabeled forget examples predominantly update the designated forget parameters rather than the retain parameters, consistent with self-reinforcing knowledge localization (Figure[4](https://arxiv.org/html/2512.05648v1#S4.F4 "Figure 4 ‣ 4.2 Results ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")(b)).

We then apply SGTM to a realistic large-scale dataset, training a 254M parameter model on English Wikipedia, targeting biology knowledge for removal. This demonstrates SGTM’s performance under realistic label noise from a real-world content classifier. SGTM provides better retain/forget trade-off than both weak (removing only biology data) and strict (also removing medicine, chemistry and environment data) filtering baselines (Figure[1](https://arxiv.org/html/2512.05648v1#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")). We show SGTM to be the best performing Gradient Routing variant on this task (Figure[7](https://arxiv.org/html/2512.05648v1#A1.F7 "Figure 7 ‣ Appendix A Gradient Routing Variants ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")). Evaluated by compute efficiency, SGTM slows training on general knowledge by 6% (Figure[9](https://arxiv.org/html/2512.05648v1#A2.F9 "Figure 9 ‣ B.2 Results ‣ Appendix B Compute penalty ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")). To ensure that aggregate loss metrics are not misleading, we analyze per-token loss and show that SGTM’s increased forget loss reflects broadly elevated error across biology tokens rather than a small number of extreme outliers (Figure[6](https://arxiv.org/html/2512.05648v1#S5.F6 "Figure 6 ‣ 5.2 Results ‣ 5 Realistic Dataset (Wikipedia) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")(b)).

Finally, we show SGTM to be robust to adversarial fine-tuning (Figure[4](https://arxiv.org/html/2512.05648v1#S4.F4 "Figure 4 ‣ 4.2 Results ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), right), with relearning speed much slower than Representation Misdirection for Unlearning (RMU)(li2024wmdp), a state-of-the-art traditional unlearning technique. It takes SGTM 7\times more forget tokens in fine-tuning than RMU to achieve the baseline forget loss (92M vs 13M tokens). In the same setup, weak data filtering takes 85M and strict data filtering takes 92M forget tokens to achieve the baseline loss.

Our contributions are as follows:

1.   1.We propose an improved variant of Gradient Routing(cloud2024gradient), Selective GradienT Masking (SGTM) (Section[3](https://arxiv.org/html/2512.05648v1#S3 "3 Method ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")), which is less disruptive to retain performance while more strictly limiting information flow from forget samples to non-forget parameters. 
2.   2.We show that SGTM achieves a better retain/forget trade-off under label noise compared to both prior Gradient Routing variants and data filtering. This holds across a controlled synthetic setup with artificial mislabeling (Section[4](https://arxiv.org/html/2512.05648v1#S4 "4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), Appendix[A](https://arxiv.org/html/2512.05648v1#A1 "Appendix A Gradient Routing Variants ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")) and a realistic dataset (Section[5](https://arxiv.org/html/2512.05648v1#S5 "5 Realistic Dataset (Wikipedia) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")). 
3.   3.We quantify the leakage of information from mislabeled forget samples into retain parameters across models ranging from 8M to 64M parameters, finding that it consistently decreases with model scale (Section[4.3](https://arxiv.org/html/2512.05648v1#S4.SS3 "4.3 Leakage ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")). While these models remain relatively small, the observed trend suggests that SGTM’s localization becomes stronger as scale increases. 
4.   4.We investigate the mechanism behind SGTM’s self-reinforcing knowledge localization, analyzing gradient norms and per-token losses. We find that (a) unlabeled forget samples primarily update forget parameters while retain samples update retain parameters (Section[4.4](https://arxiv.org/html/2512.05648v1#S4.SS4 "4.4 Gradient Norms ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")), and (b) increased forget loss reflects broadly elevated error across target tokens rather than a few extreme outliers (Section[5.4](https://arxiv.org/html/2512.05648v1#S5.SS4 "5.4 Per-token losses ‣ 5 Realistic Dataset (Wikipedia) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")). 
5.   5.We show that SGTM remains robust to adversarial finetuning, unlike finetuning–based unlearning methods, which quickly recover forgotten knowledge (Section[5.3](https://arxiv.org/html/2512.05648v1#S5.SS3 "5.3 Robustness to fine-tuning ‣ 5 Realistic Dataset (Wikipedia) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")). 

## 2 Related Work

Post-training safety mitigations. A common line of defense is to apply mitigations after pretraining. Refusal training teaches models to decline unsafe requests, but these safeguards can often be bypassed through jailbreaks and prompt engineering(kumar2024refusal; andriushchenko2024does). Output classifiers – auxiliary models that filter generated text – can be circumvented by determined adversaries(schwinn2023adversarial; mckenzie2025stack). Machine unlearning techniques instead attempt to erase specific knowledge from trained models(yao2024large; liu2024towards; barez2025open; liu2025rethinking), but remain brittle: suppressed information can often be recovered through adversarial fine-tuning(deeb2024unlearning; lermen2023lora), benign fine-tuning on unrelated tasks(hu2024unlearning), jailbreaks(lucki2024adversarial), or rephrased queries(lynch2024eight).

Data Filtering. Pre-training data filtering is increasingly adopted by frontier model developers(OpenAI2025; meta2025llama4; anthropic2025system; agarwal2025gpt; team2025gemma) and is effective at improving model safety(maini2025safety; li2025bad). By preventing the initial acquisition of dangerous knowledge, data filtering proves to be more robust to fine-tuning attacks than post-hoc unlearning(obrien2025deepignorance). However, data filtering faces a critical challenge: acquiring high-quality labels at scale. The enormous size of pre-training datasets forces developers to rely on cheap, imperfect filtering strategies such as keyword filters, heuristics, and lightweight classifiers(longpre2024pretrainer; stranisci2025they; anthropic2025pretraining; albalak2024survey). These approaches suffer from high false positive rates and miss nuanced harmful content requiring contextual understanding(welbl2021challenges; paullada2021data). For instance, the hazardous biology classifier proposed by obrien2025deepignorance achieves only 44% precision at 98% recall, leading to the removal of over 8% of training data.

Knowledge Localization. Recent work has explored an alternative pretraining-time approach to localize specific knowledge to particular model parameters during training, enabling targeted removal. Inspired by modular architectures that separate knowledge across specialized components (e.g., Mixture of Experts(shazeer2017outrageously; gururangan2021demix; park2025monet), modular architectures(jacobs1991task; jacobs1991adaptive; andreas2016nmn; alet2018modular; kirsch2018modular; ruder2019latent; pfeiffer2023modular), or adapters(hu2022lora; ponti-etal-2023-combining)), these methods explicitly enforce localization through gradient control to allow strict localization of specific knowledge that one might wish to remove. The gradient masking in SGTM mirrors adapter methods like LoRA(hu2022lora): while the forward pass uses all model parameters, the backward pass selectively updates only forget-specific parameters, analogous to how LoRA restricts gradient updates to adapter modules while keeping the base model frozen. cloud2024gradient propose Gradient Routing, applying weighted, data-dependent masks to the model’s computation graph to localize harmful knowledge into a designated subset of weights. ghosal2025memorizationsinks apply a similar approach focused on MLP layers to localize memorization of specific examples. schoepf2025redirection iteratively localize undesired knowledge to newly added neurons post-training. These methods share a crucial advantage over data filtering: the absorption property(cloud2024gradient). Even when some harmful examples are mislabeled as benign, gradient routing mechanisms can partially localize their impact to the designated parameters, maintaining effective removal despite labeling errors. Both cloud2024gradient and schoepf2025redirection demonstrate robustness to discovery rates as low as 50% of harmful samples, a scenario where data filtering fails. Our proposed method, SGTM, further improves the trade-off between retaining general capabilities and removing target knowledge, achieving better retain/forget trade-offs while maintaining robustness to labeling errors.

## 3 Method

### 3.1 Notation

![Image 3: Refer to caption](https://arxiv.org/html/2512.05648v1/x3.png)

Figure 2: Forget/Retain parameter split in Selective Gradient Masking. In each transformer block we designate certain number of attention heads and MLP hidden units to the forget data (orange). The remaining parameters are designated to the retain data. 

Intervention Parameters updated
Data Forward pass Backward pass\theta_{\text{forget}}\theta_{\text{retain}}
\mathbf{D}_{\text{forget}}—Mask retain gradients (\nabla_{\theta_{\text{retain}}}=0)✔✘
\mathbf{D}_{\text{unlabeled}}——✔✔
\mathbf{D}_{\text{retain}}Mask forget parameters (\theta_{\text{forget}}=0)—✘1✔

*   1 Due to the associated activations being set to zero.

Table 1: Training interventions applied to different data subsets in SGTM. Interventions are described in Section[3.2](https://arxiv.org/html/2512.05648v1#S3.SS2 "3.2 Training Interventions ‣ 3 Method ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"). Empty intervention (—) indicates that normal training procedure is followed. 

We consider a transformer block(vaswani2017attention) consisting of a multi-head attention and an MLP layer, with h attention heads, model dimension d, and MLP dimension d_{\text{MLP}}. We designate h_{\text{forget}} (out of h) attention heads and d_{\text{forget}} (out of d_{\text{MLP}}) MLP hidden units for the forget data, and the remaining attention heads and MLP units for the retain data. We split parameters into forget and retain segments across all transformer blocks. Figure[2](https://arxiv.org/html/2512.05648v1#S3.F2 "Figure 2 ‣ 3.1 Notation ‣ 3 Method ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") provides a simplified visualization of the split. We provide a detailed explanation of parameter designation in Appendix[F](https://arxiv.org/html/2512.05648v1#A6 "Appendix F Parameter Split ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs").

We refer to \theta_{\text{forget}} and \theta_{\text{retain}} to mean all parameters in the model with the given designation. We can write the set of all model’s parameters as \theta=\{\theta_{\text{forget}},\ \theta_{\text{retain}}\}. Parameters outside transformer blocks (namely, embeddings) are considered part of \theta_{\text{retain}}, unless explicitly specified otherwise.

For the training data, we denote forget and retain data distributions as \mathcal{D}_{\text{forget}} and \mathcal{D}_{\text{retain}} respectively. Our goal is to train a model that performs well on \mathcal{D}_{\text{retain}}, but poorly on \mathcal{D}_{\text{forget}}. Note that these are idealized oracle data distributions and might not be accessible in practice. We then refer to the actual training dataset as \mathbf{D}. Accounting for the realistic data labeling, we assume \mathbf{D} to be split into three subsets: \mathbf{D}=\{\mathbf{D}_{\text{forget}},\ \mathbf{D}_{\text{retain}},\ \mathbf{D}_{\text{unlabeled}}\}. \mathbf{D}_{\text{forget}} and \mathbf{D}_{\text{retain}} are intended to contain samples where the input classifier is confident in the corresponding label, while uncertain or ambiguous samples would be a part of \mathbf{D}_{\text{unlabeled}}.

### 3.2 Training Interventions

Our method performs two types of interventions during training as summarized in Table[1](https://arxiv.org/html/2512.05648v1#S3.T1 "Table 1 ‣ 3.1 Notation ‣ 3 Method ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs").

Selective Gradient Masking. For samples from \mathbf{D}_{\text{forget}}, we apply selective gradient masking during the backward pass so that these samples do not update \theta_{\text{retain}}. We first compute gradients for all parameters normally, and then zero out gradients for \theta_{\text{retain}} before applying the optimizer (\nabla_{\theta}=\{\nabla_{\theta_{\text{forget}}},\ 0\}). Masking parameter gradients rather than activation gradients is the key distinction from the prior Gradient Routing method(cloud2024gradient). While both approaches prevent updates to \theta_{\text{retain}} on forget examples, masking activation gradients is more disruptive, because it blocks backpropagation through the masked activations, altering gradients for all remaining parameters. It also permits greater information flow from \mathbf{D}_{\text{forget}} into non-forget parameters, since activation-gradient masking does not block updates to down-projection layers (See Appendix[A](https://arxiv.org/html/2512.05648v1#A1 "Appendix A Gradient Routing Variants ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")).

Selective Parameter Masking. For samples from \mathbf{D}_{\text{retain}} we apply selective parameter masking during the forward pass to train the model to perform well on \mathcal{D}_{\text{retain}} even when \theta_{\text{forget}} parameters are set to 0. In particular, we zero-mask \theta_{\text{forget}} parameters during the forward pass, leading to corresponding activations being set to zero as well.

Ablation. After the training is complete, we set \theta_{\text{forget}}=0 to remove knowledge specific to \mathcal{D}_{\text{forget}}.

## 4 Synthetic Dataset (TinyStories) Results

### 4.1 Experimental Setup

In this setup we aim at localizing and removing the knowledge of one language from a model trained on a bilingual dataset, with a varying level of label noise. We train a 64M-parameter model on 1.2B tokens from bilingual TinyStories dataset(eldan2023tinystories) in English and Spanish for one full epoch, following Chinchilla-optimal scaling(hoffmann2022training). We acquire the Spanish version by translating the original English dataset with Claude 3 Haiku.

We treat English data as retain (\mathcal{D}_{\text{retain}}) and Spanish data as forget (\mathcal{D}_{\text{forget}}). When constructing datasets \{\mathbf{D}_{\text{forget}},\mathbf{D}_{\text{retain}},\mathbf{D}_{\text{unlabeled}}\} under the perfect labeling, all of the Spanish data is allocated to \mathbf{D}_{\text{forget}}, and English data is randomly split between \mathbf{D}_{\text{unlabeled}} (75%) and \mathbf{D}_{\text{retain}} (25%).

Note that in this setup we have access to the ground truth labels, as the training data is synthetically generated. To quantify robustness to labeling errors, we introduce artificial mislabeling. We define “undiscovered forget percentage” as the percentage of all Spanish data that is allocated to \mathbf{D}_{\text{unlabeled}} instead of \mathbf{D}_{\text{forget}} (i.e., not explicitly labeled as forget data). This could also be seen as FNR (False Negative Rate) of the hypothetical classifier identifying the forget data.

![Image 4: Refer to caption](https://arxiv.org/html/2512.05648v1/x4.png)

(a) 

![Image 5: Refer to caption](https://arxiv.org/html/2512.05648v1/x5.png)

(b) 

Figure 3: SGTM robustly removes forget knowledge, remaining effective even when large fractions of forget data are unlabeled. We report calibrated losses (see Section[4.1](https://arxiv.org/html/2512.05648v1#S4.SS1 "4.1 Experimental Setup ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")) on (a) retain and (b) forget sets when attempting to remove Spanish from a model trained on bilingual (English/Spanish) TinyStories dataset. We vary the percentage of undiscovered forget data, i.e., the proportion of the forget set not labeled as such. (a) SGTM consistently achieves lower retain loss than the Gradient Routing variant of cloud2024gradient, while maintaining higher forget loss – Pareto dominating prior Gradient Routing across all discovery rates tested. (b) For all non-zero labeling error rates considered, SGTM demonstrates stronger forgetting than both Gradient Routing and data filtering.

As we are approximating a more realistic scenario when only a small portion of the weights would be dedicated to the forget knowledge, we designate 1 (out of 32) attention heads and 64 (out of 2048) MLP hidden units as \theta_{\text{forget}}, with the remaining parameters designated as \theta_{\text{retain}}. In this scenario \mathcal{D}_{\text{forget}} has a lot of unique tokens not present in \mathcal{D}_{\text{retain}}, so we update embeddings with both forget and retain data. For training with data filtering we simply ignore \mathbf{D}_{\text{forget}} and train for one epoch on \{\mathbf{D}_{\text{unlabeled}},\mathbf{D}_{\text{retain}}\}.

All losses reported in this section are calibrated over forget (Spanish) and retain (English) datasets. Calibration is computed with a trained logit bias post-training (see Appendix[I](https://arxiv.org/html/2512.05648v1#A9 "Appendix I Tinystories Experimental Details ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")). This calibration avoids superficially high losses post-ablation due to extremely low probabilities of relevant tokens.

Further details on the training process, logit calibration and the dataset are presented in Appendix[I](https://arxiv.org/html/2512.05648v1#A9 "Appendix I Tinystories Experimental Details ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs").

### 4.2 Results

![Image 6: Refer to caption](https://arxiv.org/html/2512.05648v1/x6.png)

(a) 

![Image 7: Refer to caption](https://arxiv.org/html/2512.05648v1/x7.png)

(b) 

Figure 4: (a) SGTM shows better retain/forget trade-off than data filtering when removing the knowledge of a language from a bilingual model (1% artificial mislabeling). We show the trade-off between forget and retain loss on the task of removing Spanish knowledge from the bilingual TinyStories model. We set the rate of undiscovered forget data to 1%. Each line represents the progress of one training run, and each point is a checkpoint at equal intervals of the training. Stars show the final checkpoint. Dashed lines show the same proportion of training completed. We compare SGTM with data filtering (removing 99% of data) and Gradient Routing(cloud2024gradient). We also show “perfect filter” and “no filter” training as a reference. SGTM provides a better trade-off (higher forget loss at any fixed value of retain loss) than both 99% filter and Gradient Routing, closely approximating the oracle model represented by perfect filtering. (b) Unlabeled forget data mostly update forget parameters, and unlabeled retain data mostly update retain parameters. Each panel shows kernel density estimates of relative gradient norms (|\nabla_{\theta}|/|\theta|) for different parameter-data combinations. Forget parameters (green) and retain parameters (blue) are evaluated on both forget data (solid) and retain data (dashed) from the test set, with no gradient masking applied. Forget data predominantly updates forget parameters (top-left), while retain data predominantly updates retain parameters (top-right). Conversely, forget parameters receive much stonger updates from forget data (bottom-left). Retain parameters receive updates of similar magnitude from either forget and retain data, with slightly stronger updates from the retain data. 

Figure[3](https://arxiv.org/html/2512.05648v1#S4.F3 "Figure 3 ‣ 4.1 Experimental Setup ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") shows retain (a) and forget (b) loss with increasing rates of undiscovered forget data. SGTM maintains strictly better performance – lower retain loss and higher forget loss – than previously proposed variant of Gradient Routing(cloud2024gradient) across all discovery rates tested (See Appendix[A](https://arxiv.org/html/2512.05648v1#A1 "Appendix A Gradient Routing Variants ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") for details on the methodological differences that we believe explain the observed performance gap here). For data filtering, forget loss drops quickly when even a few forget samples are not filtered out. Both knowledge localization methods (SGTM and Gradient Routing) show slower decline in forget loss compared to data filtering as the rate of undiscovered forget data increases. On the retain set, however, SGTM shows higher loss than data filtering.

A natural concern would be that SGTM’s higher forget loss simply reflects a generally degraded model compared to data filtering, rather than successful forgetting. To explore this possibility, Figure[4](https://arxiv.org/html/2512.05648v1#S4.F4 "Figure 4 ‣ 4.2 Results ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")(a) shows the trade-off between forget and retain performance throughout training for SGTM, Gradient Routing, and three filtering options: perfect filter (0% undiscovered), 99% filter (1% undiscovered) and no filter (100% undiscovered). Though SGTM has slightly higher final retain loss than 99% filter (roughly equivalent to the 99% filter’s checkpoint at 80% of training), it offers better forget-retain trade-off – SGTM has higher forget loss at any fixed retain loss value. Note that SGTM aims to remove knowledge learned from forget training data as if was never seen by the model, and as such we do not expect or aim for it to outperform perfect data filtering. However, SGTM closely approximates forgetting performance of the perfect filter, with almost equivalent forget loss at the end of the training.

Here we’ve considered type II errors of the data classifier (false negative forget samples). We perform a detailed analysis with a full range of false positive and false negative rates in Appendix[D](https://arxiv.org/html/2512.05648v1#A4 "Appendix D Full Labeling Sensitivity and Specificity Analysis ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs").

### 4.3 Leakage

![Image 8: Refer to caption](https://arxiv.org/html/2512.05648v1/x8.png)

(a) 

![Image 9: Refer to caption](https://arxiv.org/html/2512.05648v1/x9.png)

(b) 

Figure 5: (a) Leakage is quantified via equivalent standard training comparison with variable number of forget tokens added to the data mix. The baseline curve (blue) maps the relationship between forget token exposure and forget loss established by training models on all retain data with increasing amounts of forget tokens added. Each blue point represents a model trained with standard training procedure with a given number of forget tokens added to the training dataset. For a given SGTM run (orange) we then take its forget loss and find the number of forget tokens that would achieve the same loss when added to the data mix in standard training (965k). The leakage is then computed by normalizing this number by the total number of (unlabeled) forget tokens in SGTM run. (b) Leakage decreases with model scale. Values denote the ratio of leaked information (measured in forget token exposure) to total undiscovered forget tokens, ranging between 0 (no leakage) and 1 (all information leaked). Larger models consistently exhibit lower leakage rates, with the 64M model maintaining leakage below 0.02 for up to 40% undiscovered forget data. 

In Section[4.2](https://arxiv.org/html/2512.05648v1#S4.SS2 "4.2 Results ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), we demonstrate that SGTM maintains high forget loss even when unlabeled portion of the training data (\mathbf{D}_{\text{unlabeled}}) contains forget examples. In this section, we quantify how much information from these undiscovered samples flows into the non-forget parameters (\theta_{\text{retain}} and \theta_{\text{joint}}) using a metric we call leakage. Here we provide an intuitive definition of the metric – see Appendix[G](https://arxiv.org/html/2512.05648v1#A7 "Appendix G Leakage Definition ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") for a formal definition.

To measure it, we first establish a baseline relationship between forget token exposure and model performance, as shown in Figure[5](https://arxiv.org/html/2512.05648v1#S4.F5 "Figure 5 ‣ 4.3 Leakage ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")(a). We train standard models (i.e., without SGTM) on the complete retain dataset combined with increasing rates of forget tokens added to the training data. This creates a mapping between the number of forget tokens in training data and the resulting forget loss, while keeping the retain data constant to control “general” capabilities. Note that in this section we report losses without logit calibration to reduce the number of hyperparameters and simplify the comparison across model sizes.

With this baseline, we convert forget loss values to equivalent forget token exposure. For example, an SGTM model trained with 20% undiscovered forget data (144M tokens) achieves forget loss equivalent to a baseline model trained on 965k forget tokens. This means that with 144M undiscovered forget tokens seen by the model, the model gained as much information on the target domain as a standard model would from 965k forget tokens. Note that SGTM is always trained on full retain and forget datasets, and undiscovered forget tokens here refer to those not labeled as such, i.e., where masking is not applied.

We define this ratio of the equivalent tokens to the total undiscovered tokens as leakage. Note that under this definition, the leakage rate of data filtering would always be 1, as data filtering model with a certain percentage of undiscovered forget tokens is equivalent to a standard model with all undiscovered forget tokens in the training data.

We evaluate leakage across a range of model sizes (8M to 64M parameters), scaling dataset size following Chinchilla-optimal principles (detailed training configurations are provided in Appendix[I.1](https://arxiv.org/html/2512.05648v1#A9.SS1 "I.1 Training Hyperparameters ‣ Appendix I Tinystories Experimental Details ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")). We maintain fixed forget dimensions (d_{\text{forget}} = 64, h_{\text{forget}}=1), which means that the proportion of forget-designated parameters naturally decreases with model scale.

Figure[5](https://arxiv.org/html/2512.05648v1#S4.F5 "Figure 5 ‣ 4.3 Leakage ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")(b) shows leakage values across a range of undiscovered forget rate values for models of varying size. First, for the largest model – 64M parameters, trained for one full epoch on the bilingual TinyStories dataset in a near Chinchilla-optimal configuration – leakage remains remarkably low: between 0.005 and 0.02 for undiscovered forget rates up to 40%.

Second, we observe a clear inverse relationship between model scale and leakage rates. Across all levels of undiscovered forget data, larger models consistently exhibit lower leakage than their smaller counterparts. This scaling behavior is particularly encouraging for the application of SGTM to larger-scale models, suggesting that the method’s effectiveness improves as models grow larger. This trend contrasts with data filtering, where recent work on data poisoning(souly2025poisoning) shows that the number of malicious or mislabeled samples required to influence model behavior remains roughly constant with scale – implying that larger models demand increasingly accurate classifiers to maintain the same level of protection.

### 4.4 Gradient Norms

To understand the mechanism underlying SGTM’s robustness to label noise, we hypothesize that the model develops self-reinforcing knowledge localization. Once the model begins localizing forget knowledge based on labeled examples (where we explicitly mask gradients), we expect that unlabeled forget samples (\mathcal{D}_{\text{forget}}\cap\mathbf{D}_{\text{unlabeled}}) would naturally gravitate toward using forget parameters, thereby sending stronger gradient signals to those parameters even without explicit masking.

To test this hypothesis, we analyze gradient norms from a SGTM model trained on the bilingual TinyStories dataset under perfect labeling conditions. To account for different parameter counts and magnitudes between \theta_{\text{forget}} and \theta_{\text{retain}}, we compute _relative gradient norms_|\nabla_{\theta}|/|\theta|, which measure the relative change in parameters induced by each data point. We compare per-sample gradient norms for forget and retain test examples, treating all samples as \mathbf{D}_{\text{unlabeled}} without applying any masking during forward or backward passes.

Figure[4](https://arxiv.org/html/2512.05648v1#S4.F4 "Figure 4 ‣ 4.2 Results ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")(b) presents distributions of relative gradient norms for forget (green) and retain (blue) parameters when processing forget (solid line) and retain (dashed line) data points. Each of the four resulting distributions appears in two panels for easy pairwise comparison.

The top row demonstrates clear specialization: forget data primarily updates forget parameters (left), while retain data primarily updates retain parameters (right). The bottom-left panel shows that forget weights receive substantially stronger updates from unlabeled forget data compared to unlabeled retain data, confirming the self-reinforcing localization hypothesis. This behavior provides empirical evidence for the _absorption_ mechanism originally conjectured by cloud2024gradient, which proposed that mislabeled or undiscovered samples could still direct their updates toward the designated forget parameters. Our results confirm that this mechanism indeed emerges in practice, explaining why Gradient Routing – and SGTM in particular – remains robust to label noise: knowledge is effectively localized, causing unlabeled or mislabeled forget samples to predominantly update forget-designated parameters rather than the retain set.

The bottom-right panel shows that retain parameters also show differential updating based on data type, though the effect is considerably weaker. While retain parameters receive stronger updates from retain data than forget data, the separation between distributions is less pronounced than for forget parameters. Our previous robustness results (Figure[3](https://arxiv.org/html/2512.05648v1#S4.F3 "Figure 3 ‣ 4.1 Experimental Setup ‣ 4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")) show minimal degradation with up to 20% unlabeled forget data, which would suggest minimal updates from unlabeled forget data to the retain parameters. This apparent discrepancy suggests that gradient norm alone may not fully capture the nature of parameter updates. The direction and alignment of gradients, rather than just their magnitude, likely play a crucial role in determining effective knowledge transfer. Future work should investigate the geometric properties of these gradient updates to better understand the knowledge localization mechanisms.

## 5 Realistic Dataset (Wikipedia) Results

### 5.1 Experimental Setup

In this setup we aim at localizing and removing the biology knowledge from a 254M parameter model trained on 3.7B tokens from an English Wikipedia dataset(wikidump). We label each Wikipedia article with an articletopic property provided by Wikipedia’s content classification service(halfaker2020ores; wikimedia_ores_articletopic). The taxonomy is hierarchical, covering 64 topics grouped into high-level categories: Culture, Geography, History and Society, and STEM (see Table[5](https://arxiv.org/html/2512.05648v1#A11.T5 "Table 5 ‣ Appendix K Articletopic taxonomy ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") for the full taxonomy). We apply one category per article, choosing the one with highest classifier score.

We then consider the “STEM.Biology” subcategory (3.7% of overall tokens) as forget set, and all remaining data as retain set. For evaluation we consider 3 separate test sets: forget (Biology), related knowledge (Earth and Environment, Medicine and Health, Chemistry) and general knowledge (Culture, Geography, History and Society). In each subcategory we set aside 5000 articles for the test set. For general and related knowledge we report average loss over test sets for all subcategories. For training, \mathbf{D}_{\text{forget}} contains all data labeled as biology, while remaining non-biology data is randomly split between \mathbf{D}_{\text{unlabeled}} (90%) and \mathbf{D}_{\text{retain}} (10%).

We consider two data filtering strategies. ”Weak filter” only removes data labeled as biology from the training set. ”Strong filter” additionally removes related categories, which are most likely to contain biology knowledge (Earth and Environment, Medicine and Health, Chemistry). With both data filtering strategies we maintain the same overall compute budget, training for the same number of steps on the remaining data after filtering.

When training with SGTM, we designate h_{\text{forget}}=1 (out of 32) attention head and d_{\text{forget}}=64 (out of 4096) MLP hidden units as \theta_{\text{forget}}. As in Section[4](https://arxiv.org/html/2512.05648v1#S4 "4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), here we report the loss after calibration with a trained logit bias post-training to avoid superficially high loss values post-ablation. Reported losses are averaged over 3 independent runs. We provide further details on the training process, logit calibration and the dataset in Appendix[J](https://arxiv.org/html/2512.05648v1#A10 "Appendix J Wikipedia Experimental Details ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs").

### 5.2 Results

![Image 10: Refer to caption](https://arxiv.org/html/2512.05648v1/x10.png)

(a) 

![Image 11: Refer to caption](https://arxiv.org/html/2512.05648v1/x11.png)

(b) 

Figure 6: (a) SGTM’s knowledge removal from the Wikipedia model is robust to adversarial fine-tuning. We measure the relearning rate by performing adversarial fine-tuning after removing biology knowledge from a model trained on Wikipedia. RMU (a state-of-the-art traditional post-training machine unlearning method) is brittle, quickly reaching the baseline forget loss in only 50 steps. SGTM (350 steps) is as robust as strict data filtering (350 steps), narrowly outperforming weak filtering (325 steps). It also maintains an advantage over strict data filtering for the first 150 fine-tuning steps. Each fine-tuning step represents 260k forget tokens. (b) SGTM removes biology knowledge broadly rather than through isolated token failures. We plot calibrated per-token loss distributions (see Section[5.1](https://arxiv.org/html/2512.05648v1#S5.SS1 "5.1 Experimental Setup ‣ 5 Realistic Dataset (Wikipedia) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")) on the biology test set after ablation for SGTM and data filtering baselines. SGTM shifts the distribution toward higher loss values compared to filtering methods, indicating that its higher forget loss reflects widespread degradation of biology knowledge across tokens rather than isolated extremely high-loss tokens. 

We first note that here, unlike the synthetic data setup from Section[4](https://arxiv.org/html/2512.05648v1#S4 "4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), we do not introduce additional labeling errors, and demonstrate the effectiveness of SGTM under more realistic conditions with natural label noise. With document-level labeling we expect that documents not labeled as biology could also contain biology knowledge, and an algorithm with strong label noise robustness should outperform strict filtering in this scenario.

Figure[1](https://arxiv.org/html/2512.05648v1#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") (left) shows the trade-off between the performance on the forget set and on general knowledge. Controlling for retain loss, SGTM achieves a higher forget loss than both filtering methods. Among filtering methods, strict filter shows stronger forgetting, as it removes more biology-adjacent data from the training set. Notably, both filtering methods have lower retain loss at the end of training compared to the ”no filter” baseline, reflecting more compute budget spent on non-biology data. SGTM, on the other hand, has slightly higher loss on the general retain than the baseline. We show in Appendix[B](https://arxiv.org/html/2512.05648v1#A2 "Appendix B Compute penalty ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") that the difference between SGTM and the baseline retain loss is equivalent to 5% compute penalty on the baseline model.

Figure[1](https://arxiv.org/html/2512.05648v1#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") (right) shows the trade-off between the performance on the forget set and on the biology-adjacent knowledge. Here, SGTM also outperforms both data filtering approaches by showing stronger forgetting at any fixed retain loss. The weak filter shows a better trade-off than the strict filter, as the latter disproportionately affects the retain set by removing relevant data categories from training. As a clear example of robustness to label noise, the final retain loss for SGTM lies between weak and strong data filters, while SGTM’s forget loss is higher than both weak and strong filters. This shows that SGTM retained some non-biology knowledge from Medicine/Chemistry/Environment domains (removed by the strict filter), while learning less biology from it than the weak filter (which did not filter out these).

We note that we would expect the loss on bio-adjacent domains to be higher as a result of removing biology data from the training – in fact we see that the final retain loss for the weak filter is higher than the baseline. Nevertheless, we still aim to have higher biology loss at any fixed retain loss, as it represents a higher level of preserved capabilities which do not help performance on biology.

### 5.3 Robustness to fine-tuning

To further assess whether SGTM achieves genuine knowledge localization despite label noise – rather than superficial suppression where mislabeled forget data leaks into retain parameters – we evaluate how quickly the forget knowledge can be re-acquired through adversarial fine-tuning. Figure[6](https://arxiv.org/html/2512.05648v1#S5.F6 "Figure 6 ‣ 5.2 Results ‣ 5 Realistic Dataset (Wikipedia) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") (left) shows our findings. We perform a full-parameter fine-tuning on a 50/50 mixture of forget and retain data, measuring how quickly the forget knowledge is recovered to the level of the baseline model with no data filtering. Each fine-tuning step represents 260k forget tokens.

Post-training unlearning methods are known to be brittle(deeb2024unlearning), which is demonstrated here by the model unlearned with RMU(li2024wmdp) recovering the baseline loss within 50 fine-tuning steps (13M forget tokens). Conversely, SGTM shows strong robustness to fine-tuning, requiring the same number of steps – 350 (92M forget tokens) – as the model trained with strict data filtering. Weak filter model took 325 steps (85M forget tokens) to reach the baseline loss.

In the Appendix[E](https://arxiv.org/html/2512.05648v1#A5 "Appendix E Robustness to Finetuning ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") we investigate if SGTM’s robustness to fine-tuning can be explained by a general model degradation – as it has higher final loss on general knowledge compared to both data filtering strategies. We compare SGTM’s fine-tuning with the fune-tuning runs started from undertrained data filtering checkpoints with an equivalent retain loss. We show that the SGTM’s fine-tuning run converges in its later stages to the equivalent fine-tuning of a model trained with a weak filter.

### 5.4 Per-token losses

One potential concern with using loss as a proxy for capability removal is that it aggregates performance over every token, potentially obscuring uneven behavior. In principle, a model could achieve a high mean forget loss even if the majority of tokens remain easy to predict, provided that a small subset of tokens incur extremely high loss values. That scenario would imply shallow or localized removal rather than a true erosion of the underlying knowledge. We view such a situation as unlikely given SGTM’s robustness to adversarial fine-tuning in Section[5.3](https://arxiv.org/html/2512.05648v1#S5.SS3 "5.3 Robustness to fine-tuning ‣ 5 Realistic Dataset (Wikipedia) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), however we include a per-token analysis to explicitly rule it out.

Figure[6](https://arxiv.org/html/2512.05648v1#S5.F6 "Figure 6 ‣ 5.2 Results ‣ 5 Realistic Dataset (Wikipedia) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")(b) reports the distribution of per-token losses on the biology test set for models trained with SGTM, two data filtering strategies and “no filter” baseline. The distributions show that SGTM shifts probability mass toward moderately higher loss values relative to baselines, increasing density around the upper-mid range rather than producing a heavy-tailed explosion of extreme outliers. This aligns with the intended behavior of localized knowledge removal rather than selective token probability suppression failure.

## 6 Discussion and Limitations

Experimental Limitations. Our experiments are conducted on a relatively small-scale models (64M and 254M parameters), orders of magnitude below the size of frontier systems. Consequently, the results should be interpreted as proof-of-concept rather than direct evidence of scalability to billion-parameter regimes. The forget set in our Wikipedia setup (\sim 4\% training tokens) likely exceeds real-world frequencies. We evaluate proxy scenarios (removing Spanish or Wikipedia biology knowledge) rather than genuine CBRN risks, and rely on simpler transformer architectures instead of modern mixture-of-experts. Given computational constraints and needing to train models from scratch, our models are not large enough to yield meaningful results on evaluations that directly probe dangerous capabilities, like WMDP(li2024wmdp). Our evaluation is thus based on loss metrics as a proxy; this may not reflect downstream task performance or conclusively demonstrate elimination of dangerous capabilities.

In-Context Attacks.shumailov2024ununlearning argues that knowledge removal methods leave models vulnerable to in-context attacks, where adversaries supply dangerous information at inference time. Data filtering has been shown to be susceptible to this(obrien2025deepignorance), and we expect SGTM to behave similarly, given the nature of our method. However, we view our method as part of a defense-in-depth approach, where knowledge localization and removal serve as one layer among multiple security measures rather than a standalone solution.

Future Work. Our method enables the creation of two model versions from a single training run: one possessing dual-use capabilities (before ablation) and another that is safe (after ablation of \theta_{\text{forget}}). This dual-model approach offers significant benefits, as trusted actors could benefit from having access to a knowledgeable assistant for legitimate purposes, such as medical research or biosecurity defense. There is evidence suggesting that fine-tuning models on target data post-training may not be as effective as having that data present during pre-training(kim2024knowledge; chang2024large). SGTM allows model developers to maintain both versions and deploy them according to their specific use cases and trust requirements. We believe it is an interesting avenue for future work, and we report early findings in Appendix[C](https://arxiv.org/html/2512.05648v1#A3 "Appendix C Pre-ablation performance ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs").

## 7 Conclusion

Selective Gradient Masking (SGTM) localizes target knowledge into designated parameters and is robust to label noise. Across both a synthetic bilingual setup and Wikipedia biology removal, SGTM achieves better retain/forget trade-off than data filtering and prior gradient routing variants. In the synthetic setup, we quantify information leakage from undiscovered forget data into non-forget parameters and find it to be minimal. We further show that SGTM is robust to adversarial fine-tuning, in contrast to traditional post-training unlearning methods.

Taken together, these results suggest SGTM as a promising alternative to data filtering, forming part of a broader defense-in-depth strategy for mitigating dual-use capabilities – especially in scenarios where a certain penalty on compute efficiency can be justified by the safety benefits of more reliable capability removal.

## Acknowledgements

Authors thank Scott Johnson, Alexander Hägele, Matthieu Meeus, Krishna Patel, Ethan Perez, Alec Radford, and Jascha Sohl-Dickstein for valuable discussions and feedback throughout this work. We thank John Hughes for providing infrastructure support for the Anthropic Fellows Program. We also thank the AE Studio team working on gradient routing (Ethan Roland, Murat Cubuktepe, Erick Martinez, Stijn Servaes, Keenan Pepper) for sharing their early results.

## Appendix

## Appendix A Gradient Routing Variants

![Image 12: Refer to caption](https://arxiv.org/html/2512.05648v1/x12.png)

Figure 7: Retain/forget trade-off when removing target knowledge domain (biology) from a model trained on Wikipedia. Each line represents the progress of one training run, and each point is a checkpoint at equal intervals of training. Stars show the final checkpoint. Retain loss is computed over general knowledge (left) and biology-adjacent knowledge (right).

Gradient Routing(cloud2024gradient) was proposed as a generic framework for applying weighted masks over the model’s computation graph to isolate the target knowledge into a specified subset of parameters. In this section we explore multiple methods following this framework and show that SGTM, as described in Section[3](https://arxiv.org/html/2512.05648v1#S3 "3 Method ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), provides better retain/forget trade-off than other Gradient Routing variants. Here we use \theta_{\text{joint}} to refer to parameters which are updated by both retain and forget data, but are not removed after training.

Note that below we will use the term Gradient Routing to refer to the particular instantiation of the framework used by cloud2024gradient, rather than the framework itself.

In addition to SGTM, we consider the following methods (for details on parameter notaion see Appendix[F](https://arxiv.org/html/2512.05648v1#A6 "Appendix F Parameter Split ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")):

*   •Gradient Routing applies zero-masks to the activation gradients. It differs from SGTM in two key aspects. First, the original variant masks the activation gradients, unlike SGTM which masks parameter gradients. Similarly to SGTM that also leads to \theta_{\text{retain}} not being updated on forget examples, but also prevents gradients from being backpropagated, thus changing the gradients for remaining parameters as well – which is disruptive for the model’s training. Second, the original variant allows for higher information flow from \mathbf{D}_{\text{forget}} to non-forget parameters: it does not mask all layers (leaving all non-target layers in \theta_{\text{joint}}), and by the virtue of masking activation gradients it does not block updates from forget data to down projection layers (W_{2},\ b_{2},\ W_{O},\ b_{O}). For SGTM, on the other hand, \theta_{\text{joint}} is empty by default. 
*   •Activation Masking zero-masks \theta_{\text{retain}} during the forward pass, effectively acting as a deterministic data-dependent dropout layer. Similarly to SGTM this strategy also leads to \theta_{\text{retain}} not being updated, but also affects both the loss and the gradients of all parameters. Similar to Gradient Routing, it also does not block updates to down projection layers. 
*   •SGTM (Joint Projection). This is a version of SGTM where MLP and attention projection parameters are considered to be joint weights: \theta_{\text{joint}}=\{W_{2},b_{2},W_{O},b_{O}\} 
*   •SGTM (Joint Attention). This is a version of SGTM where we only apply masking to the first MLP layer in each transformer block, setting all attention heads and the second MLP layer to be joint weights: \theta_{\text{joint}}=\{W_{2},b_{2},W_{O},b_{O},W_{QKV}^{(i)}\text{ for }i\in\{1,\ldots,h\}\}. We note that this approach closely resembles Memorization Sinks by ghosal2025memorizationsinks. 

All methods reported in this section use the same masking hyperparameters (if applicable), dedicating h_{\text{forget}}=1 (out of 32) attention head and d_{\text{forget}}=64 (out of 4096) MLP hidden units to the forget data.

Figure[7](https://arxiv.org/html/2512.05648v1#A1.F7 "Figure 7 ‣ Appendix A Gradient Routing Variants ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") shows the retain/forget trade-off on a task of removing biology knowledge from a model trained on Wikipedia (See Section[5](https://arxiv.org/html/2512.05648v1#S5 "5 Realistic Dataset (Wikipedia) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")). We plot the four methods described above alongside previously reported data for SGTM and three options for data filtering: weak (removing only biology category), strict (also removing biology-adjacent knowledge: Medicine, Chemistry, Environment), and no filtering at all.

Gradient Routing and Activation Masking have trade-off curves similar to the weak filter, but are less compute efficient: for the same compute budget both methods lie further right along the same curve. The compute penalty is most pronounced for Gradient Routing, where the loss at the end of training (both forget and retain) is roughly equivalent to that of a weak filter model at 20% of training.

SGTM variants which do not mask certain gradients (Joint Projection and Joint Attention) do have better final retain loss on both general and biology-adjacent knowledge, but overall provide worse retain/forget trade-off than SGTM, as well as both data filtering options.

## Appendix B Compute penalty

Raw loss values can be difficult to interpret due to their non-linear nature. For instance, reducing the loss from 10 to 9 (near random performance) requires substantially less data and compute than reducing from 3 to 2 (where the model already performs well). To provide a more interpretable measure, we convert loss values to the compute budget required to achieve that performance on a baseline model trained with unfiltered data.

### B.1 Scaling Laws

![Image 13: Refer to caption](https://arxiv.org/html/2512.05648v1/x13.png)

Figure 8: Scaling laws for models trained on Wikipedia. We fit and report three compute-to-loss scaling laws, one for each test subset we report: biology (forget), biology-adjacent (retain), general knowledge (retain)

We establish compute-to-loss scaling laws by training four models of different sizes (34M, 64M, 125M, and 254M parameters) on the Wikipedia dataset. Following Chinchilla scaling principles(hoffmann2022training), we scale the number of training tokens proportionally with model size while maintaining a constant data mixture. We set number of training tokens to be Chinchilla-optimal in every case (20 \times number of parameters).

We compute losses and fit scaling laws separately for three evaluation subsets:

*   •Biology (forget set) 
*   •Biology-adjacent knowledge (Medicine/Chemistry/Environment - retain set) 
*   •General knowledge (Culture/Geography/History - retain set) 

Figure[8](https://arxiv.org/html/2512.05648v1#A2.F8 "Figure 8 ‣ B.1 Scaling Laws ‣ Appendix B Compute penalty ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") presents the fitted scaling laws. We use the standard approximation of 6\times\text{parameters}\times\text{tokens} to compute FLOPs. The fitted curves follow the form \ell=\alpha\times C^{-\beta} where \ell represents loss and C represents compute in FLOPs.

### B.2 Results

![Image 14: Refer to caption](https://arxiv.org/html/2512.05648v1/x14.png)

Figure 9: Compute penalty for three knowledge removal methods. Reported on three test subsets: biology (forget), biology-adjacent (retain), general knowledge (retain). For each method we convert final loss to the equivalent training compute budget on a baseline (no-filter) model (Figure[8](https://arxiv.org/html/2512.05648v1#A2.F8 "Figure 8 ‣ B.1 Scaling Laws ‣ Appendix B Compute penalty ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")). We then report relative delta between the resulting compute and the baseline.

Using these scaling laws, we convert the final losses from each knowledge removal method to equivalent compute budgets. We then report the relative difference compared to baseline training. These values can be interpreted as: ”achieving this performance with the baseline model on the original data mix would require X% less compute compared to the full baseline training.”

Figure[9](https://arxiv.org/html/2512.05648v1#A2.F9 "Figure 9 ‣ B.2 Results ‣ Appendix B Compute penalty ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") presents the compute penalty analysis across all three evaluation sets.

First, on biology (forget set) both data filtering (95% and 96% penalty) achieve substantial forgetting, with SGTM only slightly outperforming both of them (99% penalty). All methods require less than 5% of the original compute to reach similar biology performance. This high penalty might suggest data filtering could be practical in this setup. However, this reflects limitations of our experimental design: loss metrics serve only as proxies for actual capabilities, and frontier models’ high sample efficiency means they could potentially learn from very few examples that escape filtering, making filtering less viable option in practice.

Second, on biology-adjacent knowledge SGTM’s 38% penalty falls between weak filtering (7%) and strict filtering (67%). This demonstrates SGTM’s key advantage: achieving stronger biology forgetting while preserving some medicine/chemistry/environment knowledge that strict filtering removes entirely. Notably, even weak filtering shows a 7% penalty because removing biology data inherently degrades performance on related domains – an effect that would be more pronounced with perfect filtering.

Finally, on general knowledge SGTM incurs a 5% compute penalty, while both filtering approaches show negative penalties (-10% for weak, -15% for strict). These negative values occur because filtering methods operate at a fixed compute budget, replacing removed biology and biology-adjacent data with additional general knowledge examples. This effectively increases training on general topics beyond what the baseline receives.

The compute penalty analysis reveals important trade-offs between methods. While SGTM requires additional compute compared to data filtering on general tasks, it provides superior forgetting of target knowledge while maintaining better performance on related domains. The 6% penalty on general knowledge represents the computational cost of the gradient masking operations during training – a price that may be justified in certain scenarios by SGTM’s robustness advantages demonstrated elsewhere in this work.

## Appendix C Pre-ablation performance

![Image 15: Refer to caption](https://arxiv.org/html/2512.05648v1/x15.png)

Figure 10: SGTM can produce a capable pre-ablation model while still enabling strong post-ablation removal. We report compute penalty relative to a no-filter baseline across forget knowledge (Biology), related retain knowledge (Medicine / Chemistry / Environment), and general retain knowledge (Culture / Geography / History). Solid bars show pre-ablation performance and hatched bars show post-ablation performance. We compare three SGTM variants that differ by how aggressively they mask gradients from biology data: (i) full masking, (ii) joint-projection masking that leaves MLP down-projection parameters unconstrained, and (iii) joint embeddings and down-projections. Weak and strict data-filtering baselines are shown for reference (diamonds). The full masking setup excels post-ablation yet learns biology inefficiently pre-ablation (85% compute penalty). Relaxing gradient masking yields substantially improved biology capability before ablation with minimal retain-set degradation, while still recovering strong forgetting afterward. 

In the main paper we evaluate SGTM exclusively after ablation, focusing on label-noise robustness and adversarial fine-tuning resistance. However, selective gradient masking also enables a dual-model deployment strategy: a single training run produces (i) a knowledgeable model with full capabilities, and (ii) a safe variant with target knowledge removed. This paradigm could support workflows where general users receive the safe model, while vetted access is granted to the full-capability version. While this is not the primary goal of our study, we include an initial empirical assessment to illustrate its potential.

Ideally, this setup would satisfy two criteria:

1.   1.before ablation: strong performance on both retain and forget domains 
2.   2.after ablation: strong retention of general knowledge while effectively removing forget-domain capability. 

We therefore examine the full trade-off surface rather than focusing solely on post-ablation behavior.

Following Appendix[B](https://arxiv.org/html/2512.05648v1#A2 "Appendix B Compute penalty ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), we express performance differences in terms of compute penalty relative to the no-filter baseline. Lower values indicate more efficient learning of that capability.

Figure[10](https://arxiv.org/html/2512.05648v1#A3.F10 "Figure 10 ‣ Appendix C Pre-ablation performance ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") reports compute penalty on three evaluation subsets: forget (Biology), related retain knowledge (Medicine / Chemistry / Environment), and general retain knowledge (Culture / Geography / History). For SGTM variants both pre-ablation (solid) and post-ablation (hatched) values are shown.

We evaluate three SGTM hyperparameter configurations:

*   •Default configuration:d_{\text{forget}}=64; full masking; 10% confident retain sampling. This setup is used for all main-text Wikipedia results. 
*   •Joint down-projection:d_{\text{forget}}=128; MLP down-projection is unmasked (\theta_{\text{joint}}=\{W_{2},b_{2}\} as per Appendix[F](https://arxiv.org/html/2512.05648v1#A6 "Appendix F Parameter Split ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")); 10% confident retain sampling. 
*   •Joint embeddings:d_{\text{forget}}=256; MLP down-projections and embedding layers are unmasked; 10% confident retain (retaled), 25% confident retain (general). 

Weak and strict filtering baselines are included for comparison (diamonds). The no-filter baseline corresponds to zero compute penalty on all subsets.

Figure[10](https://arxiv.org/html/2512.05648v1#A3.F10 "Figure 10 ‣ Appendix C Pre-ablation performance ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") shows the results. The default configuration – which provides the strongest post-ablation retain/forget trade-off in the main text – also severely underperforms on biology pre-ablation, with a 85% compute penalty. That is, the pre-ablation model never becomes a truly capable “full-access” system on forget knowledge.

Relaxing gradient isolation, however, yields more balanced dual-model behavior. In particular, the joint embeddings configuration achieves substantially better pre-ablation biology performance: 30% compute penalty, with retain-set capabilities nearly unchanged from baseline (3% penalty on related knowledge and 1% improvement on general knowledge. After ablation, it regains strong forgetting with 88% compute penalty on biology, only modestly below data filtering with 95%.

These results suggest that SGTM can support dual-model construction when tuned for a more moderate trade-off: the pre-ablation model remains capable on the target domain, while the post-ablation model still removes most of that knowledge. Identifying optimal configurations for such deployments remains an open direction for future work.

## Appendix D Full Labeling Sensitivity and Specificity Analysis

![Image 16: Refer to caption](https://arxiv.org/html/2512.05648v1/x16.png)

(a) 

![Image 17: Refer to caption](https://arxiv.org/html/2512.05648v1/x17.png)

(b) 

Figure 11: SGTM vs Data Filtering across full input classifier performance range. We report the difference in loss (\Delta\ell=\ell_{\text{SGTM}}-\ell_{\text{filter}}) between SGTM and data filtering on the task of removing the knowledge of Spanish from a model trained on bilingual (English/Spanish) TinyStories dataset. We vary the quality of the input labeling with TPR and FPR of the input classifier.

In Section[4](https://arxiv.org/html/2512.05648v1#S4 "4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), we examined SGTM’s robustness to false negative labeling errors, where forget data is mislabeled as retain. Here we present a comprehensive analysis exploring the full range of classifier performance by varying both false positive rates (FPR) and true positive rates (TPR).

### D.1 Experimental Setup

We use the same synthetic bilingual TinyStories setup as Section[4](https://arxiv.org/html/2512.05648v1#S4 "4 Synthetic Dataset (TinyStories) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), with models trained on English (retain) and Spanish (forget) data. We simulate different classifier performances by artificially controlling TPR and FPR rates. For each (TPR, FPR) pair, we construct the training datasets as follows:

*   •\mathbf{D}_{\text{forget}}: Contains TPR fraction of actual forget data plus FPR fraction of retain data 
*   •\mathbf{D}_{\text{unlabeled}}: Contains (1-TPR) fraction of forget data plus (1-FPR) fraction of retain data 
*   •\mathbf{D}_{\text{retain}}: Fixed at 10% of ground truth retain data (unchanged across all experiments) 

For data filtering, which doesn’t train on \mathbf{D}_{\text{forget}}, false positives result in less training data overall. All models train for one epoch on their respective datasets, with compute budgets varying based on the amount of filtered data.

### D.2 Results

Figure[11](https://arxiv.org/html/2512.05648v1#A4.F11 "Figure 11 ‣ Appendix D Full Labeling Sensitivity and Specificity Analysis ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") presents the retain and forget loss differences between SGTM and data filtering (\Delta\ell=\ell_{\text{SGTM}}-\ell_{\text{filter}}) across the full spectrum of TPR and FPR values. This visualization captures both classifier quality and decision threshold choice – an ROC curve for any specific classifier can be overlaid on these heatmaps, with each point on the curve representing a different threshold setting. Note that the scales differ between the two subfigures to better visualize the patterns in each metric.

First, Figure[11](https://arxiv.org/html/2512.05648v1#A4.F11 "Figure 11 ‣ Appendix D Full Labeling Sensitivity and Specificity Analysis ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")(a) shows that SGTM consistently shows higher retain loss than data filtering across most of the parameter space, confirming that SGTM incurs a certain penalty in compute efficiency. With the penalty being most pronounced in the top-left region (high TPR, low FPR), the retain performance penalty is decreasing with the decrease in classifier performance.

Second, Figure[11](https://arxiv.org/html/2512.05648v1#A4.F11 "Figure 11 ‣ Appendix D Full Labeling Sensitivity and Specificity Analysis ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")(b) shows that outside of perfect recall (TPR=1), SGTM demonstrates a clear advantage over data filtering, achieving substantially higher forget loss across most classifier operating points. The advantage is particularly strong in regions with moderate TPR (0.4-0.8), where data filtering begins to fail due to leaked forget examples in the training data.

The results suggest that SGTM is particularly valuable when perfect classification is unattainable—a realistic scenario given the difficulty of identifying harmful content at scale. While data filtering can match SGTM’s forgetting performance only with perfect recall (TPR=1), achieving this in practice would involve a trade-off with removing useful data.

## Appendix E Robustness to Finetuning

![Image 18: Refer to caption](https://arxiv.org/html/2512.05648v1/x18.png)

(a) 

![Image 19: Refer to caption](https://arxiv.org/html/2512.05648v1/x19.png)

(b) 

Figure 12: SGTM remains robust to adversarial fine-tuning when controlling for retain-set performance. (a) We select undertrained weak- and strict-filtering checkpoints whose loss on general knowledge matches the final SGTM model, ensuring a fair robustness comparison. Stars indicate final checkpoints, and markers show selected undertrained starting points. (b) We perform adversarial fine-tuning using the same procedure as in Section[5.3](https://arxiv.org/html/2512.05648v1#S5.SS3 "5.3 Robustness to fine-tuning ‣ 5 Realistic Dataset (Wikipedia) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"). SGTM reaches the baseline forget loss in a similar number of steps as the weak filter model, while maintaining a lead over strict filtering for the first 100 steps. 

In Section[5.3](https://arxiv.org/html/2512.05648v1#S5.SS3 "5.3 Robustness to fine-tuning ‣ 5 Realistic Dataset (Wikipedia) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), we evaluated adversarial fine-tuning robustness using models trained to completion under Chinchilla-optimal compute budgets. SGTM required the same number of fine-tuning steps as strict filtering to reach the baseline forget loss. Since SGTM’s final checkpoint exhibits a slightly higher loss on general knowledge compared to both weak and strict data filtering, a natural concern is whether this robustness simply reflects a more degraded model.

To decouple robustness from overall capability degradation, we extend this analysis by controlling for general knowledge performance at the start of fine-tuning. Specifically, for each data filtering baseline we identify intermediate checkpoints whose loss on general knowledge matches that of the SGTM final checkpoint. Figure[12](https://arxiv.org/html/2512.05648v1#A5.F12 "Figure 12 ‣ Appendix E Robustness to Finetuning ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") shows the chosen starting points: 8500 training steps for the weak filter and 8000 steps for the strict filter, each showing slightly elevated retain loss relative to SGTM.

We then repeat adversarial fine-tuning using the same protocol from Section[5.3](https://arxiv.org/html/2512.05648v1#S5.SS3 "5.3 Robustness to fine-tuning ‣ 5 Realistic Dataset (Wikipedia) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), applied to these retain-matched baselines. As shown in Figure[12](https://arxiv.org/html/2512.05648v1#A5.F12 "Figure 12 ‣ Appendix E Robustness to Finetuning ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")(b), SGTM either matches the robustness of data filtering even when controlling for starting loss. SGTM reaches baseline forget-set performance in approximately the same number of steps as the weak filter, while preserving a consistent early-stage advantage over strict filtering during roughly the first 100 fine-tuning steps.

These results suggest that SGTM’s robustness cannot be attributed solely to reduced capability on the retain set. Instead, the fine-tuning behavior indicates a structural interference with re-learning the target knowledge, consistent with SGTM’s intended localization of forget-set updates into removable parameters.

Interestingly, SGTM’s forget-set loss initially drops faster than weak filtering before converging to an equivalent trajectory. This behavior implies that some aspects of biology knowledge may be erased from SGTM model in a more superficial manner and are thus more easily recovered. However, the persistence of higher loss deep into fine-tuning suggests that a substantial fraction of knowledge is removed deeply enough that recovery remains slow even under targeted adversarial updates.

## Appendix F Parameter Split

We consider a transformer block(vaswani2017attention) consisting of a multi-head attention and an MLP layer, with h attention heads, model dimension d, MLP dimension d_{\text{MLP}}, and per-head dimension d_{h}=d/h.

The trainable parameters in a multi-head attention block are 1 1 1 For notation simplicity we omit trainable parameters in normalization layers and treat them as \theta_{\text{joint}}.:

\displaystyle W_{QKV}^{(i)}\in\mathbb{R}^{3\times d\times d_{h}}(query, key, value for attention head i)
\displaystyle W_{O}\in\mathbb{R}^{d\times d},\quad b_{O}\in\mathbb{R}^{d}(output attention projection)
\displaystyle W_{1}\in\mathbb{R}^{d\times d_{\text{MLP}}},\quad b_{1}\in\mathbb{R}^{d_{\text{MLP}}}(First MLP layer)
\displaystyle W_{2}\in\mathbb{R}^{d_{\text{MLP}}\times d},\quad b_{2}\in\mathbb{R}^{d}(Second MLP layer)

We designate h_{\text{forget}} attention heads and d_{\text{forget}} MLP hidden units to be dedicated to the forget data, and the remaining attention heads and MLP units to the retain data. We splits weights and biases of the relevant linear layers into forget and retain segments.

We can now define non-overlapping sets of forget (\theta_{\text{forget}}) and retain (\theta_{\text{retain}}) parameters.

\displaystyle W_{1}\displaystyle=[W_{1}^{\text{forget}}\ W_{1}^{\text{retain}}];\quad W_{1}^{\text{forget}}\in\mathbb{R}^{d\times d_{\text{forget}}},\ W_{1}^{\text{retain}}\in\mathbb{R}^{d\times(d_{\text{MLP}}-d_{\text{forget}})}
\displaystyle W_{2}\displaystyle=\begin{bmatrix}W_{2}^{\text{forget}}\\
W_{2}^{\text{retain}}\end{bmatrix};\quad W_{2}^{\text{forget}}\in\mathbb{R}^{d_{\text{forget}}\times d},\ W_{2}^{\text{retain}}\in\mathbb{R}^{(d_{\text{MLP}}-d_{\text{forget}})\times d}
\displaystyle W_{O}\displaystyle=\begin{bmatrix}W_{O}^{\text{forget}}\\
W_{O}^{\text{retain}}\end{bmatrix};\quad W_{O}^{\text{forget}}\in\mathbb{R}^{(d_{h}*h_{\text{forget}})\times d},\ W_{2}^{\text{retain}}\in\mathbb{R}^{(d_{h}*(h-h_{\text{forget}}))\times d}
\displaystyle b_{1}\displaystyle=[b_{1}^{\text{forget}}\ b_{1}^{\text{retain}}];\quad b_{1}^{\text{forget}}\in\mathbb{R}^{d_{\text{forget}}},\ b_{1}^{\text{retain}}\in\mathbb{R}^{(d_{\text{MLP}}-d_{\text{forget}})}
\displaystyle\theta_{\text{forget}}\displaystyle=\{W_{1}^{\text{forget}},\ b_{1}^{\text{forget}},\ W_{2}^{\text{forget}},\ W_{O}^{\text{forget}},\ W_{QKV}^{(i)}\text{ for }i\in\{1,\ldots,h_{\text{forget}}\}\}
\displaystyle\theta_{\text{retain}}\displaystyle=\{W_{1}^{\text{retain}},\ b_{1}^{\text{retain}},\ W_{2}^{\text{retain}},\ b_{2},\ W_{O}^{\text{retain}},\ b_{O},\ W_{QKV}^{(i)}\text{ for }i\in\{h_{\text{forget}}+1,\ldots,h\}\}

## Appendix G Leakage Definition

Following the notation from Section[3.1](https://arxiv.org/html/2512.05648v1#S3.SS1 "3.1 Notation ‣ 3 Method ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") and Appendix[F](https://arxiv.org/html/2512.05648v1#A6 "Appendix F Parameter Split ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), we consider SGTM training process \mathcal{A}_{\text{SGTM}} with the dataset split into three subsets: \mathbf{D}_{\text{SGTM}}=\{\mathbf{D}_{\text{forget}},\ \mathbf{D}_{\text{retain}},\ \mathbf{D}_{\text{unlabeled}}\}. We denote subsets of \mathbf{D}_{\text{unlabeled}} with corresponding ground truth labels as \mathbf{D}^{\text{retain}\vphantom{\big|}}_{\text{unlabeled}} and \mathbf{D}^{\text{forget}\vphantom{\big|}}_{\text{unlabeled}} respectively:

\displaystyle\theta_{\text{SGTM}}\leftarrow\mathcal{A}_{\text{SGTM}}(\mathbf{D}_{\text{forget}},\mathbf{D}_{\text{retain}},\mathbf{D}_{\text{unlabeled}})
\displaystyle\mathbf{D}_{\text{unlabeled}}=\mathbf{D}^{\text{retain}\vphantom{\big|}}_{\text{unlabeled}}\quad\cup\quad\mathbf{D}^{\text{forget}\vphantom{\big|}}_{\text{unlabeled}}
\displaystyle\mathbf{D}_{\text{forget}},\ \mathbf{D}^{\text{forget}\vphantom{\big|}}_{\text{unlabeled}}\ \sim\mathcal{D}_{\text{forget}}
\displaystyle\mathbf{D}_{\text{retain}},\ \mathbf{D}^{\text{retain}\vphantom{\big|}}_{\text{unlabeled}}\ \sim\mathcal{D}_{\text{retain}}

Similarly, we consider standard training process \mathcal{A}_{\text{standard}} on a dataset \mathbf{D}_{\text{standard}}. While standard training does not distinguish between forget and retain samples during training, we denote parts of \mathbf{D}_{\text{standard}} coming from respective data distributions as \mathbf{D}^{\text{retain}\vphantom{\big|}}_{\text{standard}} and \mathbf{D}^{\text{forget}\vphantom{\big|}}_{\text{standard}}:

\displaystyle\theta_{\text{standard}}\leftarrow\mathcal{A}_{\text{standard}}(\mathbf{D}_{\text{standard}})
\displaystyle\mathbf{D}_{\text{standard}}=\mathbf{D}^{\text{retain}\vphantom{\big|}}_{\text{standard}}\quad\cup\quad\mathbf{D}^{\text{forget}\vphantom{\big|}}_{\text{standard}}
\displaystyle\mathbf{D}^{\text{forget}\vphantom{\big|}}_{\text{standard}}\ \sim\mathcal{D}_{\text{forget}}
\displaystyle\mathbf{D}^{\text{retain}\vphantom{\big|}}_{\text{standard}}\ \sim\mathcal{D}_{\text{retain}}

For any given SGTM run we then consider corresponding standard training run with the same retain training data, while varying the size of the forget training data to achieve the same forget loss \ell(\theta,\mathcal{D}_{\text{forget}}) as the SGTM run. Below \mathbf{D}_{\text{forget}},\mathbf{D}_{\text{retain}},\mathbf{D}_{\text{unlabeled}}^{*} are used to refer to SGTM training data, while \mathbf{D}_{\text{standard}}^{*} refers to standard training run data, and \left|\mathbf{D}_{*}\right| refers to the number of samples is a given dataset.

###### Definition G.1.

_Leakage_ of an SGTM run is then defined as the size of the forget dataset used in standard training required to achieve the equivalent forget loss, normalized by the number of unlabeled forget samples seen by the SGTM run.

\textit{Leakage}(\theta_{\text{SGTM}}):=\frac{\left|\mathbf{D}^{\text{forget}\vphantom{\big|}}_{\text{standard}}\right|}{\left|\mathbf{D}^{\text{forget}\vphantom{\big|}}_{\text{unlabeled}}\right|}(1)

Such that

\displaystyle\theta_{\text{SGTM}}\leftarrow\mathcal{A}_{\text{SGTM}}(\mathbf{D}_{\text{forget}},\ \mathbf{D}_{\text{retain}},\ \mathbf{D}^{\text{retain}\vphantom{\big|}}_{\text{unlabeled}}\ \cup\ \mathbf{D}^{\text{forget}\vphantom{\big|}}_{\text{unlabeled}})
\displaystyle\theta_{\text{standard}}\leftarrow\mathcal{A}_{\text{standard}}(\mathbf{D}^{\text{retain}\vphantom{\big|}}_{\text{standard}}\ \cup\ \mathbf{D}^{\text{forget}\vphantom{\big|}}_{\text{standard}})
\displaystyle\mathbf{D}^{\text{retain}\vphantom{\big|}}_{\text{standard}}=\mathbf{D}_{\text{forget}}\ \cup\ \mathbf{D}^{\text{retain}\vphantom{\big|}}_{\text{unlabeled}}\displaystyle(\text{Same retain data})
\displaystyle\ell(\theta_{\text{SGTM}},\mathcal{D}_{\text{forget}})=\ell(\theta_{\text{standard}},\mathcal{D}_{\text{forget}})\displaystyle(\text{Same forget loss})

In practice, finding the exact \mathbf{D}^{\text{forget}\vphantom{\big|}}_{\text{standard}} so that losses match exactly is computationally intractable. We instead compute losses for a range of increasingly large \mathbf{D}^{\text{forget}\vphantom{\big|}}_{\text{standard}} and then perform a linear interpolation between the two closest baseline models.

## Appendix H Additional Forget Categories

![Image 20: Refer to caption](https://arxiv.org/html/2512.05648v1/x20.png)

(a) 

![Image 21: Refer to caption](https://arxiv.org/html/2512.05648v1/x21.png)

(b) 

Figure 13: Retain/forget trade-off when removing “Military and Warfare” category knowledge from a model trained on Wikipedia.

To assess the robustness of our findings beyond the biology domain, we repeat the Wikipedia experiment from Section[5](https://arxiv.org/html/2512.05648v1#S5 "5 Realistic Dataset (Wikipedia) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") using an alternative forget category. Instead of removing biology knowledge, we now designate “Military and Warfare” as the forget category. As before, we evaluate performance on (i) the target domain, (ii) related categories (History, Biography, Transportation), and (iii) general knowledge.

Figure[13](https://arxiv.org/html/2512.05648v1#A8.F13 "Figure 13 ‣ Appendix H Additional Forget Categories ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs") shows that the resulting trends closely mirror those observed in our biology experiments. SGTM consistently achieves a better retain/forget trade-off than both weak and strict filtering: at any fixed retain loss, SGTM reaches higher loss on the Military/Warfare forget set. As in Section[5](https://arxiv.org/html/2512.05648v1#S5 "5 Realistic Dataset (Wikipedia) Results ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"), SGTM ends with higher retain loss, reflecting incurred compute efficiency penalty – but provides stronger forgetting than either filtering strategy.

## Appendix I Tinystories Experimental Details

### I.1 Training Hyperparameters

Hyperparameter 64M 34M 18M 8M
Parameters 63,855k 33,732k 17,781k 7,736k
Layers 12 8 6 6
Model dimenstion d 512 384 256 128
MLP dimension d_{\text{MLP}}2048 1536 1024 512
Warmup steps 1000 1000 1000 500
Training steps 33120 17484 9204 3989
Vocabulary size 50257
Attention heads h 32
Learning rate 5e-3
Tied embeddings True
LR Schedule Cosine annealing with warmup
Batch size 128
Context size 512
Tokenizer gpt-2
Optimizer AdamW
Weight decay 0.1
\beta_{1}0.9
\beta_{2}0.95

Table 2: Training hyperparameters for synthetic bilingual experiments.

All relevant training hyperparameters are listed in Table [2](https://arxiv.org/html/2512.05648v1#A9.T2 "Table 2 ‣ I.1 Training Hyperparameters ‣ Appendix I Tinystories Experimental Details ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"). We aim to roughly double the size of our largest model compared to the model from eldan2023tinystories (64M vs. 33M) to account for the extra Spanish data.

### I.2 Logit Calibration

To avoid artificially low token probabilities after model ablation, potentially leading to unreasonably high loss values, we perform port-training logit calibration. We train a simple logit bias layer (one parameter per logit, 50257 in total) minimizing the combination of the forget and retain loss: \ell=\ell_{\text{forget}}+\alpha\ell_{\text{retain}}. It’s important to note that we perform the calibration assuming full access to both forget and retain data, including setups with the full data filtering, where none of the forget data was used during training.

The goal of this step is to stabilise the forget loss post-ablation and recover loss spikes caused by extremely low probabilities (potentially worse than random) of the relevant tokens. As such we set \alpha=100 to make sure we recover the forget loss where possible, but not at the expense of the retain loss.

### I.3 Dataset

We use the original TinyStories dataset(eldan2023tinystories) in English (466M tokens), and create a Spanish translation of the entire dataset 718M tokens. Note that the Spanish dataset contains more tokens than the English version because the GPT-2 tokenizer was primarily trained on English text, making it less efficient for encoding Spanish. Our final bilingual training dataset consists of both the original English stories and their Spanish translations, totaling 1.2B tokens. In both English and Spanish and additional 1% of data is set aside as a test set (12M tokens).

We consider each story to be an individual training example, truncating and padding the text when necessary.

We translate the Tinystories dataset(eldan2023tinystories) into Spanish using claude-3-haiku-20240307 with the following prompt:

Translate the following short story into Spanish. Keep the same tone, style, and meaning. The translation should be natural and fluent Spanish, appropriate for children.English story: story Spanish translation:

Below is an example of an original story in English:

There was a little girl with dark hair. Her name was Joy. She lived in a big house with her parents. One day, Joy was playing outside in her garden. Suddenly, she felt something on her leg - something pinching her. It was a big, black bug!Joy screamed and tried to get away, but the bug kept following her. She tried to run and hide, but it was too quick.Joy’s parents heard her cries and came running. They used a stick to help her get rid of the bug. After the bug was gone, they hugged Joy and told her everything would be alright.When the bug was gone, Joy felt relieved and happy. She went back to playing in the garden, making sure she didn’t step on any more bugs.

And the corresponding Spanish version:

Había una niña pequeña con el cabello oscuro. Su nombre era Alegría. Vivía en una casa grande con sus padres. Un día, Alegría estaba jugando afuera en su jardín. De repente, sintió algo en su pierna - algo que le estaba pellizcando. ¡Era un bicho grande y negro!Alegría gritó e intentó alejarse, pero el bicho seguía persiguiéndola. Trató de correr y esconderse, pero era demasiado rápido.Los padres de Alegría escucharon sus gritos y corrieron hacia ella. Usaron un palo para ayudarla a deshacerse del bicho. Después de que el bicho se fue, abrazaron a Alegría y le dijeron que todo estaría bien.Cuando el bicho se fue, Alegría se sintió aliviada y feliz. Volvió a jugar en el jardín, asegurándose de no pisar más bichos.

## Appendix J Wikipedia Experimental Details

### J.1 Training Hyperparameters

Hyperparameter 254M 125M 64M 34M
Parameters 254,054k 124,462k 64,117k 33,929k
Layers 16 12 12 8
Model dimenstion d 1024 768 512 384
MLP dimension d_{\text{MLP}}4096 3072 2048 1536
Warmup steps 1000 1000 500 500
Batch size 512 256 256 128
Training steps 9689 9491 4887 5169
Learning rate 6e-3
LR Schedule Cosine annealing with warmup
Tied embeddings True
Attention heads h 32
Vocabulary size 50257
Context size 1024
Tokenizer gpt-2
Optimizer AdamW
Weight decay 0.1
\beta_{1}0.9
\beta_{2}0.95

Table 3: Training hyperparameters for Wikipedia model.

All relevant training hyperparameters are listed in Table [3](https://arxiv.org/html/2512.05648v1#A10.T3 "Table 3 ‣ J.1 Training Hyperparameters ‣ Appendix J Wikipedia Experimental Details ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs").

For logit calibration we follow the procedure idential to the one described for the TinyStories setup (Appendix[I.2](https://arxiv.org/html/2512.05648v1#A9.SS2 "I.2 Logit Calibration ‣ Appendix I Tinystories Experimental Details ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs")).

### J.2 RMU

Hyperparameter Value
Steps 250
\alpha 100
Steering coefficient 20
Batch size 4
Layer to unlearn 7
Update layers 5, 6, 7
Update parameters MLP weights and biases (W_{1},\ b_{1},\ W_{2},\ b_{2})

Table 4: RMU hyperparameters.

Using the implementation provided by li2024wmdp, we apply RMU to the baseline model trained on the full dataset with no filtering. Hyperparameters are reported in Table[4](https://arxiv.org/html/2512.05648v1#A10.T4 "Table 4 ‣ J.2 RMU ‣ Appendix J Wikipedia Experimental Details ‣ Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs").

## Appendix K Articletopic taxonomy

Culture Geography History and Society STEM
Biography Geographical Business and economics STEM*
Biography*Regions Education Biology
Women*Africa History Chemistry
Food and drink Africa*Military and warfare Computing
Internet culture Central Africa Politics and government Earth and environment
Linguistics Eastern Africa Society Engineering
Literature Northern Africa Transportation Libraries and Information
Media Southern Africa Mathematics
Media*Western Africa Medicine and Health
Books Americas Physics
Entertainment Central America Space
Films North America Technology
Music South America
Radio Asia
Software Asia*
Television Central Asia
Video games East Asia
Performing arts North Asia
Philosophy and religion South Asia
Sports Southeast Asia
Visual arts West Asia
Visual arts*Europe
Comics and Anime Eastern Europe
Fashion Northern Europe
Southern Europe
Western Europe
Oceania

Table 5: Wikipedia articletopic taxonomy
