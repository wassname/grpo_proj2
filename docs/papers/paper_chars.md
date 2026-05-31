# CHaRS: Concept Heterogeneity-aware Representation Steering

Source: https://arxiv.org/html/2603.02237v1 (fetched 2026-05-26)
Epistemic note: no released code; trust signals = the paper alone. Treat
findings as plausible but unvalidated by external use.

## Abstract

CHaRS controls LLM behavior through representation steering. Rather than
applying uniform shifts across embedding space, CHaRS models representations
as Gaussian mixture models and formulates steering as an optimal transport
problem between semantic clusters. Yields context-dependent steering vectors
that vary smoothly across the representation manifold, claimed to outperform
global steering baselines.

## 1. Introduction

Traditional steering: single direction via difference-in-means over
contrastive pairs. Assumes homogeneous concept representation in embedding
space.

Authors' claim: LLM representations are non-homogeneous, clustered,
context-dependent — global steering becomes brittle.

### Key Contributions

1. Generalization to multimodality: extends steering from unimodal Gaussian to
   GMMs via Mixture Wasserstein distance, formulated as discrete OT between
   semantic clusters.
2. Input-adaptive steering: cluster-level transport plans produce
   context-sensitive control where directions vary smoothly across the manifold.
3. Spectral factorization: Principal Component Thresholding (PCT) reveals
   inherent low-rank structure (rank ≤ 2K−2).

## 2. Background

### 2.1 Optimal Transport Framework

p-Wasserstein distance:

$$W_p(\mu, \nu) = \left(\inf_{\pi \in \Pi(\mu,\nu)} \int \|\mathbf{x} - \mathbf{y}\|_p\, d\pi(\mathbf{x}, \mathbf{y})\right)^{1/p}$$

### 2.2 Gaussian OT and Representation Steering

For $\mu = \mathcal{N}(m_1, \Sigma_1)$ and $\nu = \mathcal{N}(m_2, \Sigma_2)$:

$$W_2^2(\mu, \nu) = \|m_1 - m_2\|_2^2 + d_B^2(\Sigma_1, \Sigma_2)$$

OT map is affine: $T(\mathbf{x}) = \mathbf{m}_2 + \mathbf{A}(\mathbf{x} - \mathbf{m}_1)$.
Under identical covariance, reduces to pure translation = difference-in-means.

### 2.3 Gaussian Mixture Wasserstein Distance

$$\mu = \sum_{k=1}^K p_k \mathcal{N}(\mathbf{a}_k, \Sigma_k),\quad \nu = \sum_{l=1}^L q_l \mathcal{N}(\mathbf{b}_l, \Gamma_l)$$

Mixture Wasserstein:

$$MW_2^2(\mu, \nu) = \min_{\gamma \in \Gamma(p,q)} \sum_{k,l} \gamma_{kl}\, W_2^2(\mathcal{N}(a_k, \Sigma_k), \mathcal{N}(b_l, \Gamma_l))$$

Tractable discrete OT between components.

## 3. CHaRS

### 3.1 Barycentric Projection

$$\hat{T}(\mathbf{x}) := \mathbb{E}_\pi[\mathbf{y}|\mathbf{x}]$$

For GMM-OT:

$$\hat{T}(\mathbf{x}) = \sum_{k,l} p(k|\mathbf{x}) \cdot \frac{\gamma^*_{kl}}{p_k} \cdot T_{kl}(\mathbf{x})$$

### 3.2 Clustering-based Steering

- k-means on activations → centroids $a_i$, $b_j$.
- Cluster matching via entropy-regularized OT (Sinkhorn):

$$\mathbf{P}^* = \arg\min_{\mathbf{P} \in \Pi(w_A, w_B)} \langle \mathbf{P}, \mathbf{C} \rangle + \lambda H(\mathbf{P}),\quad C_{ij} = \|a_i - b_j\|_2^2$$

- Kernel-based gating (RBF, σ = median centroid distance):

$$\hat{p}(i|\mathbf{x}) = \frac{p_i\, k(\mathbf{x}, a_i)}{\sum_m p_m\, k(\mathbf{x}, a_m)}$$

- Isotropic-covariance simplification: $T_{ij}(\mathbf{x}) = \mathbf{x} + \mathbf{v}_{ij}$ where $\mathbf{v}_{ij} = b_j - a_i$.
- Final steering (Definition 3.1):

$$\hat{T}_\alpha(\mathbf{x}) = \mathbf{x} + \alpha \hat{\mathbf{v}}(\mathbf{x})$$
$$\hat{\mathbf{v}}(\mathbf{x}) = \sum_{i,j} \frac{P^*_{ij}\, k(\mathbf{x}, a_i)}{\sum_{p,q} P^*_{pq}\, k(\mathbf{x}, a_p)}\, \mathbf{v}_{ij}$$

### 3.3 Principal Component Thresholding

Weighted covariance of local shifts:

$$\Sigma_{total} = \sum_{i,j} P_{ij}(\mathbf{v}_{ij} - \bar{\mathbf{v}})(\mathbf{v}_{ij} - \bar{\mathbf{v}})^T = \mathbf{U}\Lambda\mathbf{U}^T$$

Rank bound: $\text{rank}(\Sigma_{total}) \le 2K - 2$.

**CHaRS-PCT** keeps top $L$ components:

$$\tilde{\mathbf{v}}(\mathbf{x}) = \bar{\mathbf{v}} + \sum_{k \in [L]} \hat{\alpha}_k(\mathbf{x})\, \mathbf{u}_k$$

## 4. Experiments

### 4.1 Jailbreaking

AdvBench 80/20 train/eval. Examples (ASR):

| Model | ActAdd | CHaRS | CHaRS-PCT |
|---|---|---|---|
| Gemma2-9B | 91.35% | 98.08% | 98.08% |
| Llama3.1-8B | 95.19% | 98.08% | 99.04% |
| Qwen2.5-7B | 91.35% | 95.19% | 93.27% |

### 4.2 Toxicity Mitigation

Llama3-8B (lower = less toxic):

| Method | Classifier | 0-shot |
|---|---|---|
| Linear-AcT | 1.93% | 7.73% |
| CHaRS | 1.23% | 4.80% |
| CHaRS-PCT | 1.17% | 4.47% |

Authors note PCT often outperforms CHaRS in sequential settings — implicit
regularization.

### 4.3 Image Style Control

FLUX.1 + 512 COCO captions w/ style tags. Pareto frontier improves over
Linear-Act.

## 5. Ablations

- K > 1 consistently > K=1. Optimal K model-dependent (10-15).
- 100% variance with 2(K−1) PCs (matches theoretical bound).
- Component count for PCT requires tuning.

## 6. Concluding Remarks

Generalizes diff-in-means to multimodal distributions. Limitations: isotropic
covariances, k-means. Future: anisotropic mixtures, feature weighting.

---

## Note for our use case (projected_grpo)

CHaRS does **activation** steering (forward-pass). We do **gradient** projection
(backward-pass). The analog:
- Source distribution = clean-rollout gradients
- Target distribution = hack-rollout gradients
- Diff-in-means = `mean(g_hack) - mean(g_clean)` (current v_hack)
- CHaRS analog = cluster hack-grad and clean-grad into k clusters each,
  Sinkhorn-match, derive a per-input transport map.

Simpler step (PCT-like) for our setting: stack per-pair diff vectors,
SVD/PCA, keep top-L singular directions, project gradient out of their span.
This is what user proposed. Theoretical rank bound for K clusters is 2K-2;
with 12 pairs treated as 12 clusters that's 22, but L<<22 is fine in practice.
