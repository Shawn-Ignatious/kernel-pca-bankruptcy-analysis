# kernel-pca-bankruptcy-analysis for Exploring Corporate Bankruptcy Structure

**STTN623 — Nonlinear Dimension Reduction | Part B Project**

A comparison of **RBF Kernel PCA** and **Linear PCA** for exploring the structure
of corporate bankruptcy in the Taiwanese Bankruptcy Prediction dataset.

Based on Schölkopf, Smola & Müller (1996) — *"Nonlinear Component Analysis as a
Kernel Eigenvalue Problem."*

---

## Authors

| Name & Surname | Student Number |
|---|---|
| Shawn Cloete | 41142276 |
| Mlungisi Mthethwa | 45984530 |
| Luyanda Senotlela | 56471955 |

---

## Project Overview

Linear PCA finds orthogonal linear directions of maximum variance — it fails
when the underlying structure is **nonlinear** (e.g. two concentric rings become
one overlapping band on a single PC). **Kernel PCA** solves this by mapping data
into a high-dimensional feature space where structure becomes linearly
separable, without ever computing that mapping explicitly (the *kernel trick*).

This project asks:

> **How does the 2D representation of Taiwanese firms' financial ratios produced
> by RBF kernel PCA differ from that of linear PCA?**

Sub-questions:

1. How well does each method preserve the structure of the original data?
2. How sensitive are the conclusions to the RBF `gamma` parameter?

Both methods are **unsupervised** — the bankruptcy label is used **only
post-hoc** to evaluate the quality of the embeddings.

---

## Dataset

- **Source:** [UCI ML Repository — Taiwanese Bankruptcy Prediction (ID 572)](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)
- **Collected from:** Taiwan Economic Journal, 1999–2009
- **Observations:** 6,819 firm-year records
- **Predictors:** 95 financial ratios (94 after removing a constant column)
- **Target:** `Bankrupt` (binary) — **220 bankrupt** vs **6,599 solvent** (3.23%)

### Handling class imbalance

Because solvent firms outnumber bankrupt firms by ~30:1, the full dataset is
**downsampled to a balanced 440-firm subsample** (220 per class) for the
exploratory visual comparison. This prevents the majority class from dominating
the unsupervised variance structure.

> Downsampling is appropriate for **exploration** but not for estimating the
> true bankruptcy rate or building a predictive model.

---

## 🔧 Preprocessing Pipeline

| Step | What it does | Why |
|---|---|---|
| **1. Drop constants** | Removes `Net Income Flag` | Zero information, undefined std |
| **2. Winsorise** | Cap continuous vars at 1st / 99th percentiles | Median \|skew\| drops **11.36 → 1.66** |
| **3. Drop zero-variance** | Removes any column constant *within the balanced subsample* | Prevents NaN correlations |
| **4. Standardise** | Z-score: `z = (x − μ) / σ` | Both PCA and RBF are **not scale-invariant** |

Standardising is critical for RBF kernel PCA because the kernel depends directly
on squared Euclidean distances — unscaled features would dominate similarity.

---

## Methodology

### Kernel PCA formulation

Given data `x₁, …, xₙ` in ℝᵈ and a nonlinear map Φ into a feature space H:

$$
\Sigma_H = \frac{1}{n}\sum_{i=1}^{n}\Phi(x_i)\Phi(x_i)^\top, \qquad \Sigma_H v = \lambda v
$$

Solutions lie in the span of the mapped data, so we solve the **kernel
eigenvalue problem** instead:

$$
\tilde{K}\alpha = \lambda \alpha
$$

where `K_ij = k(x_i, x_j)` is the Gram matrix. The embedding for a test point is:

$$
\text{PC}_k(x) = \sum_{i=1}^{n} \alpha_i^{(k)} \, k(x_i, x)
$$

### Kernel choice

**Radial Basis Function (RBF / Gaussian):**

$$
k(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)
$$

Chosen for its ability to capture **local nonlinear similarity** between firms.

### Gamma selection

A **feature-count heuristic** is used rather than data-driven tuning (which
would leak label information into an unsupervised method):

$$
\gamma = \frac{1}{d} = \frac{1}{94} \approx 0.0106
$$

Sensitivity to `gamma` is checked visually across `{0.001, 0.005, 0.0106, 0.05, 0.1}`.

---

## Results

### Explained variance / kernel power

| Method | Component 1 | Component 2 | Total |
|---|---:|---:|---:|
| **Linear PCA** | 25.40% | 7.50% | **32.90%** |
| **RBF Kernel PCA** | 50.59%* | 49.41%* | — |

\* *Kernel percentages are **relative power within the two retained components**,
not explained variance in the original variables. They sum to 100% by construction
and describe the balance between components, not the amount of kernel structure
captured by the 2D embedding.*

### Class-separation diagnostic (silhouette, using bankruptcy label)

| Method | Silhouette Score |
|---|---:|
| **Linear PCA** | **0.2887** |
| RBF Kernel PCA | 0.2400 |

> A higher silhouette score means stronger alignment with the bankruptcy
> labels. **Linear PCA edged out Kernel PCA here** — despite Kernel PCA using a
> more flexible representation.

### Visual comparison

| Linear PCA | RBF Kernel PCA (γ ≈ 0.0106) |
|---|---|
| Dense central mass, weak PC1 ordering (bankrupt → negative, solvent → positive) | Broader spread; solvent mostly in upper half, bankrupt mostly lower half |
| A few extreme bankrupt firms far left (PC1 ≈ −15) | Bounded kernel limits outlier influence |
| Moderate class overlap in central region | Mixed transition zone on the left; compact bankrupt cluster at lower-right |

---

## Key Findings

1. **Linear PCA already captured a moderate global structure** — the first two
   PCs split bankrupt and solvent firms along PC1 with a silhouette of 0.29.
2. **Kernel PCA revealed nonlinear structure but not better class separation.**
   The RBF embedding spread firms across the plane, but silhouette dropped to 0.24.
3. **Outliers matter.** Linear PCA is stretched by a handful of extreme firms;
   the bounded RBF kernel reduces that distortion.
4. **Gamma controls geometry.** Small `gamma` → smooth, arch-shaped embeddings
   (a known artefact of local kernel methods — Diaconis et al., 2008). Large
   `gamma` → fragmented, over-local embeddings.
5. **Downsampling seed is not driving the conclusion.** Across 10 random
   balanced subsamples, Linear PCA consistently produced higher silhouette
   scores than Kernel PCA (see `STTN623_Project_code.py`, section 10).

### Bottom line

> On this dataset, **kernel PCA did not outperform linear PCA for revealing
> bankruptcy structure**, even though it produced a visually richer embedding.
> The most likely explanations are (a) the underlying class boundary is largely
> linear in the standardised feature space, and (b) a bounded kernel with a
> heuristic `gamma` may over-smooth similarity between firms.

