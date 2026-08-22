# PCA + Autoencoder (AE-PCA) Experiment Plan

## 1. Objective

Evaluate whether augmenting pretrained embeddings with learned nonlinear autoencoder features before PCA improves dimensionality reduction performance.

The core hypothesis is:

> Learned nonlinear features can expose structure that ordinary PCA cannot access directly, while retaining the original embedding preserves useful linear/global structure.

The proposed method is called **Autoencoder-Augmented PCA (AE-PCA)**.

---

## 2. Core Method

Given an input embedding

\[
x \in \mathbb{R}^{D},
\]

train an autoencoder:

\[
z = f_\theta(x), \qquad z \in \mathbb{R}^{d_a}
\]

\[
\hat{x} = g_\phi(z).
\]

The encoder \(f_\theta\) must contain nonlinear operations so that \(z\) is a nonlinear transformation of \(x\).

Augment the original embedding with the learned nonlinear representation:

\[
\tilde{x}
=

[x; f_\theta(x)]
\in
\mathbb{R}^{D+d_a}.
\]

Apply PCA to the augmented representation:

\[
y
=

\operatorname{PCA}_d(\tilde{x}),
\]

where the final output dimension \(d\) is fixed across all methods.

Therefore:

\[
\boxed{
x
\rightarrow
f_\theta(x)
\rightarrow
[x;f_\theta(x)]
\rightarrow
\operatorname{PCA}_d
\rightarrow
y
}
\]

---

## 3. Mathematical Formulation

For a dataset

\[
X = \{x_1,\ldots,x_N\},
\qquad
x_i \in \mathbb{R}^{D},
\]

train an autoencoder using reconstruction loss:

\[
\mathcal{L}_{AE}
=

\frac{1}{N}
\sum_{i=1}^{N}
\left\|
x_i-g_\phi(f_\theta(x_i))
\right\|_2^2.
\]

After training:

\[
z_i=f_\theta(x_i).
\]

Construct:

\[
\tilde{x}_i
=

\begin{bmatrix}
x_i\\
z_i
\end{bmatrix}.
\]

Let the centered augmented data matrix be

\[
\tilde{X}_c
=

\tilde{X}
-

\mathbf{1}\mu^T.
\]

Compute covariance:

\[
C
=

\frac{1}{N}
\tilde{X}_c^T\tilde{X}_c.
\]

Let

\[
W_d=[w_1,\ldots,w_d]
\]

contain the top \(d\) eigenvectors of \(C\).

The final representation is:

\[
y_i=W_d^T(\tilde{x}_i-\mu).
\]

Because

\[
\tilde{x}
=

\begin{bmatrix}
x\\
f_\theta(x)
\end{bmatrix},
\]

each PCA component can be written as:

\[
y_j(x)
=

w_{x,j}^Tx +
w_{AE,j}^Tf_\theta(x).
\]

Thus AE-PCA is linear in the augmented feature space but nonlinear with respect to the original embedding \(x\).

---

## 4. Interpretation

AE-PCA should be interpreted as:

> Linear PCA performed over a feature space containing both the original embedding coordinates and learned nonlinear coordinates.

The autoencoder does not create new information in the strict information-theoretic sense because \(f_\theta(x)\) is deterministic from \(x\). Instead, it creates a learned nonlinear coordinate system in which certain nonlinear relationships may become accessible to a subsequent linear projection.

Ordinary PCA:

\[
x
\rightarrow
\operatorname{PCA}
\rightarrow
y.
\]

AE-PCA:

\[
x
\rightarrow
[x,f_\theta(x)]
\rightarrow
\operatorname{PCA}
\rightarrow
y.
\]

---

## 5. Primary Hypothesis

The primary hypothesis is:

\[
\boxed{
\operatorname{AE\text{-}PCA}_d

>

\operatorname{PCA}_d
}
\]

for downstream representation quality at the same final dimension \(d\).

The method should be considered successful only if the improvement is demonstrated empirically.

Do not assume AE-PCA will outperform PCA.

---

## 6. Required Baselines

Compare at least the following methods.

### Baseline A: Original PCA

\[
x
\rightarrow
\operatorname{PCA}_d
\]

This is the primary baseline.

### Baseline B: Autoencoder

\[
x
\rightarrow
f_\theta(x)
\]

with

\[
\dim(f_\theta(x))=d.
\]

This tests whether the nonlinear AE representation itself is better than PCA.

### Proposed: AE-PCA

\[
x
\rightarrow
[x;f_\theta(x)]
\rightarrow
\operatorname{PCA}_d.
\]

### Optional Baseline D: Kernel PCA

\[
x
\rightarrow
\operatorname{KPCA}_d.
\]

This provides a classical nonlinear dimensionality-reduction comparison.

---

## 7. Critical Experimental Control

All compared methods must have the **same final dimensionality**.

For example:

\[
d\in\{96,192,384\}.
\]

Do not compare:

- PCA-96 against AE-384
- PCA-96 against AE-PCA-512
- or any other unequal final representation sizes.

The downstream evaluator must receive vectors of exactly the same dimensionality for each method.

---

## 8. Important AE Configuration

The AE latent dimension \(d_a\) is an experimental hyperparameter and does not have to equal the final PCA dimension \(d\).

Example:

\[
D=768,
\qquad
d_a=384,
\qquad
d=96.
\]

Then:

\[
x\in\mathbb{R}^{768}
\]

\[
f_\theta(x)\in\mathbb{R}^{384}
\]

\[
[x;f_\theta(x)]\in\mathbb{R}^{1152}
\]

\[
\operatorname{PCA}_{96}([x;f_\theta(x)])
\in\mathbb{R}^{96}.
\]

The latent dimension should be treated as a hyperparameter and reported explicitly.

---

## 9. Data Leakage Prevention

The AE and PCA must be fitted only on the appropriate training data.

For a train/test or train/evaluation setup:

1. Train AE using training embeddings only.
2. Generate AE features for training embeddings.
3. Fit PCA using augmented training embeddings only.
4. For an unseen sample \(x_*\):
    - compute \(z__=f\_\theta(x__)\),
    - construct \([x__;z__]\),
    - apply the already-fitted PCA transformation.

Do not refit the AE or PCA using evaluation/test samples.

If the downstream benchmark provides only a single unsupervised dataset split, document the exact fitting protocol and avoid using evaluation labels during representation learning.

---

## 10. Scaling / Standardization

The original embedding and AE feature blocks may have different scales.

Do not assume that raw concatenation is appropriate.

Evaluate at least:

### Variant A: Raw concatenation

\[
\tilde{x}=[x;z].
\]

### Variant B: Block-normalized concatenation

Normalize the two blocks before concatenation:

\[
\tilde{x}
=

[\operatorname{norm}(x);
\lambda\operatorname{norm}(z)].
\]

The weighting parameter \(\lambda\) controls the relative contribution of the nonlinear features.

If block normalization or weighting is used, it must be fitted/tuned using training data only.

This is important because PCA is variance-sensitive. A feature block with larger numerical variance can dominate the principal components.

---

## 11. Downstream Evaluation

The primary evaluation should be the same downstream task used for the existing embedding-compression experiments.

For clustering:

\[
\text{embedding}
\rightarrow
\text{dimensionality reduction}
\rightarrow
\text{clustering metric}.
\]

Use the same clustering algorithm, metric, dataset, random seeds, and evaluation protocol for every method.

Primary comparison:

\[
\operatorname{PCA}_d
\quad\text{vs}\quad
\operatorname{AE}_d
\quad\text{vs}\quad
\operatorname{AE\text{-}PCA}_d.
\]

Optional:

\[
\operatorname{KPCA}_d.
\]

---

## 12. Ablation Studies

At minimum, investigate:

### A. AE latent dimension

\[
d_a\in\{96,192,384\}
\]

or another justified range.

### B. Final dimension

\[
d\in\{96,192,384\}.
\]

### C. Concatenation weighting

Compare:

\[
[x;z]
\]

against appropriately normalized/weighted variants.

### D. AE architecture

At least compare a linear AE against a nonlinear AE if feasible.

This is important.

A linear AE provides a useful control:

\[
f_\theta(x)=Wx+b.
\]

If linear AE-PCA performs similarly to nonlinear AE-PCA, then the observed improvement may come primarily from feature expansion/reweighting rather than nonlinear structure.

---

## 13. Key Diagnostic Experiment: Linear AE vs Nonlinear AE

This experiment directly tests the conceptual motivation.

### Linear AE-PCA

\[
x
\rightarrow
f_{\text{linear}}(x)
\rightarrow
[x;f_{\text{linear}}(x)]
\rightarrow
PCA.
\]

### Nonlinear AE-PCA

\[
x
\rightarrow
f_{\text{nonlinear}}(x)
\rightarrow
[x;f_{\text{nonlinear}}(x)]
\rightarrow
PCA.
\]

If nonlinear AE-PCA consistently improves over linear AE-PCA, that provides stronger evidence that the nonlinear learned features contribute useful information.

---

## 14. Additional Diagnostic: PCA Explained Variance

Report PCA explained variance for the augmented representation:

\[
\frac{\sum_{j=1}^{d}\lambda_j}
{\sum_{j=1}^{D+d_a}\lambda_j}.
\]

However, do not treat explained variance as the primary success criterion.

The actual objective is downstream representation quality.

A method can preserve more variance while producing worse clustering/classification performance.

---

## 15. Expected Outcomes

There are several possible outcomes.

### Outcome 1: AE-PCA > PCA

This supports the hypothesis that nonlinear AE features provide useful structure that PCA over the original embedding cannot access directly.

### Outcome 2: AE-PCA ≈ PCA

The nonlinear features may be redundant with the original embedding for the evaluated task.

### Outcome 3: AE-PCA < PCA

Possible explanations include:

- AE reconstruction objective learns irrelevant structure.
- AE features introduce noise.
- PCA allocates components to reconstruction-related variance that is not useful downstream.
- Poor scale/normalization between \(x\) and \(z\).
- AE capacity or optimization is inadequate.
- The original pretrained embedding already captures the relevant nonlinear structure sufficiently well.

### Outcome 4: AE > PCA but AE-PCA ≈ AE

The original embedding may add little useful information after the nonlinear AE transformation.

---

## 16. Important Conceptual Limitation

AE-PCA is not equivalent to Kernel PCA.

Kernel PCA uses a predefined kernel:

\[
K_{ij}=k(x_i,x_j)
\]

and implicitly maps:

\[
x\rightarrow\phi(x).
\]

AE-PCA learns:

\[
x\rightarrow f_\theta(x)
\]

using a neural network objective.

Therefore:

\[
\boxed{
\text{Kernel PCA: kernel-defined nonlinear geometry}
}
\]

\[
\boxed{
\text{AE-PCA: learned nonlinear feature geometry}
}
\]

The AE mapping is also directly evaluable on unseen samples:

\[
x__\rightarrow f\_\theta(x__),
\]

without requiring an explicit kernel similarity vector against every training sample.

---

## 17. Recommended Experimental Matrix

For an embedding dimension \(D=768\), start with:

| Method       | AE latent \(d_a\) | Final dimension \(d\) |
| ------------ | ----------------: | --------------------: |
| PCA          |                 — |                    96 |
| PCA          |                 — |                   192 |
| PCA          |                 — |                   384 |
| Linear AE    |                96 |                    96 |
| Linear AE    |               192 |                   192 |
| Linear AE    |               384 |                   384 |
| Nonlinear AE |                96 |                    96 |
| Nonlinear AE |               192 |                   192 |
| Nonlinear AE |               384 |                   384 |
| AE-PCA       |        96/192/384 |                    96 |
| AE-PCA       |        96/192/384 |                   192 |
| AE-PCA       |        96/192/384 |                   384 |

For a first experiment, reduce the matrix:

- \(d_a=384\)
- \(d\in\{96,192,384\}\)

and compare:

\[
PCA_d,\quad AE_d,\quad AE\text{-}PCA_d.
\]

Add linear AE and Kernel PCA after establishing the basic result.

---

## 18. Success Criterion

The main result should be reported as downstream performance versus final dimensionality.

The strongest evidence would be a consistent pattern such as:

\[
AE\text{-}PCA_{96}

>

PCA_{96}
\]

\[
AE\text{-}PCA_{192}

>

PCA_{192}
\]

\[
AE\text{-}PCA_{384}

>

PCA_{384}.
\]

If the improvement occurs only at particular dimensions or datasets, report it as such rather than claiming a general superiority.

The central research question is:

> **Can learned nonlinear feature augmentation improve PCA-based compression of pretrained embeddings while retaining the information present in the original embedding?**

---

## 19. Implementation Requirements

The implementation should:

- Reuse the same input embeddings for all methods.
- Keep the final output dimensionality identical.
- Separate AE training from PCA fitting.
- Support arbitrary AE latent dimension.
- Support linear and nonlinear AE variants.
- Support optional feature-block normalization.
- Save fitted AE and PCA parameters for inference.
- Produce embeddings for unseen samples using the fitted transformations.
- Record random seeds.
- Record all hyperparameters.
- Record training and inference time where practical.
- Record reconstruction loss for AE experiments.
- Record downstream evaluation metrics.
- Avoid using downstream labels during unsupervised AE/PCA fitting.

---

## 20. Minimal Reference Implementation

Conceptually:

```python
# X: [N, D]

# 1. Train nonlinear autoencoder
ae.fit(X_train)

# 2. Generate nonlinear features
Z_train = ae.encode(X_train)
Z_eval = ae.encode(X_eval)

# 3. Augment
X_aug_train = concatenate([X_train, Z_train], axis=1)
X_aug_eval = concatenate([X_eval, Z_eval], axis=1)

# 4. Fit PCA only on training data
pca.fit(X_aug_train)

# 5. Compress
Y_train = pca.transform(X_aug_train)
Y_eval = pca.transform(X_aug_eval)
```

Baseline:

```python
pca.fit(X_train)

Y_train = pca.transform(X_train)
Y_eval = pca.transform(X_eval)
```

AE baseline:

```python
ae.fit(X_train)

Y_train = ae.encode(X_train)
Y_eval = ae.encode(X_eval)
```

The three methods must be evaluated at identical final dimensionalities.

---

## 21. Main Takeaway

The proposed method can be summarized by one equation:

\[
\boxed{
\operatorname{AE\text{-}PCA}_d(x)
=

\operatorname{PCA}_d
\left(
\left[
x;
f_\theta(x)
\right]
\right)
}
\]

where \(f_\theta\) is a learned nonlinear encoder.

The hypothesis is not that the AE creates additional information. The hypothesis is that it creates **useful nonlinear coordinates** of the existing information, allowing PCA to construct a low-dimensional representation using both:

1. the original embedding geometry, and
2. learned nonlinear structure.

The experiment should determine whether this additional flexibility improves downstream performance over ordinary PCA.
