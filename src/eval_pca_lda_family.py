"""Experiment 2 from plans/PROPOSAL_FIX_EXPERIMENTS.md: PCA/LDA family
ablation. "Variance of PCA, LDA techniques" means the *family* of each
method (PPCA, Kernel PCA, Sparse PCA, ... = PCA + some add-in; shrinkage /
kernel variants = LDA + some add-in), not explained-variance curves.

PPCA is skipped here on purpose: under maximum-likelihood fitting on
complete data its principal subspace is provably identical to ordinary
PCA's (see GPT_PCA_FORMALIZE.md, section 2's "Key Result") -- it only adds
a scalar isotropic noise term, no new eigenvectors, so an empirical
PPCA-vs-PCA run would just reproduce the PCA row.

Robust PCA is skipped too: no drop-in sklearn primitive (needs a custom
low-rank + sparse solver, e.g. IALM/ADMM) -- out of scope for this pilot,
noted as future work.

Protocol: reuse utils.assign_domains()'s pseudo-domains from Experiment 1
(src/eval_same_domain_gain.py); for each domain, fit every family member on
that domain's own disjoint fit-split and score on its own eval-split (no
cross-domain transfer here -- this experiment asks "does the add-in change
downstream quality at all," not "does it transfer"). Same TARGET_DIM list as
Experiment 1 so PCA-family and LDA-family variants are compared at matched
final dimensionality throughout.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis, KernelPCA, MiniBatchSparsePCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils import assign_domains, cluster_metrics, load_split, split_fit_eval

N_DOMAINS = 5
TARGET_DIM = [4, 8, 12, 16, 19]  # capped by (classes_per_domain - 1), the LDA rank limit
SEEDS = range(3)

PCA_FAMILY = ["PCA", "KernelPCA", "SparsePCA", "FactorAnalysis"]
LDA_FAMILY = ["LDA", "ShrinkageLDA", "KernelLDA"]


class KernelLDA:
    """Non-linear LDA stand-in: sklearn has no built-in Kernel Discriminant
    Analysis, so approximate it with a KernelPCA projection followed by
    plain LDA on the projected features."""

    def __init__(self, n_components, kernel_components=50, seed=0):
        self.kpca = KernelPCA(n_components=kernel_components, kernel="rbf", random_state=seed)
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)

    def fit(self, X, y):
        self.kpca.fit(X)
        self.lda.fit(self.kpca.transform(X), y)
        return self

    def transform(self, X):
        return self.lda.transform(self.kpca.transform(X))


def make_variant(name, dim, seed):
    if name == "PCA":
        return PCA(n_components=dim, random_state=seed)
    if name == "KernelPCA":
        return KernelPCA(n_components=dim, kernel="rbf", random_state=seed)
    if name == "SparsePCA":
        return MiniBatchSparsePCA(n_components=dim, alpha=1.0, random_state=seed)
    if name == "FactorAnalysis":
        return FactorAnalysis(n_components=dim, random_state=seed)
    if name == "LDA":
        return LinearDiscriminantAnalysis(n_components=dim)
    if name == "ShrinkageLDA":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", n_components=dim)
    if name == "KernelLDA":
        return KernelLDA(n_components=dim, seed=seed)
    raise ValueError(name)


def run_family_ablation(encoder_name, prefix):
    print(f"\n=== {encoder_name}: PCA/LDA family ablation ===")
    print("skipping PPCA (analytically == PCA subspace) and Robust PCA (no sklearn primitive)")

    X, y = load_split(prefix, "train")
    domain_of_sample, chunks = assign_domains(y, N_DOMAINS)
    print(f"domains: {[len(c) for c in chunks]} classes each")

    rows = []
    for seed in SEEDS:
        for domain_id in range(N_DOMAINS):
            mask = domain_of_sample == domain_id
            domain_X, domain_y = X[mask], y[mask]
            fit_X, fit_y, eval_X, eval_y = split_fit_eval(domain_X, domain_y, seed)
            raw_metrics = cluster_metrics(eval_X, eval_y, seed)

            for family_name, variants in (("PCA", PCA_FAMILY), ("LDA", LDA_FAMILY)):
                for variant in variants:
                    for dim in TARGET_DIM:
                        model = make_variant(variant, dim, seed)
                        if variant in PCA_FAMILY:
                            model.fit(fit_X)
                        else:
                            model.fit(fit_X, fit_y)
                        transformed = model.transform(eval_X)
                        metrics = cluster_metrics(transformed, eval_y, seed)
                        rows.append(
                            {
                                "Encoder": encoder_name,
                                "Family": family_name,
                                "Variant": variant,
                                "Domain": f"Domain_{domain_id}",
                                "Target Dimension": dim,
                                "Seed": seed,
                                **{f"raw_{k}": v for k, v in raw_metrics.items()},
                                **metrics,
                            }
                        )
            print(f"  seed={seed} domain={domain_id} done")

    df = pd.DataFrame(rows)
    for metric in ("nmi", "ari", "acc", "v_measure"):
        df[f"gain_{metric}"] = df[metric] - df[f"raw_{metric}"]
    df.to_csv(f"outputs/pca_lda_family_results_{encoder_name}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (family_name, variants) in zip(axes, (("PCA", PCA_FAMILY), ("LDA", LDA_FAMILY))):
        subset = df[df["Family"] == family_name]
        for variant in variants:
            agg = subset[subset["Variant"] == variant].groupby("Target Dimension")["acc"].mean()
            ax.plot(agg.index, agg.values, marker="o", label=variant)
        ax.set_title(f"{encoder_name} / {family_name} family")
        ax.set_xlabel("Target dimension")
    axes[0].set_ylabel("Clustering ACC (mean over domains/seeds)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"outputs/pca_lda_family_{encoder_name}.png")
    plt.close()

    print(f"\n--- {encoder_name} family summary (mean ACC, mean gain over raw) ---")
    print(df.groupby(["Family", "Variant"])[["acc", "gain_acc"]].mean())


if __name__ == "__main__":
    run_family_ablation("CLIP", "mini-imagenet_clip-vit-base-patch32")
    run_family_ablation("DINO", "mini-imagenet_dinov2_vitb14")
