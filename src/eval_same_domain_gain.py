"""Experiment 1 from plans/PROPOSAL_FIX_EXPERIMENTS.md: same-domain vs.
cross-domain compression gain.

Preliminary-experiment/n-2-cat2.py runs this over live ImageNet streaming +
CLIP inference across real semantic domains (Dogs/Birds/Vehicles/Household ->
Cats). This is the same protocol reimplemented against the cached
MINI_IMAGE_NET embeddings instead, using utils.assign_domains() to build
structural (not semantic) pseudo-domains -- see that function's docstring.
Intended to run on a Colab GPU runtime for the KMeans/sklearn fitting cost
(see CLAUDE.md's Colab-CLI section); no torch/streaming dependency here.

For each source pseudo-domain, fit PCA/LDA on that domain's full sample and
zero-shot-transform the disjoint target-domain eval split; separately fit on
the target domain's own disjoint fit split ("Target (same-domain)") as the
gain ceiling. Gain = compressed clustering ACC - raw clustering ACC;
TransferRetention = cross-domain mean Gain / same-domain mean Gain (can
legitimately exceed 1.0 or go negative -- report both directions).
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils import assign_domains, cluster_metrics, load_split, split_fit_eval

N_DOMAINS = 5
TARGET_DOMAIN = N_DOMAINS - 1
TARGET_DIM = [4, 8, 12, 16, 19]  # capped by (classes_per_domain - 1), the LDA rank limit
SEEDS = range(3)


def fit_transform(method, source_X, source_y, target_X, dim, seed):
    if method == "PCA":
        model = PCA(n_components=dim, random_state=seed).fit(source_X)
    else:
        model = LinearDiscriminantAnalysis(n_components=dim).fit(source_X, source_y)
    return model.transform(target_X)


def run_same_domain_gain(encoder_name, prefix):
    print(f"\n=== {encoder_name}: same-domain vs. cross-domain gain ===")

    X, y = load_split(prefix, "train")
    domain_of_sample, chunks = assign_domains(y, N_DOMAINS)
    print(f"domains: {[len(c) for c in chunks]} classes each")

    target_mask = domain_of_sample == TARGET_DOMAIN
    target_X, target_y = X[target_mask], y[target_mask]

    rows = []
    for seed in SEEDS:
        target_fit_X, target_fit_y, target_eval_X, target_eval_y = split_fit_eval(
            target_X, target_y, seed
        )
        raw_metrics = cluster_metrics(target_eval_X, target_eval_y, seed)

        sources = {"Target (same-domain)": (target_fit_X, target_fit_y)}
        for domain_id in range(N_DOMAINS):
            if domain_id == TARGET_DOMAIN:
                continue
            mask = domain_of_sample == domain_id
            sources[f"Domain_{domain_id}"] = (X[mask], y[mask])

        for method in ("PCA", "LDA"):
            for dim in TARGET_DIM:
                for source_name, (source_X, source_y) in sources.items():
                    transformed = fit_transform(
                        method, source_X, source_y, target_eval_X, dim, seed
                    )
                    metrics = cluster_metrics(transformed, target_eval_y, seed)
                    rows.append(
                        {
                            "Encoder": encoder_name,
                            "Method": method,
                            "Source Domain": source_name,
                            "Target Dimension": dim,
                            "Seed": seed,
                            **{f"raw_{k}": v for k, v in raw_metrics.items()},
                            **metrics,
                        }
                    )
                    print(
                        f"  {method:>4} dim={dim:>2} seed={seed} {source_name:>22}: "
                        f"acc={metrics['acc']:.4f} (raw={raw_metrics['acc']:.4f})"
                    )

    df = pd.DataFrame(rows)
    for metric in ("nmi", "ari", "acc", "v_measure"):
        df[f"gain_{metric}"] = df[metric] - df[f"raw_{metric}"]
    df.to_csv(f"outputs/same_domain_gain_{encoder_name}.csv", index=False)

    mean_gain = (
        df.groupby(["Method", "Source Domain", "Target Dimension"])["gain_acc"]
        .mean()
        .reset_index()
    )
    same_domain_gain = mean_gain[
        mean_gain["Source Domain"] == "Target (same-domain)"
    ].set_index(["Method", "Target Dimension"])["gain_acc"]
    mean_gain["transfer_retention_acc"] = mean_gain.apply(
        lambda r: r["gain_acc"] / same_domain_gain.loc[(r["Method"], r["Target Dimension"])],
        axis=1,
    )
    mean_gain.to_csv(f"outputs/same_domain_gain_summary_{encoder_name}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, method in zip(axes, ("PCA", "LDA")):
        subset = mean_gain[mean_gain["Method"] == method]
        for source_name, group in subset.groupby("Source Domain"):
            group = group.sort_values("Target Dimension")
            ax.plot(group["Target Dimension"], group["gain_acc"], marker="o", label=source_name)
        ax.set_title(f"{encoder_name} / {method}")
        ax.set_xlabel("Target dimension")
        ax.axhline(0, color="gray", linewidth=0.8)
    axes[0].set_ylabel("Clustering ACC gain over raw baseline")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"outputs/same_domain_gain_{encoder_name}.png")
    plt.close()

    print(f"\n--- {encoder_name} summary (mean gain / transfer retention) ---")
    print(mean_gain.groupby(["Method", "Source Domain"])[["gain_acc", "transfer_retention_acc"]].mean())


if __name__ == "__main__":
    run_same_domain_gain("CLIP", "mini-imagenet_clip-vit-base-patch32")
    run_same_domain_gain("DINO", "mini-imagenet_dinov2_vitb14")
