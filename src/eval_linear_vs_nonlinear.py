"""Experiment 3 from plans/PROPOSAL_FIX_EXPERIMENTS.md: empirical evidence
for "why not non-linear dimensionality reduction."

The proposal currently justifies skipping non-linear DR purely by citing the
linear-probe literature (Alain & Bengio). Preliminary-experiment already has
non-linear-vs-linear MTEB results on disk (AUTO_ENCODER_*/results/ etc.),
but that's a pure-aggregation job over Preliminary-experiment's data (see
that plan's Experiment 3 design) -- this script is the self-contained src/
counterpart: it trains a small non-linear AutoEncoder directly on cached
MINI_IMAGE_NET embeddings and compares it against PCA, LDA, and a
RandomProjection lower-bound control, all at matched target dimension, via
the same clustering-ACC protocol as Experiments 1-2.

Requires torch -- intended to run on a Colab GPU runtime (see CLAUDE.md's
Colab-CLI section), not the local CPU-only venv.

Runs only on the target pseudo-domain (utils.assign_domains()'s last chunk,
same domain Experiment 1 calls "Target (same-domain)") to keep AutoEncoder
training cost bounded; this is a pilot-scale, few-epoch non-linear baseline,
not an exhaustively tuned one -- see plans/PCA_AE_PLAN.md for the fuller
AE-PCA hybrid design this is a cheap precursor to.
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import assign_domains, cluster_metrics, load_split, split_fit_eval

N_DOMAINS = 5
TARGET_DOMAIN = N_DOMAINS - 1
TARGET_DIM = [4, 8, 12, 16, 19]  # capped by (classes_per_domain - 1), the LDA rank limit
SEEDS = range(3)
AE_EPOCHS = 15
AE_HIDDEN = 512
AE_BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TinyAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden=AE_HIDDEN):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim))
        self.decoder = nn.Sequential(
            nn.ReLU(), nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def fit_autoencoder(X, latent_dim, seed):
    torch.manual_seed(seed)
    model = TinyAutoEncoder(X.shape[1], latent_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=AE_BATCH_SIZE, shuffle=True, drop_last=True)

    model.train()
    for _ in range(AE_EPOCHS):
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon, _ = model(batch)
            loss = nn.functional.mse_loss(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return model


def encode(model, X):
    with torch.no_grad():
        z = model.encoder(torch.from_numpy(X).float().to(DEVICE))
    return z.cpu().numpy()


def run_linear_vs_nonlinear(encoder_name, prefix):
    print(f"\n=== {encoder_name}: linear vs. non-linear DR (device={DEVICE}) ===")

    X, y = load_split(prefix, "train")
    domain_of_sample, chunks = assign_domains(y, N_DOMAINS)
    target_X, target_y = X[domain_of_sample == TARGET_DOMAIN], y[domain_of_sample == TARGET_DOMAIN]

    rows = []
    for seed in SEEDS:
        fit_X, fit_y, eval_X, eval_y = split_fit_eval(target_X, target_y, seed)
        raw_metrics = cluster_metrics(eval_X, eval_y, seed)

        for dim in TARGET_DIM:
            variants = {
                "PCA": PCA(n_components=dim, random_state=seed).fit(fit_X),
                "LDA": LinearDiscriminantAnalysis(n_components=dim).fit(fit_X, fit_y),
                "RandomProjection": GaussianRandomProjection(n_components=dim, random_state=seed).fit(fit_X),
            }
            for name, model in variants.items():
                transformed = model.transform(eval_X)
                metrics = cluster_metrics(transformed, eval_y, seed)
                rows.append(
                    {
                        "Encoder": encoder_name,
                        "Method": name,
                        "Target Dimension": dim,
                        "Seed": seed,
                        **{f"raw_{k}": v for k, v in raw_metrics.items()},
                        **metrics,
                    }
                )

            ae_model = fit_autoencoder(fit_X, dim, seed)
            ae_transformed = encode(ae_model, eval_X)
            ae_metrics = cluster_metrics(ae_transformed, eval_y, seed)
            rows.append(
                {
                    "Encoder": encoder_name,
                    "Method": "AutoEncoder",
                    "Target Dimension": dim,
                    "Seed": seed,
                    **{f"raw_{k}": v for k, v in raw_metrics.items()},
                    **ae_metrics,
                }
            )
            print(f"  seed={seed} dim={dim}: PCA/LDA/RandomProjection/AutoEncoder done")

    df = pd.DataFrame(rows)
    for metric in ("nmi", "ari", "acc", "v_measure"):
        df[f"gain_{metric}"] = df[metric] - df[f"raw_{metric}"]
    df.to_csv(f"outputs/linear_vs_nonlinear_{encoder_name}.csv", index=False)

    plt.figure(figsize=(6, 5))
    for method, group in df.groupby("Method"):
        agg = group.groupby("Target Dimension")["acc"].mean()
        plt.plot(agg.index, agg.values, marker="o", label=method)
    plt.title(f"{encoder_name}: linear vs. non-linear DR")
    plt.xlabel("Target dimension")
    plt.ylabel("Clustering ACC (mean over seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/linear_vs_nonlinear_{encoder_name}.png")
    plt.close()

    print(f"\n--- {encoder_name} summary (mean ACC, mean gain over raw) ---")
    print(df.groupby("Method")[["acc", "gain_acc"]].mean())


if __name__ == "__main__":
    run_linear_vs_nonlinear("CLIP", "mini-imagenet_clip-vit-base-patch32")
    run_linear_vs_nonlinear("DINO", "mini-imagenet_dinov2_vitb14")
