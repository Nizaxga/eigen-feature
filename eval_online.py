import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from eval_embeddings import fit_train_pca_sklearn, fit_train_subspace, load_split
from utils import project_topk

SEED = 0
CHECKPOINTS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 50000]


def knn_accuracy(train_X, train_y, test_X, test_y):
    clf = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    clf.fit(train_X, train_y)
    return clf.score(test_X, test_y)


def _method_accuracy(method, seen_X, seen_y, test_X, test_y, n):
    if method == "eigen-uncentered":
        k, vec, mean = fit_train_subspace(seen_X, center=False)
    elif method == "eigen-centered":
        k, vec, mean = fit_train_subspace(seen_X, center=True)
    else:
        k, pca = fit_train_pca_sklearn(seen_X)

    k = min(k, n - 1)

    if method in ("eigen-uncentered", "eigen-centered"):
        seen_proj = project_topk(seen_X, vec, k, mean=mean)
        test_proj = project_topk(test_X, vec, k, mean=mean)
    else:
        seen_proj = pca.transform(seen_X)[:, :k]
        test_proj = pca.transform(test_X)[:, :k]

    return knn_accuracy(seen_proj, seen_y, test_proj, test_y), k


def _find_stable_crossover(ns, baseline_accs, method_accs):
    for i in range(len(ns)):
        if method_accs[i] > baseline_accs[i] and all(
            method_accs[j] >= baseline_accs[j] for j in range(i, len(ns))
        ):
            return ns[i]
    return None


def run_streaming_race(encoder_name, train_X, train_y, test_X, test_y):
    print(f"\n=== {encoder_name}: online streaming race ===")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(train_X.shape[0])
    checkpoints = [n for n in CHECKPOINTS if n <= train_X.shape[0]]
    methods = ["eigen-uncentered", "eigen-centered", "sklearn-pca"]

    rows = {"n": [], "baseline": [], **{m: [] for m in methods}, **{f"{m}_k": [] for m in methods}}

    for n in checkpoints:
        seen_idx = order[:n]
        seen_X, seen_y = train_X[seen_idx], train_y[seen_idx]

        baseline_acc = knn_accuracy(seen_X, seen_y, test_X, test_y)
        rows["n"].append(n)
        rows["baseline"].append(baseline_acc)

        line = f"n={n:>6} baseline_acc={baseline_acc:.4f}"
        for method in methods:
            acc, k = _method_accuracy(method, seen_X, seen_y, test_X, test_y, n)
            rows[method].append(acc)
            rows[f"{method}_k"].append(k)
            line += f"  {method}(k={k})={acc:.4f}"
        print(line)

    for method in methods:
        crossover = _find_stable_crossover(rows["n"], rows["baseline"], rows[method])
        if crossover is None:
            gap = rows[method][-1] - rows["baseline"][-1]
            print(f"{method}: no stable crossover (final gap={gap:+.4f} at n={rows['n'][-1]})")
        else:
            print(f"{method}: stable crossover at n={crossover}")

    plt.figure()
    plt.plot(rows["n"], rows["baseline"], label="full (baseline)", marker="o")
    for method in methods:
        plt.plot(rows["n"], rows[method], label=method, marker="o")
    plt.xscale("log")
    plt.xlabel("# train samples seen (streamed)")
    plt.ylabel("k-NN accuracy on fixed test set")
    plt.title(f"{encoder_name}: online streaming race")
    plt.legend()
    plt.savefig(f"online_{encoder_name}.png")
    plt.close()


if __name__ == "__main__":
    clip_train_X, clip_train_y = load_split("mini-imagenet_clip-vit-base-patch32", "train")
    clip_test_X, clip_test_y = load_split("mini-imagenet_clip-vit-base-patch32", "test")
    run_streaming_race("CLIP", clip_train_X, clip_train_y, clip_test_X, clip_test_y)

    dino_train_X, dino_train_y = load_split("mini-imagenet_dinov2_vitb14", "train")
    dino_test_X, dino_test_y = load_split("mini-imagenet_dinov2_vitb14", "test")
    run_streaming_race("DINO", dino_train_X, dino_train_y, dino_test_X, dino_test_y)
