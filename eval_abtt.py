import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from eval_embeddings import fit_train_pca_sklearn, fit_train_subspace, load_split
from utils import cos_similarity_matrix, effective_rank, remove_topd

D_VALUES = [1, 2, 3, 5, 10]


def knn_accuracy(train_X, train_y, test_X, test_y):
    clf = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    clf.fit(train_X, train_y)
    return clf.score(test_X, test_y)


def spectrum_effective_rank(embedding):
    similarity = cos_similarity_matrix(embedding.T)
    val = np.linalg.eigvalsh(similarity)
    return effective_rank(val / val.sum())


def get_method_basis(train_X, method):
    if method == "eigen-uncentered":
        _, vec, mean = fit_train_subspace(train_X, center=False)
        return vec, mean
    if method == "eigen-centered":
        _, vec, mean = fit_train_subspace(train_X, center=True)
        return vec, mean
    _, pca = fit_train_pca_sklearn(train_X)
    return pca.components_.T, pca.mean_


def run_abtt(encoder_name, train_X, train_y, test_X, test_y):
    print(f"\n=== {encoder_name}: all-but-the-top ===")

    baseline_rank = spectrum_effective_rank(train_X)
    baseline_acc = knn_accuracy(train_X, train_y, test_X, test_y)
    print(f"baseline (D=0): effective_rank={baseline_rank:.2f} acc={baseline_acc:.4f}")

    for method in ["eigen-uncentered", "eigen-centered", "sklearn-pca"]:
        basis, mean = get_method_basis(train_X, method)
        print(f"-- {method} --")
        for d in D_VALUES:
            train_res = remove_topd(train_X, basis, d, mean=mean)
            test_res = remove_topd(test_X, basis, d, mean=mean)
            rank = spectrum_effective_rank(train_res)
            acc = knn_accuracy(train_res, train_y, test_res, test_y)
            print(
                f"   D={d:>2}: effective_rank={rank:.2f} (baseline {baseline_rank:.2f})  "
                f"acc={acc:.4f} (baseline {baseline_acc:.4f})"
            )


if __name__ == "__main__":
    clip_train_X, clip_train_y = load_split("mini-imagenet_clip-vit-base-patch32", "train")
    clip_test_X, clip_test_y = load_split("mini-imagenet_clip-vit-base-patch32", "test")
    run_abtt("CLIP", clip_train_X, clip_train_y, clip_test_X, clip_test_y)

    dino_train_X, dino_train_y = load_split("mini-imagenet_dinov2_vitb14", "train")
    dino_test_X, dino_test_y = load_split("mini-imagenet_dinov2_vitb14", "test")
    run_abtt("DINO", dino_train_X, dino_train_y, dino_test_X, dino_test_y)
