import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from utils import (
    cos_similarity_matrix,
    effective_rank,
    principal_angle_cosines,
    project_topk,
)

SEED = 0


def fit_train_subspace(train_embedding, center=False):
    mean = train_embedding.mean(axis=0) if center else None
    X = train_embedding - mean if center else train_embedding

    similarity = cos_similarity_matrix(X.T)
    val, vec = np.linalg.eigh(similarity)

    order = np.argsort(val)[::-1]
    val, vec = val[order], vec[:, order]

    rank = effective_rank(val / val.sum())
    k = max(1, round(rank))
    return k, vec, mean


def build_projectors(train_embedding):
    projectors = {"full": lambda X: X}
    bases = {}

    for label, center in [("eigen-uncentered", False), ("eigen-centered", True)]:
        k, vec, mean = fit_train_subspace(train_embedding, center=center)
        key = f"{label} (k={k})"
        projectors[key] = (
            lambda X, vec=vec, k=k, mean=mean: project_topk(X, vec, k, mean=mean)
        )
        bases[key] = vec[:, :k]
        print(f"{label}: k={k} / {train_embedding.shape[1]}")

    k_pca, pca = fit_train_pca_sklearn(train_embedding)
    key = f"sklearn-pca (k={k_pca})"
    projectors[key] = lambda X, pca=pca, k=k_pca: pca.transform(X)[:, :k]
    bases[key] = pca.components_[:k_pca].T
    print(f"sklearn-pca: k={k_pca} / {train_embedding.shape[1]}")

    return projectors, bases


def subspace_similarity_report(encoder_name, bases):
    print(f"\n--- [{encoder_name}] subspace similarity (principal angle cosines) ---")

    labels = list(bases)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            cosines = principal_angle_cosines(bases[a], bases[b])
            mean_sq = np.mean(cosines**2)
            print(
                f"{a} vs {b}: mean cos^2={mean_sq:.4f} "
                f"(k_min={min(bases[a].shape[1], bases[b].shape[1])}), "
                f"min cos={cosines.min():.4f}"
            )


def knn_test(encoder_name, projectors, train_X, train_y, test_X, test_y, val_X, val_y):
    print(f"\n--- [{encoder_name}] k-NN classification (k=5, cosine) ---")

    for label, project in projectors.items():
        train_proj = project(train_X)
        for split_name, X, y in [("test", test_X, test_y), ("validation", val_X, val_y)]:
            clf = KNeighborsClassifier(n_neighbors=5, metric="cosine")
            clf.fit(train_proj, train_y)
            acc = clf.score(project(X), y)
            print(f"{label:>24} dim={train_proj.shape[1]:>4} {split_name:>10}: acc={acc:.4f}")


def _sample_episode(rng, train_y, query_y, classes_pool, n_way, k_shot, n_query_per_class):
    classes = rng.choice(classes_pool, size=n_way, replace=False)

    support_idx, support_labels = [], []
    query_idx, query_labels = [], []
    for i, c in enumerate(classes):
        train_class_idx = np.flatnonzero(train_y == c)
        support_idx.append(rng.choice(train_class_idx, size=k_shot, replace=False))
        support_labels.append(np.full(k_shot, i))

        query_class_idx = np.flatnonzero(query_y == c)
        query_idx.append(rng.choice(query_class_idx, size=n_query_per_class, replace=False))
        query_labels.append(np.full(n_query_per_class, i))

    return (
        np.concatenate(support_idx),
        np.concatenate(support_labels),
        np.concatenate(query_idx),
        np.concatenate(query_labels),
    )


def _prototype_accuracy(
    support_embedding, support_idx, support_labels, query_embedding, query_idx, query_labels, n_way
):
    support = support_embedding[support_idx]
    query = query_embedding[query_idx]

    support_norm = support / np.linalg.norm(support, axis=1, keepdims=True)
    query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)

    prototypes = np.stack(
        [support_norm[support_labels == c].mean(axis=0) for c in range(n_way)]
    )
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)

    similarity = query_norm @ prototypes.T
    predictions = np.argmax(similarity, axis=1)
    return np.mean(predictions == query_labels)


def few_shot_test(
    encoder_name,
    projectors,
    train_X,
    train_y,
    query_X,
    query_y,
    n_way=5,
    k_shots=(1, 5),
    n_episodes=200,
    n_query_per_class=15,
):
    print(f"\n--- [{encoder_name}] few-shot classification ({n_way}-way, support=train, query=eval split) ---")

    classes_pool = np.intersect1d(np.unique(train_y), np.unique(query_y))
    projected_train = {label: project(train_X) for label, project in projectors.items()}
    projected_query = {label: project(query_X) for label, project in projectors.items()}

    for k_shot in k_shots:
        print(f"{k_shot}-shot:")
        for label in projectors:
            rng = np.random.default_rng(SEED)
            accs = []
            for _ in range(n_episodes):
                support_idx, support_labels, query_idx, query_labels = _sample_episode(
                    rng, train_y, query_y, classes_pool, n_way, k_shot, n_query_per_class
                )
                accs.append(
                    _prototype_accuracy(
                        projected_train[label],
                        support_idx,
                        support_labels,
                        projected_query[label],
                        query_idx,
                        query_labels,
                        n_way,
                    )
                )
            print(f"  {label:>24}: acc={np.mean(accs):.4f}+/-{np.std(accs):.4f}")


def retrieval_overlap_test(encoder_name, projectors, train_X, query_X, top_k=10, n_queries=500):
    print(f"\n--- [{encoder_name}] retrieval overlap vs full (top-{top_k}, gallery=train) ---")

    rng = np.random.default_rng(SEED)
    n_queries = min(n_queries, query_X.shape[0])
    query_sample_idx = rng.choice(query_X.shape[0], size=n_queries, replace=False)

    def top_k_neighbors(gallery, queries):
        gallery_norm = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
        query_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        similarity = query_norm @ gallery_norm.T
        return np.argsort(similarity, axis=1)[:, ::-1][:, :top_k]

    full_neighbors = top_k_neighbors(train_X, query_X[query_sample_idx])
    chance = top_k / train_X.shape[0]

    for label, project in projectors.items():
        if label == "full":
            continue
        proj_neighbors = top_k_neighbors(project(train_X), project(query_X)[query_sample_idx])
        overlaps = [
            len(np.intersect1d(full_neighbors[i], proj_neighbors[i])) / top_k
            for i in range(n_queries)
        ]
        print(f"{label:>24}: mean overlap={np.mean(overlaps):.4f} (chance ~= {chance:.4f})")


def load_split(prefix, split):
    data = np.load(f"MINI_IMAGE_NET/{prefix}_{split}.npz")
    return data["embeddings"], data["labels"]


def run_all_tests(encoder_name, prefix):
    print(f"\n=== {encoder_name} ===")

    train_X, train_y = load_split(prefix, "train")
    test_X, test_y = load_split(prefix, "test")
    val_X, val_y = load_split(prefix, "validation")

    projectors, bases = build_projectors(train_X)
    subspace_similarity_report(encoder_name, bases)

    knn_test(encoder_name, projectors, train_X, train_y, test_X, test_y, val_X, val_y)
    few_shot_test(encoder_name, projectors, train_X, train_y, test_X, test_y)
    retrieval_overlap_test(encoder_name, projectors, train_X, test_X)


if __name__ == "__main__":
    run_all_tests("CLIP", "mini-imagenet_clip-vit-base-patch32")
    run_all_tests("DINO", "mini-imagenet_dinov2_vitb14")
