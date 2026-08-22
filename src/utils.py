import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score


def effective_rank(eigenvalues):
    eigenvalues = np.maximum(eigenvalues, 0)
    p = eigenvalues / eigenvalues.sum()
    p = p[p > 0]
    return np.exp(-np.sum(p * np.log(p)))


def cos_similarity_matrix(X: np.ndarray) -> np.ndarray:
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X_norm @ X_norm.T


def effective_dimension(eigenvalues, threshold):
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(eigenvalues)
    return np.searchsorted(cumulative, threshold) + 1


def project_topk(embedding, eigenvectors, k, mean=None):
    if mean is not None:
        embedding = embedding - mean
    return embedding @ eigenvectors[:, :k]


def principal_angle_cosines(basis_a, basis_b):
    return np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)


def remove_topd(embedding, basis, d, mean=None):
    if mean is not None:
        embedding = embedding - mean
    top = basis[:, :d]
    return embedding - (embedding @ top) @ top.T


def fit_train_pca_sklearn(train_embedding):
    pca = PCA().fit(train_embedding)
    rank = effective_rank(pca.explained_variance_ / pca.explained_variance_.sum())
    k = max(1, round(rank))
    return k, pca


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


def load_split(prefix, split):
    data = np.load(f"MINI_IMAGE_NET/{prefix}_{split}.npz")
    return data["embeddings"], data["labels"]


def cluster_accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n_labels = int(max(y_true.max(), y_pred.max())) + 1
    w = np.zeros((n_labels, n_labels), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        w[p, t] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def cluster_metrics(embedding, labels, seed, n_init=10):
    """K-means clustering proxy metrics, matching the thesis's own evaluation
    protocol (see proposal/gradschool-newformat/3-literature-review.tex,
    "Clustering as a Proxy Evaluation Task")."""
    n_classes = len(np.unique(labels))
    clusters = KMeans(n_clusters=n_classes, random_state=seed, n_init=n_init).fit_predict(embedding)
    return {
        "nmi": normalized_mutual_info_score(labels, clusters),
        "ari": adjusted_rand_score(labels, clusters),
        "v_measure": v_measure_score(labels, clusters),
        "acc": cluster_accuracy(labels, clusters),
    }


def assign_domains(labels, n_domains=5):
    """Partition labels into n_domains contiguous chunks of the sorted class-id
    space. MINI_IMAGE_NET's 100 classes carry no cached semantic domain metadata
    (no WordNet synset ranges), so this is a structural analog of
    Preliminary-experiment/n-2-cat2.py's ImageNet-synset-range domains
    (Dogs=151-268, etc.), not a semantic one. Returns (domain_of_sample,
    chunks) where chunks[i] is the array of class ids belonging to domain i."""
    classes = np.unique(labels)
    chunks = np.array_split(classes, n_domains)
    domain_of_class = {c: i for i, chunk in enumerate(chunks) for c in chunk}
    domain_of_sample = np.array([domain_of_class[label] for label in labels])
    return domain_of_sample, chunks


def split_fit_eval(X, y, seed, frac=0.5):
    """Disjoint per-class fit/eval split, so a projection fit on the "fit" half
    and scored on the "eval" half never sees the same image twice."""
    rng = np.random.default_rng(seed)
    fit_idx, eval_idx = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * frac))
        fit_idx.append(idx[:cut])
        eval_idx.append(idx[cut:])
    fit_idx, eval_idx = np.concatenate(fit_idx), np.concatenate(eval_idx)
    return X[fit_idx], y[fit_idx], X[eval_idx], y[eval_idx]
