import numpy as np
from sklearn.decomposition import PCA


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
