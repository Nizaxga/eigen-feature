import numpy as np


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
