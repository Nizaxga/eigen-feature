import matplotlib.pyplot as plt
import numpy as np

from utils import (
    cos_similarity_matrix,
    effective_dimension,
    effective_rank,
    project_topk,
)


def run_test(train_embedding, test_embedding, validation_embedding):
    train_similarity = cos_similarity_matrix(train_embedding.T)
    test_similarity = cos_similarity_matrix(test_embedding.T)
    validation_similarity = cos_similarity_matrix(validation_embedding.T)

    train_val, train_vec = np.linalg.eigh(train_similarity)
    test_val, test_vec = np.linalg.eigh(test_similarity)
    validation_val, validation_vec = np.linalg.eigh(validation_similarity)

    train_order = np.argsort(train_val)[::-1]
    test_order = np.argsort(test_val)[::-1]
    validation_order = np.argsort(validation_val)[::-1]

    train_val, train_vec = train_val[train_order], train_vec[:, train_order]
    test_val, test_vec = test_val[test_order], test_vec[:, test_order]
    validation_val, validation_vec = (
        validation_val[validation_order],
        validation_vec[:, validation_order],
    )

    train_val /= train_val.sum()
    test_val /= test_val.sum()
    validation_val /= validation_val.sum()

    plt.plot(train_val, label="Train")
    plt.plot(test_val, label="Test")
    plt.plot(validation_val, label="Validation")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.savefig("eigenvalues.png")

    for name, eigenvalues in [
        ("Train", train_val),
        ("Test", test_val),
        ("Validation", validation_val),
    ]:
        print(
            name,
            "90%:",
            effective_dimension(eigenvalues, 0.90),
            "95%:",
            effective_dimension(eigenvalues, 0.95),
            "99%:",
            effective_dimension(eigenvalues, 0.99),
        )

    train_rank = effective_rank(train_val)
    test_rank = effective_rank(test_val)
    validation_rank = effective_rank(validation_val)

    print("Train effective rank:", train_rank)
    print("Test effective rank:", test_rank)
    print("Validation effective rank:", validation_rank)

    for name, embedding, vec, rank in [
        ("Train", train_embedding, train_vec, train_rank),
        ("Test", test_embedding, test_vec, test_rank),
        ("Validation", validation_embedding, validation_vec, validation_rank),
    ]:
        k = max(1, round(rank))
        projected = project_topk(embedding, vec, k)
        print(
            f"{name} projected onto top-{k} eigenvectors (effective rank {rank:.2f}):",
            projected.shape,
        )


if __name__ == "__main__":
    print("[LOG] Running From main.py")

    print("[LOG] Loading CLIP miniImageNet embeddings")
    MINI_NET_CLIP_TRAIN = np.load(
        "MINI_IMAGE_NET/mini-imagenet_clip-vit-base-patch32_train.npz"
    )
    MINI_NET_CLIP_TEST = np.load(
        "MINI_IMAGE_NET/mini-imagenet_clip-vit-base-patch32_test.npz"
    )
    MINI_NET_CLIP_VALIDATION = np.load(
        "MINI_IMAGE_NET/mini-imagenet_clip-vit-base-patch32_validation.npz"
    )

    run_test(
        MINI_NET_CLIP_TRAIN["embeddings"],
        MINI_NET_CLIP_TEST["embeddings"],
        MINI_NET_CLIP_VALIDATION["embeddings"],
    )

    print("[LOG] Loading DINO miniImageNet embeddings")
    MINI_NET_DINO_TRAIN = np.load(
        "MINI_IMAGE_NET/mini-imagenet_dinov2_vitb14_train.npz"
    )
    MINI_NET_DINO_TEST = np.load("MINI_IMAGE_NET/mini-imagenet_dinov2_vitb14_test.npz")
    MINI_NET_DINO_VALIDATION = np.load(
        "MINI_IMAGE_NET/mini-imagenet_dinov2_vitb14_validation.npz"
    )

    run_test(
        MINI_NET_DINO_TRAIN["embeddings"],
        MINI_NET_DINO_TEST["embeddings"],
        MINI_NET_DINO_VALIDATION["embeddings"],
    )
