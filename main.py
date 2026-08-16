import matplotlib.pyplot as plt
import numpy as np
from utils import effective_rank, cos_similarity_matrix, effective_dimension

def run_test(train_embedding, test_embedding, validation_embedding):
    train_similarity = cos_similarity_matrix(train_embedding.T)
    test_similarity = cos_similarity_matrix(test_embedding.T)
    validation_similarity = cos_similarity_matrix(validation_embedding.T)

    train_val, train_vec = np.linalg.eigh(train_similarity)
    test_val, test_vec = np.linalg.eigh(test_similarity)
    validation_val, validation_vec = np.linalg.eigh(validation_similarity)

    train_val = np.sort(train_val)[::-1]
    test_val = np.sort(test_val)[::-1]
    validation_val = np.sort(validation_val)[::-1]

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
            "90%:", effective_dimension(eigenvalues, 0.90),
            "95%:", effective_dimension(eigenvalues, 0.95),
            "99%:", effective_dimension(eigenvalues, 0.99),
        )

    print("Train effective rank:", effective_rank(train_val))
    print("Test effective rank:", effective_rank(test_val))
    print("Validation effective rank:", effective_rank(validation_val))


if __name__ == "__main__":
    print("[LOG] Running From main.py")

    print("[LOG] Loading CLIP miniImageNet embeddings")
    MINI_NET_CLIP_TRAIN = np.load("MINI_IMAGE_NET/mini-imagenet_clip-vit-base-patch32_train.npz")
    MINI_NET_CLIP_TEST = np.load("MINI_IMAGE_NET/mini-imagenet_clip-vit-base-patch32_test.npz")
    MINI_NET_CLIP_VALIDATION = np.load("MINI_IMAGE_NET/mini-imagenet_clip-vit-base-patch32_validation.npz")

    run_test(MINI_NET_CLIP_TRAIN["embeddings"], MINI_NET_CLIP_TEST["embeddings"], MINI_NET_CLIP_VALIDATION["embeddings"])

    print("[LOG] Loading DINO miniImageNet embeddings")
    MINI_NET_DINO_TRAIN = np.load("MINI_IMAGE_NET/mini-imagenet_dinov2_vitb14_train.npz")
    MINI_NET_DINO_TEST = np.load("MINI_IMAGE_NET/mini-imagenet_dinov2_vitb14_test.npz")
    MINI_NET_DINO_VALIDATION = np.load("MINI_IMAGE_NET/mini-imagenet_dinov2_vitb14_validation.npz")

    run_test(MINI_NET_DINO_TRAIN["embeddings"], MINI_NET_DINO_TEST["embeddings"], MINI_NET_DINO_VALIDATION["embeddings"])
