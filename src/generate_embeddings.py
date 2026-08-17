"""Generate image embeddings for (model, dataset, split) combos.

Primary usage: notebooks/generate_embeddings_colab.ipynb on a Colab GPU runtime —
imports MODEL_REGISTRY/DATASET_REGISTRY/run() directly and calls run(...) from a cell
(no shell invocation). That notebook also handles the Drive mount + outputs/ symlink
(same convention as distrillation-model-expr's colab notebooks) so results survive a
session reset.

CLI usage also works (e.g. local runs, no notebook):
    !pip install -q torch transformers datasets pillow
    !python src/generate_embeddings.py --model siglip-base-patch16 --dataset cifar100 \
        --split train --max-samples 200   # smoke test first
    !python src/generate_embeddings.py --model siglip-base-patch16 --dataset cifar100 --split train

_maybe_mount_drive_outputs() below only runs on the CLI path (inside __main__) — the
notebook does its own mount+symlink in its own cells instead, since importing this
module from a notebook never executes __main__.

Output: <output_root>/<dataset>/<model>_<split>.npz with keys "embeddings" (float32) and "labels" (int64).
Default --output-root is "outputs/embeddings".
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel


def _maybe_mount_drive_outputs():
    try:
        from google.colab import drive
    except ImportError:
        return

    drive.mount("/content/drive")
    save_dir = "/content/drive/MyDrive/eigen-feature"
    drive_outputs = os.path.join(save_dir, "output-embedding-generation")

    if os.path.isdir("/content/drive/MyDrive") and not os.path.exists("outputs"):
        os.makedirs(drive_outputs, exist_ok=True)
        os.symlink(drive_outputs, "outputs")
        print(f"[LOG] outputs/ symlinked to {drive_outputs}")

MODEL_REGISTRY = {
    "clip-vit-base-patch32": {"hf_id": "openai/clip-vit-base-patch32", "kind": "clip_like"},
    "dinov2-vitb14": {"hf_id": "facebook/dinov2-base", "kind": "pooler"},
    "siglip-base-patch16": {"hf_id": "google/siglip-base-patch16-224", "kind": "clip_like"},
    "resnet50": {"hf_id": "microsoft/resnet-50", "kind": "pooler"},
}

DATASET_REGISTRY = {
    "mini-imagenet": {
        "hf_id": "timm/mini-imagenet",
        "image_key": "image",
        "label_key": "label",
        "splits": {"train": "train", "test": "test", "validation": "validation"},
    },
    "cifar100": {
        "hf_id": "uoft-cs/cifar100",
        "image_key": "img",
        "label_key": "fine_label",
        "splits": {"train": "train", "test": "test"},
    },
    "oxford-pets": {
        "hf_id": "pcuenq/oxford-pets",
        "image_key": "image",
        "label_key": "label",
        "splits": {"train": "train", "test": "test"},
    },
}


def extract_features(model, kind, pixel_values):
    if kind == "clip_like":
        return model.get_image_features(pixel_values=pixel_values)
    return model(pixel_values=pixel_values).pooler_output


def run(model_name, dataset_name, split, batch_size, max_samples, output_root, device):
    model_cfg = MODEL_REGISTRY[model_name]
    dataset_cfg = DATASET_REGISTRY[dataset_name]

    if split not in dataset_cfg["splits"]:
        raise ValueError(
            f"dataset '{dataset_name}' has no split '{split}'; "
            f"available: {list(dataset_cfg['splits'])}"
        )

    processor = AutoImageProcessor.from_pretrained(model_cfg["hf_id"])
    model = AutoModel.from_pretrained(model_cfg["hf_id"]).to(device).eval()

    ds = load_dataset(dataset_cfg["hf_id"], split=dataset_cfg["splits"][split])
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    image_key, label_key = dataset_cfg["image_key"], dataset_cfg["label_key"]
    all_embeddings, all_labels = [], []

    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            batch = ds[start : start + batch_size]
            images = [img.convert("RGB") for img in batch[image_key]]
            inputs = processor(images=images, return_tensors="pt").to(device)
            features = extract_features(model, model_cfg["kind"], inputs["pixel_values"])

            all_embeddings.append(features.cpu().numpy().astype(np.float32))
            all_labels.append(np.array(batch[label_key], dtype=np.int64))
            print(f"[{model_name}/{dataset_name}/{split}] {min(start + batch_size, len(ds))}/{len(ds)}")

    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)

    out_dir = Path(output_root) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_{split}.npz"
    np.savez(out_path, embeddings=embeddings, labels=labels)
    print(f"saved {out_path}: embeddings{embeddings.shape} labels{labels.shape}")


if __name__ == "__main__":
    _maybe_mount_drive_outputs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--split", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/embeddings")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args.model, args.dataset, args.split, args.batch_size, args.max_samples, args.output_root, device)
