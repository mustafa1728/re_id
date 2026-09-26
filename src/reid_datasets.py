"""Access to the datasets written by src/process_datasets.py:

    <processed_root>/<dataset>/
        img_name2label.json        {"<category>/<img_name>": label}
        stats.json                 dataset, task, num_images, num_categories, attributes
        images/<img_name>.png
        features/<features>/<img_name>.pt
"""
import os
import json

import numpy as np
import torch
import tqdm

from .config import PROCESSED_DATA_ROOT


def load_img_name2label(dataset, processed_root=PROCESSED_DATA_ROOT):
    with open(os.path.join(processed_root, dataset, "img_name2label.json")) as f:
        return json.load(f)


def load_stats(dataset, processed_root=PROCESSED_DATA_ROOT):
    with open(os.path.join(processed_root, dataset, "stats.json")) as f:
        return json.load(f)


def image_path(dataset, img_name, processed_root=PROCESSED_DATA_ROOT):
    return os.path.join(processed_root, dataset, "images", f"{img_name}.png")


def get_dataset_dict(dataset, processed_root, features_name):
    """(img_name2label, directory of the dataset's <features_name> features)."""
    feat_root = os.path.join(processed_root, dataset, "features", features_name)
    return load_img_name2label(dataset, processed_root), feat_root


def load_all_embeddings_labels(feat_root, img_name2label, dataset_name=None):
    """Stack feat_root/<img_name>.pt for every image in img_name2label that has features.
    Returns (embeddings (N, D) array, labels, img_names), in img_name2label's order."""
    embeddings, labels, img_names = [], [], []
    for img_name, label in tqdm.tqdm(img_name2label.items(), desc="Loading features"):
        feat_path = os.path.join(feat_root, f"{img_name}.pt")
        if not os.path.exists(feat_path):
            continue
        embeddings.append(torch.load(feat_path))
        labels.append(label)
        img_names.append(img_name)
    if len(img_names) < len(img_name2label):
        print(f"WARNING: {len(img_name2label) - len(img_names)} of {len(img_name2label)} images "
              f"have no features under {feat_root}, skipping them")
    return torch.stack(embeddings).numpy().astype(np.float32), labels, img_names
