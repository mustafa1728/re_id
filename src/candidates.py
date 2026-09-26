"""The candidate clusterings C^1..C^M of one dataset (Sec 3.1), from cluster_benchmark.py's predictions:

    <predictions_root>/<run>/predictions/<dataset>/<dataset>_<features>_preds.json
        {"<dim>/<algo>": {"nmi": ..., "silhouette": ..., ..., "preds": {img_name: cluster_id}}}

A candidate's key is "<run>/<features>/<dim>/<algo>", e.g. "umap_16/dinov3_vith16plus/UMAP_16D/KMeans_k5".
"""
import os
import json
import hashlib
from glob import glob
from functools import cached_property
from dataclasses import dataclass

import numpy as np

from .metrics import SUPERVISED_METRICS, supervised_metrics, cluster_discovery_error
from .reid_datasets import load_img_name2label

# Supervised metric -> field cluster_benchmark.py stores it under
STORED_METRIC_FIELDS = {"id_f1": "id_f1_weighted", "id_f1_macro": "id_f1_macro", "nmi": "nmi", "ari": "ari",
                        "homogeneity": "homogeneity", "completeness": "completeness", "v_measure": "v_measure"}
UNSUPERVISED_METRICS = ["silhouette", "dbi", "calinski_harabasz"]

FEATURE_NAMES = {"dinov3_vith16plus": "DINOv3", "clip_vith14": "CLIP", "ijepa_vith14": "I-JEPA",
                 "imagenet_vith14": "ViT-ImageNet"}

NOISE = -1  # HDBSCAN/OPTICS noise label; like cluster_benchmark.py's metrics, treated as one more cluster


@dataclass
class Candidates:
    dataset: str
    keys: list              # (M,) candidate keys
    img_names: list         # (N,) images, the columns of `labels`
    labels: np.ndarray      # (M, N) int32, labels[j, i] = z^j_i
    gt: np.ndarray          # (N,) int, true cluster index y_i
    metrics: dict           # SUPERVISED_METRICS + "n_clusters" + "ek" -> (M,) values
    unsup: dict             # UNSUPERVISED_METRICS -> (M,) values, NaN where undefined

    @property
    def M(self):
        return self.labels.shape[0]

    @property
    def N(self):
        return self.labels.shape[1]

    @property
    def true_n_clusters(self):
        return len(np.unique(self.gt))

    @cached_property
    def labels_T(self):
        """(N, M) copy of labels, so the M votes on a pair (i, l) are two contiguous row reads."""
        return np.ascontiguousarray(self.labels.T)

    @cached_property
    def fingerprint(self):
        h = hashlib.md5("\n".join(self.keys).encode())
        h.update(self.labels.tobytes())
        return h.hexdigest()[:12]

    def to_json(self):
        rows = []
        for j, key in enumerate(self.keys):
            run, features, dim, algo = key.split("/")
            rows.append({
                "key": key, "run": run, "features": features, "feature": FEATURE_NAMES.get(features, features),
                "dim": dim, "algo": algo, "family": algo_family(algo),
                **{m: v[j] for m, v in self.metrics.items()},
                **{m: v[j] for m, v in self.unsup.items()},
            })
        return {"dataset": self.dataset, "n_points": self.N, "true_n_clusters": self.true_n_clusters,
                "candidates": rows}


def algo_family(algo):
    """"KMeans_k5" -> "KMeans", "HDBSCAN_mcs2_ms1" -> "HDBSCAN", "OPTICS_eps_q0.800" -> "OPTICS"."""
    return algo.split("_")[0]


def find_prediction_files(predictions_root, dataset):
    """[(run, features, path)] for every prediction file of `dataset` under predictions_root or one of
    its direct sub-folders (one per run, e.g. umap_16/ and umap_64/); run is "root" at the top level."""
    found = []
    for run_dir in [predictions_root] + sorted(glob(os.path.join(predictions_root, "*", ""))):
        run = "root" if run_dir == predictions_root else os.path.basename(os.path.normpath(run_dir))
        for path in sorted(glob(os.path.join(run_dir, "predictions", dataset, f"{dataset}_*_preds.json"))):
            features = os.path.basename(path)[len(dataset) + 1:-len("_preds.json")]
            found.append((run, features, path))
    return found


def list_datasets(predictions_root):
    pattern = os.path.join(predictions_root, "**", "predictions", "*", "")
    return sorted({os.path.basename(os.path.normpath(d)) for d in glob(pattern, recursive=True)})


def _unsup_value(entry, name):
    """cluster_benchmark.py stores -1.0 when an index is undefined (fewer than 2 clusters, or all
    singletons). DBI and CHI are always > 0 and a silhouette of exactly -1 does not happen, so -1.0 is
    unambiguous. It must not be read literally: a DBI of -1 would look like the best clustering."""
    value = entry.get(name)
    return np.nan if value is None or value == -1.0 else float(value)


def load_candidates(dataset, predictions_root, processed_root):
    """Every candidate clustering of `dataset`, restricted to the images all of them (and the ground
    truth) cover."""
    img_name2label = load_img_name2label(dataset, processed_root)
    img_names = sorted(img_name2label)
    column = {img: i for i, img in enumerate(img_names)}

    keys, rows, n_preds, stored = [], [], [], []
    files = find_prediction_files(predictions_root, dataset)
    if not files:
        raise FileNotFoundError(f"No predictions for {dataset!r} under {predictions_root}")
    for run, features, path in files:
        try:
            with open(path) as f:
                entries = json.load(f)
        except json.JSONDecodeError:
            print(f"[{dataset}] WARNING: skipping unreadable {path} (still being written?)")
            continue
        for algo_key, entry in entries.items():
            preds = entry.pop("preds")
            cols = np.array([column.get(img, -1) for img in preds])
            row = np.full(len(img_names), np.iinfo(np.int32).min, dtype=np.int32)
            row[cols[cols >= 0]] = np.fromiter(preds.values(), np.int32, len(preds))[cols >= 0]
            keys.append(f"{run}/{features}/{algo_key}")
            rows.append(row)
            n_preds.append(len(preds))
            stored.append(entry)

    order = np.argsort(keys)
    keys = [keys[j] for j in order]
    labels = np.stack([rows[j] for j in order])
    n_preds = np.array(n_preds)[order]
    stored = [stored[j] for j in order]

    covered = (labels != np.iinfo(np.int32).min).all(axis=0)
    if not covered.all():
        print(f"[{dataset}] {int((~covered).sum())} of {len(img_names)} images are missing from some "
              f"candidates -- using the {int(covered.sum())} covered by all of them")
    labels = np.ascontiguousarray(labels[:, covered])
    img_names = [img for img, c in zip(img_names, covered) if c]
    _, gt = np.unique([img_name2label[img] for img in img_names], return_inverse=True)

    # Stored metrics were computed on each candidate's own image set; recompute where that differs
    metrics = {m: np.array([e[STORED_METRIC_FIELDS[m]] for e in stored], dtype=float) for m in SUPERVISED_METRICS}
    for j in np.flatnonzero(n_preds != labels.shape[1]):
        for m, v in supervised_metrics(gt, labels[j]).items():
            metrics[m][j] = v
    n_clusters = np.array([len(np.unique(row[row != NOISE])) for row in labels])
    metrics["n_clusters"] = n_clusters.astype(float)
    metrics["ek"] = np.array([cluster_discovery_error(k, len(np.unique(gt))) for k in n_clusters])

    unsup = {m: np.array([_unsup_value(e, m) for e in stored]) for m in UNSUPERVISED_METRICS}
    return Candidates(dataset, keys, img_names, labels, gt, metrics, unsup)
