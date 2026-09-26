import sys
import time
import numpy as np
import pandas as pd
import tqdm
import os
import json
import warnings
import argparse

import torch
import torch.nn.functional as F



from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from collections import defaultdict


from src.config import PROCESSED_DATA_ROOT
from src.reid_datasets import get_dataset_dict, load_all_embeddings_labels
from src.dim_reduce import reduce_dimensions_cached
from src.clustering import get_clustering_algorithms

# Suppress minor warnings for clean output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


argparser = argparse.ArgumentParser(description="Clustering Benchmark for Re-ID Embeddings")
argparser.add_argument("--features", type=str, default="dinov3")
argparser.add_argument("--dataset", type=str, default="cars")
argparser.add_argument("--root_dir", type=str, default=PROCESSED_DATA_ROOT,
                       help="Directory containing the datasets: the output of src/process_datasets.py (default), "
                            "or the original downloads for the older per-dataset loaders")
argparser.add_argument("--save_root", type=str, default="./benchmark_results/latest", help="Directory to save benchmark results")

argparser.add_argument("--db_only", action="store_true", help="Only run DBSCAN and HDBSCAN for a quicker benchmark")
argparser.add_argument("--grid_search", action="store_true", help="run hyperparameter grid search over KMeans, Agglomerative, Spectral, GMM, Mean-shift, FINCH, HDBSCAN, OPTICS, Affinity propagation, and Dirichlet-process GMM")
argparser.add_argument("--dim", type=int, default=-1)
argparser.add_argument("--dim_method", type=str, default="pca")
argparser.add_argument("--skip_algo", type=str, default="",
                       help="comma-separated algorithm name prefixes to skip, e.g. AffinityPropagation")
argparser.add_argument("--only_algo", type=str, default="",
                       help="comma-separated algorithm name prefixes to run exclusively, e.g. AffinityPropagation")
argparser.add_argument("--save_each_algo", action="store_true", help="Save results after each algorithm to avoid losing progress")
args = argparser.parse_args()


def compute_id_f1(labels, cluster_labels):
    """
    ID-F1 using global optimal one-to-one matching (Hungarian algorithm).

    Builds a GT-identity × predicted-cluster contingency matrix, finds the
    assignment that maximises total overlap, then computes global TP/FP/FN.

    Returns (id_f1_macro, id_f1_weighted):
      - id_f1_macro: unmatched GT identities contribute 0; matched ones
        contribute their per-identity F1; plain mean across all GT identities.
      - id_f1_weighted: same but weighted by GT identity size.
    """
    labels         = np.asarray(labels)
    cluster_labels = np.asarray(cluster_labels)

    gt_ids   = np.unique(labels)
    pred_ids = np.unique(cluster_labels)
    G, P     = len(gt_ids), len(pred_ids)

    gt_idx   = {v: i for i, v in enumerate(gt_ids)}
    pred_idx = {v: i for i, v in enumerate(pred_ids)}

    # Build G×P contingency matrix
    C = np.zeros((G, P), dtype=np.int64)
    for gt, pred in zip(labels, cluster_labels):
        C[gt_idx[gt], pred_idx[pred]] += 1

    # Hungarian: maximise overlap → minimise negative overlap
    row_ind, col_ind = linear_sum_assignment(-C)

    # Per-GT-identity sizes and predicted-cluster sizes
    gt_sizes   = np.array([np.sum(labels == gt) for gt in gt_ids])
    pred_sizes = np.array([np.sum(cluster_labels == p) for p in pred_ids])

    f1_scores = np.zeros(G)
    for r, c in zip(row_ind, col_ind):
        tp        = C[r, c]
        precision = tp / pred_sizes[c] if pred_sizes[c] > 0 else 0.0
        recall    = tp / gt_sizes[r]   if gt_sizes[r]   > 0 else 0.0
        f1_scores[r] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    id_f1_macro    = float(np.mean(f1_scores))
    id_f1_weighted = float(np.average(f1_scores, weights=gt_sizes))
    return id_f1_macro, id_f1_weighted


def compute_inertia(embeddings, cluster_labels):
    """Sum of squared distances of each point to its cluster's centroid (noise label -1 excluded)."""
    inertia = 0.0
    for label in set(cluster_labels):
        if label == -1:
            continue
        cluster_points = embeddings[cluster_labels == label]
        centroid = cluster_points.mean(axis=0)
        inertia += np.sum((cluster_points - centroid) ** 2)
    return inertia


def benchmark_clustering(embeddings, labels, algorithms, img_names=None, save_fn=None, save_preds_fn=None):
    """
    Runs and times a dictionary of clustering algorithms, returning
    a comprehensive suite of ground-truth-based quality metrics.
    """
    results = {}
    preds = {}

    header = f"{'Algorithm':<22} | {'Time(s)':<8} | {'ARI':<6} | {'NMI':<6} | {'IDF1-m':<6} | {'IDF1-w':<6} | {'Homog':<6} | {'Compl':<6} | {'V-Meas':<6} | {'Silh':<6} | {'Clusts'}"
    print(header)
    print("-" * len(header))

    for name, model in algorithms.items():
        # time.perf_counter is strictly better for benchmarking than time.time
        start_time = time.perf_counter()

        # Fit and Predict
        if hasattr(model, 'fit_predict'):
            cluster_labels = model.fit_predict(embeddings)
        else:
            model.fit(embeddings)
            cluster_labels = model.labels_

        elapsed_time = time.perf_counter() - start_time

        # --- Ground Truth Metrics ---
        ari = adjusted_rand_score(labels, cluster_labels)
        nmi = normalized_mutual_info_score(labels, cluster_labels)
        homog = homogeneity_score(labels, cluster_labels)
        compl = completeness_score(labels, cluster_labels)
        v_meas = v_measure_score(labels, cluster_labels)
        id_f1_macro, id_f1_weighted = compute_id_f1(labels, cluster_labels)

        # --- Internal Metric (Silhouette) ---
        cluster_labels = np.asarray(cluster_labels)

        # Determine number of valid clusters (ignoring noise label -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        # sklearn's internal metrics need 2 to n_samples - 1 distinct labels (noise
        # counts as a label), so e.g. every point in its own cluster is invalid.
        n_labels = len(np.unique(cluster_labels))

        if n_clusters > 1 and n_labels < len(cluster_labels):
            # Silhouette is O(N^2), so score a random subsample of up to 10000 points.
            # Drawn here rather than via silhouette_score(sample_size=...) so the same
            # label-count limit can be checked on the subsample: when nearly every
            # cluster is a singleton, the subsample can hit it even if the full set doesn't.
            sub_idx = np.random.permutation(len(cluster_labels))[:10000]
            sub_labels = cluster_labels[sub_idx]
            if 1 < len(np.unique(sub_labels)) < len(sub_idx):
                silhouette = float(silhouette_score(embeddings[sub_idx], sub_labels))
            else:
                silhouette = -1.0 # Invalid
            dbi = float(davies_bouldin_score(embeddings, cluster_labels))
            ch_index = float(calinski_harabasz_score(embeddings, cluster_labels))
        else:
            silhouette = -1.0 # Invalid
            dbi = -1.0 # Invalid
            ch_index = -1.0 # Invalid

        # Reuse KMeans' own inertia_ when available (cheap, already computed during fit);
        # otherwise compute it directly from the embeddings/labels for any other estimator.
        inertia = getattr(model, 'inertia_', None)
        if inertia is None:
            inertia = getattr(getattr(model, 'model_', None), 'inertia_', None)
        if inertia is None:
            inertia = compute_inertia(embeddings, cluster_labels)
        inertia = float(inertia)

        results[name] = {
            'time': elapsed_time,
            'ari': ari,
            'nmi': nmi,
            'homogeneity': homog,
            'completeness': compl,
            'v_measure': v_meas,
            'id_f1_macro': id_f1_macro,
            'id_f1_weighted': id_f1_weighted,
            'silhouette': silhouette,
            'dbi': dbi,
            'calinski_harabasz': ch_index,
            'inertia': inertia,
            'n_clusters': n_clusters,
            'true_clusters': len(set(labels)),
            'examples': len(labels),
        }

        if img_names is not None:
            preds[name] = {
                'nmi': nmi,
                'ari': ari,
                'homogeneity': homog,
                'completeness': compl,
                'v_measure': v_meas,
                'id_f1_macro': id_f1_macro,
                'id_f1_weighted': id_f1_weighted,
                'n_clusters': n_clusters,
                'silhouette': silhouette,
                'dbi': dbi,
                'calinski_harabasz': ch_index,
                'inertia': inertia,
                'preds': {img: int(cid) for img, cid in zip(img_names, cluster_labels)},
            }

        print(f"{name:<22} | {elapsed_time:<8.4f} | {ari:<6.3f} | {nmi:<6.3f} | {id_f1_macro:<6.3f} | {id_f1_weighted:<6.3f} | {homog:<6.3f} | {compl:<6.3f} | {v_meas:<6.3f} | {silhouette:<6.3f} | {n_clusters}")
        if save_fn is not None:
            save_fn(results)
        if save_preds_fn is not None and img_names is not None:
            save_preds_fn(preds)

    return results, preds

if __name__ == "__main__":

    save_root = args.save_root
    save_path = os.path.join(save_root, f"{args.dataset}/{args.dataset}_{args.features}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # if args.db_only:
    #     save_path = os.path.join(save_root, f"{args.dataset}_{args.features}_db_only.json")
    # if args.only_dim != -1:
    #     save_path = save_path.replace(".json", f"_dim{args.only_dim}.json")

    

    img_name2label, FEAT_ROOT = get_dataset_dict(
        args.dataset, 
        args.root_dir, 
        args.features
    )



    all_labels = list(img_name2label.values())
    N_TRUE_CLUSTERS = len(set(all_labels))
    print(f"Unique labels: {N_TRUE_CLUSTERS}")


    embeddings, labels, img_names = load_all_embeddings_labels(
        FEAT_ROOT,
        img_name2label,
        args.dataset
    )
    print(f"Loaded embeddings shape: {embeddings.shape}")

    # L2 Normalization maps Euclidean distance to Cosine similarity
    print("Applying L2 Normalization...")
    embeddings_normalized = normalize(embeddings, norm='l2')

    # --- Initialize Models ---
    print("Initializing models...")
    clustering_algorithms = get_clustering_algorithms(
        N_TRUE_CLUSTERS, args.db_only, args.skip_algo,
        grid_search=args.grid_search, n_points=embeddings.shape[0],
        only_algo=args.only_algo
    )
    
    

    result_key = "Original"
    if args.dim != -1:
        n_points = embeddings_normalized.shape[0]
        if args.dim_method.lower() == "umap":
            max_dim = n_points - 2
        else:
            max_dim = min(n_points, embeddings_normalized.shape[1])
        if args.dim > max_dim:
            print(f"ERROR: --dim {args.dim} ({args.dim_method}) is not valid for "
                  f"{n_points} points (max supported dim is {max_dim}). Exiting.")
            sys.exit(0)

        # Cached next to the features (FEAT_ROOT is <data>/<dataset>/features/<features>)
        # as <data>/<dataset>/reduced_features/<features>/, keyed by the full --dataset
        # name since modifiers like cls@/img@ change the image set.
        data_dir = os.path.dirname(os.path.dirname(os.path.normpath(FEAT_ROOT)))
        cache_path = os.path.join(
            data_dir, "reduced_features", args.features,
            f"{args.dataset}_{args.dim_method.lower()}{args.dim}_rs42.npz"
        )
        embeddings_normalized = reduce_dimensions_cached(
            embeddings_normalized,
            img_names,
            cache_path,
            method=args.dim_method,
            dim=args.dim,
            random_state=42
        )
        result_key = f"{args.dim_method.upper()}_{args.dim}D"

    preds_dir = os.path.join(save_root, "predictions", args.dataset)
    os.makedirs(preds_dir, exist_ok=True)
    preds_path = os.path.join(preds_dir, f"{args.dataset}_{args.features}_preds.json")

    def save_fn(intermediate_results):
        all_results = {}
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                all_results = json.load(f)
        all_results[result_key] = intermediate_results
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=4)

    def save_preds_fn(intermediate_preds):
        all_preds = {}
        if os.path.exists(preds_path):
            with open(preds_path, "r") as f:
                all_preds = json.load(f)
        for algo_name, entry in intermediate_preds.items():
            all_preds[f"{result_key}/{algo_name}"] = entry
        with open(preds_path, "w") as f:
            json.dump(all_preds, f, indent=4)

    cur_results, cur_preds = benchmark_clustering(
        embeddings_normalized,
        labels,
        clustering_algorithms,
        img_names=img_names,
        save_fn=save_fn if args.save_each_algo else None,
        save_preds_fn=save_preds_fn if args.save_each_algo else None,
    )
    save_fn(cur_results)
    save_preds_fn(cur_preds)

    print("\n\n" + "="*50)
    print(f"--- Final Benchmark Results for {result_key} ---")
    print("Algorithm,Time(s),Homogeneity,Completeness,V-Measure,ID-F1-macro,ID-F1-weighted,N_Clusters")
    print("="*50)
    
    for alg, metrics in cur_results.items():
        print(f"{alg},{metrics['time']:.4f},{metrics['homogeneity']:.3f},{metrics['completeness']:.3f},{metrics['v_measure']:.3f},{metrics['id_f1_macro']:.3f},{metrics['id_f1_weighted']:.3f},{metrics['n_clusters']}")