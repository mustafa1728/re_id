import os
import json
import time
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def reduce_dimensions(embeddings, method="PCA", dim=128, random_state=42):
    if method.lower() == "pca":
        print(f"\n--- PCA reduced to {dim} dimensions ---")
        pca = PCA(n_components=dim, random_state=random_state)
        reduced_embeddings = pca.fit_transform(embeddings)
    elif method.lower() == "umap":
        print(f"\n--- UMAP reduced to {dim} dimensions ---")
        reducer = umap.UMAP(n_components=dim, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(embeddings)
    reduced_embeddings_norm = normalize(reduced_embeddings, norm='l2')
    return reduced_embeddings_norm


def reduce_dimensions_cached(embeddings, img_names, cache_path, method="PCA", dim=128, random_state=42):
    """reduce_dimensions, saved to / loaded from cache_path (.npz) so that every run on
    the same dataset, features, method and dim uses the exact same embedding. UMAP is
    not bit-reproducible across nodes even with a fixed random_state, so recomputing it
    per run gives each run a slightly different embedding. The cache stores img_names
    too and is recomputed if they don't match the current image set and order."""
    if os.path.exists(cache_path):
        with np.load(cache_path) as cached:
            if cached["img_names"].tolist() == list(img_names):
                print(f"\n--- Loaded cached {method.upper()} {dim}D embeddings from {cache_path} ---")
                return cached["embeddings"]
        print(f"Cached embeddings at {cache_path} are for a different image set, recomputing")

    reduced = reduce_dimensions(embeddings, method=method, dim=dim, random_state=random_state)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    # Written to a temp file and renamed into place, so a job killed mid-write can't
    # leave a truncated cache for later runs to load.
    tmp_path = f"{cache_path}.tmp.{os.getpid()}.npz"
    np.savez(tmp_path, embeddings=reduced, img_names=np.array(img_names))
    os.replace(tmp_path, cache_path)
    print(f"Saved {method.upper()} {dim}D embeddings to {cache_path}")
    return reduced

    