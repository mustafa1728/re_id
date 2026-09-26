import numpy as np
from sklearn.cluster import (
    DBSCAN, 
    MiniBatchKMeans, 
    BisectingKMeans, 
    SpectralClustering, 
    KMeans,
    AgglomerativeClustering,
    OPTICS,
    MeanShift,
    AffinityPropagation,
    Birch,
    estimate_bandwidth,
    cluster_optics_dbscan
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import SpectralEmbedding
from scipy.cluster.hierarchy import linkage
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import davies_bouldin_score



from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline

# --- FINCH Import ---
try:
    from finch import FINCH
except ImportError:
    raise ImportError("Please install finch-clust: pip install finch-clust")

import hdbscan

# --- Sklearn Wrapper for FINCH ---
class FINCHWrapper(BaseEstimator, ClusterMixin):
    """
    Wraps the FINCH algorithm in a scikit-learn compatible API.
    When req_clust is None, it selects a clustering from the natural hierarchy.
    """
    def __init__(self, req_clust=None, distance='cosine', partition_idx=0):
        self.req_clust = req_clust
        self.distance = distance
        self.partition_idx = partition_idx
        self.labels_ = None
        
    def fit_predict(self, X, y=None):
        # FINCH returns: c (partitions), num_clust (clusters per partition), req_c (labels for req_clust)
        c, num_clust, req_c = FINCH(X, req_clust=self.req_clust, distance=self.distance, verbose=False)
        
        if self.req_clust is not None and req_c is not None:
            # Flatten to ensure it matches sklearn's expected 1D array
            self.labels_ = req_c.flatten()
        else:
            # Select the specified partition level from the hierarchy
            # partition_idx = 0 is the finest clustering, -1 is the coarsest
            # Make sure the requested index doesn't exceed the number of partitions found
            idx = min(self.partition_idx, c.shape[1] - 1)
            self.labels_ = c[:, idx].flatten()
            
        return self.labels_
        
    def fit(self, X, y=None):
        self.fit_predict(X)
        return self

class KMeansUnknown(BaseEstimator, ClusterMixin):
    """
    Automates KMeans for high-k regimes by executing a strided search 
    using MiniBatchKMeans to rapidly find the geometric elbow, followed 
    by a final fit at the optimal k.
    """
    def __init__(self, k_min=50, k_max=2000, step=30, random_state=42):
        self.k_min = k_min
        self.k_max = k_max
        self.step = step
        self.random_state = random_state

    def fit_predict(self, X, y=None):
        inertias = []
        k_range = list(range(self.k_min, min(self.k_max + 1, X.shape[0]), self.step))
        
        # Fast search using MiniBatchKMeans
        for k in k_range:
            km = MiniBatchKMeans(n_clusters=k, n_init='auto', random_state=self.random_state, batch_size=2048)
            km.fit(X)
            inertias.append(km.inertia_)
            
        # Geometric Elbow calculation
        p1 = np.array([k_range[0], inertias[0]])
        p2 = np.array([k_range[-1], inertias[-1]])
        
        best_k = self.k_min
        max_dist = -1
        for i, k in enumerate(k_range):
            p0 = np.array([k, inertias[i]])
            # Distance from point p0 to the line connecting p1 and p2
            dist = np.abs(np.cross(p2 - p1, p1 - p0)) / np.linalg.norm(p2 - p1)
            if dist > max_dist:
                max_dist = dist
                best_k = k
                
        # Final robust fit at the optimal k
        self.model_ = KMeans(n_clusters=best_k, n_init='auto', random_state=self.random_state, max_iter=20, tol=0.0)
        self.labels_ = self.model_.fit_predict(X)
        return self.labels_

    def fit(self, X, y=None):
        self.fit_predict(X)
        return self

class AgglomerativeUnknown(BaseEstimator, ClusterMixin):
    """
    Automates Agglomerative Clustering by building a dendrogram and 
    finding the geometric elbow of the merge distances, avoiding the 
    bias toward k=2/k=3 caused by the absolute largest vertical gap.
    """
    def __init__(self, k_min=50, k_max=2000):
        self.k_min = k_min
        self.k_max = k_max

    def _get_elbow(self, k_values, distances):
        """Helper to compute the geometric elbow from a curve."""
        p1 = np.array([k_values[0], distances[0]])
        p2 = np.array([k_values[-1], distances[-1]])
        
        best_k = k_values[0]
        max_dist = -1
        
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0: 
            return best_k
            
        for i, k in enumerate(k_values):
            p0 = np.array([k, distances[i]])
            dist = np.abs(np.cross(line_vec, p1 - p0)) / line_len
            if dist > max_dist:
                max_dist = dist
                best_k = k
                
        return best_k

    def fit_predict(self, X, y=None):
        # Compute Ward linkage (O(N^2) time complexity)
        Z = linkage(X, method='ward')
        distances = Z[:, 2] 
        N = X.shape[0]
        
        # Safely bound the search limits
        actual_k_max = min(self.k_max, N - 1)
        actual_k_min = max(2, self.k_min)
        
        # Build the curve of (k vs. merge_distance)
        # Z[:, 2] is sorted ascending (from N clusters down to 1 cluster)
        # Z[N - k - 1, 2] gives the merge distance that resulted in k clusters
        k_values = []
        merge_distances = []
        
        # We build the lists from k_min to k_max (ascending k)
        for k in range(actual_k_min, actual_k_max + 1):
            k_values.append(k)
            merge_distances.append(distances[N - k - 1])
            
        # Find the elbow on this curve
        best_k = self._get_elbow(k_values, merge_distances)

        # Final fit
        self.model_ = AgglomerativeClustering(n_clusters=best_k)
        self.labels_ = self.model_.fit_predict(X)
        return self.labels_
        
    def fit(self, X, y=None):
        self.fit_predict(X)
        return self


class SpectralUnknown(BaseEstimator, ClusterMixin):
    """
    Automates Spectral Clustering using the Eigengap heuristic.
    Note: Extracting thousands of eigenvalues is computationally 
    heavy. Sparse solvers are enforced where possible.
    """
    def __init__(self, k_min=50, k_max=2000, n_components=128, n_neighbors=10, random_state=42):
        self.k_min = k_min
        self.k_max = k_max
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_predict(self, X, y=None):
        A = kneighbors_graph(X, n_neighbors=self.n_neighbors, mode='connectivity', include_self=True)
        A = 0.5 * (A + A.T) 
        L = laplacian(A, normed=True)
        
        k_search = min(self.k_max + 2, X.shape[0] - 1)
        L_float = L.astype(np.float64)
        
        try:
            eigenvalues, _ = eigsh(L_float, k=k_search, which='SM', tol=1e-2)
        except:
            # Fallback to dense eigh. This will consume high RAM for large N.
            eigenvalues, _ = eigh(L.toarray())
            
        eigenvalues = np.sort(eigenvalues)
        
        gaps = np.diff(eigenvalues)
        
        # Ensure we don't index out of bounds if eigh returned fewer values
        search_limit = min(self.k_max, len(gaps))
        search_gaps = gaps[self.k_min - 1 : search_limit]
        
        if len(search_gaps) > 0:
            best_k = np.argmax(search_gaps) + self.k_min
        else:
            best_k = self.k_min
        
        self.model_ = SpectralClustering(
            n_clusters=best_k, 
            n_components=self.n_components,
            affinity='nearest_neighbors', 
            n_neighbors=self.n_neighbors, 
            random_state=self.random_state, 
            n_jobs=-1
        )
        self.labels_ = self.model_.fit_predict(X)
        return self.labels_
        
    def fit(self, X, y=None):
        self.fit_predict(X)
        return self



# class ClusterSearchWrapper(BaseEstimator, ClusterMixin):
#     """
#     A generic wrapper that finds the optimal k by minimizing the Davies-Bouldin Index.
#     It takes a variable list of steps to perform a dynamic, multi-level hierarchical 
#     search on a subset of the data before a final fit.
#     """
#     def __init__(self, estimator_class, estimator_kwargs=None, 
#                  k_min=50, k_max=2000, steps=(50, 5, 1), 
#                  max_search_samples=10000, random_state=42):
#         self.estimator_class = estimator_class
#         self.estimator_kwargs = estimator_kwargs if estimator_kwargs is not None else {}
#         self.k_min = k_min
#         self.k_max = k_max
#         self.steps = steps
#         self.max_search_samples = max_search_samples
#         self.random_state = random_state

#     def _evaluate_k(self, X, k):
#         """Helper to fit a model and compute the Davies-Bouldin score."""
#         model = self.estimator_class(n_clusters=k, **self.estimator_kwargs)
#         labels = model.fit_predict(X)
        
#         n_labels = len(set(labels)) - (1 if -1 in labels else 0)
#         if 1 < n_labels < X.shape[0]:
#             # Davies-Bouldin is O(N * k), exceptionally fast for large N
#             return davies_bouldin_score(X, labels)
            
#         # Return infinity because lower is better for Davies-Bouldin
#         return float('inf') 

#     def fit_predict(self, X, y=None):
#         # --- 0. Subsample for Search Phase ---
#         if X.shape[0] > self.max_search_samples:
#             rng = np.random.default_rng(self.random_state)
#             indices = rng.choice(X.shape[0], size=self.max_search_samples, replace=False)
#             X_search = X[indices]
#         else:
#             X_search = X

#         actual_k_max = min(self.k_max, X_search.shape[0] - 1)
#         evaluated_k = {}

#         def _search_range(start, end, step, current_best_k, current_best_score):
#             """Executes a search over a specific range and step, finding the minimum score."""
#             safe_start = max(self.k_min, start)
#             safe_end = min(actual_k_max, end)
            
#             search_list = list(range(safe_start, safe_end + 1, step))
            
#             best_k = current_best_k
#             best_score = current_best_score
            
#             for k in search_list:
#                 if k not in evaluated_k:
#                     score = self._evaluate_k(X_search, k)
#                     evaluated_k[k] = score
#                 else:
#                     score = evaluated_k[k]
                    
#                 if score < best_score: # Lower is better!
#                     best_score = score
#                     best_k = k
                    
#             return best_k, best_score

#         # --- 1. Dynamic Hierarchical Search ---
#         best_k = self.k_min
#         best_score = float('inf')
        
#         for i, step in enumerate(self.steps):
#             if i == 0:
#                 # First level searches the entire absolute range
#                 start_val = self.k_min
#                 end_val = actual_k_max
#             else:
#                 # Subsequent levels search a window around the best_k from the previous level
#                 prev_step = self.steps[i - 1]
#                 start_val = best_k - prev_step
#                 end_val = best_k + prev_step
                
#             best_k, best_score = _search_range(
#                 start=start_val, 
#                 end=end_val, 
#                 step=step, 
#                 current_best_k=best_k, 
#                 current_best_score=best_score
#             )

#         # --- 2. Final Fit on Full Dataset ---
#         self.model_ = self.estimator_class(n_clusters=best_k, **self.estimator_kwargs)
#         self.labels_ = self.model_.fit_predict(X)
#         return self.labels_
        
#     def fit(self, X, y=None):
#         self.fit_predict(X)
#         return self

# class ClusterSearchWrapper(BaseEstimator, ClusterMixin):
#     """
#     A generic wrapper that finds the optimal k by minimizing the Davies-Bouldin Index.
#     It takes a variable list of steps to perform a dynamic, multi-level hierarchical 
#     search on a subset of the data before a final fit.
#     """
#     def __init__(self, estimator_class, estimator_kwargs=None, 
#                  k_min=50, k_max=2000, budget=100, 
#                  max_search_samples=10000, random_state=42):
#         self.estimator_class = estimator_class
#         self.estimator_kwargs = estimator_kwargs if estimator_kwargs is not None else {}
#         self.k_min = k_min
#         self.k_max = k_max
#         self.budget = budget
#         self.max_search_samples = max_search_samples
#         self.random_state = random_state

#     def _evaluate_k(self, X, k):
#         """Helper to fit a model and compute the Davies-Bouldin score."""
#         model = self.estimator_class(n_clusters=k, **self.estimator_kwargs)
#         labels = model.fit_predict(X)
        
#         n_labels = len(set(labels)) - (1 if -1 in labels else 0)
#         if 1 < n_labels < X.shape[0]:
#             # Davies-Bouldin is O(N * k), exceptionally fast for large N
#             return davies_bouldin_score(X, labels)
            
#         # Return infinity because lower is better for Davies-Bouldin
#         return float('inf') 

#     def fit_predict(self, X, y=None):
#         # --- 0. Subsample for Search Phase ---
#         if X.shape[0] > self.max_search_samples:
#             rng = np.random.default_rng(self.random_state)
#             indices = rng.choice(X.shape[0], size=self.max_search_samples, replace=False)
#             X_search = X[indices]
#         else:
#             X_search = X

#         actual_k_max = min(self.k_max, X_search.shape[0] - 1)

#         # --- 1. Random Search ---
#         best_k = self.k_min
#         best_score = float('inf')
        
#         for _ in range(self.budget):
#             # random_k = np.random.randint(self.k_min, actual_k_max + 1)
#             # sample uniformly in log space to better cover the range of k
#             log_k_min = np.log(self.k_min)
#             log_k_max = np.log(actual_k_max)
#             random_k = int(np.exp(np.random.uniform(log_k_min, log_k_max)))

#             score = self._evaluate_k(X_search, random_k)  # Populate evaluated_k with random samples
#             if score < best_score: # Lower is better!
#                 best_score = score
#                 best_k = random_k

#         # --- 2. Final Fit on Full Dataset ---
#         self.model_ = self.estimator_class(n_clusters=best_k, **self.estimator_kwargs)
#         self.labels_ = self.model_.fit_predict(X)
#         return self.labels_
        
#     def fit(self, X, y=None):
#         self.fit_predict(X)
#         return self


class ClusterSearchWrapper(BaseEstimator, ClusterMixin):
    """
    A generic wrapper that finds the optimal k by calculating the Davies-Bouldin Index.
    It combines a log-uniform random search budget with the Min-Max geometric elbow 
    method to prevent over-segmentation in high-dimensional feature spaces.
    """
    def __init__(self, estimator_class, estimator_kwargs=None, 
                 k_min=50, k_max=2000, budget=100, 
                 max_search_samples=10000, random_state=42):
        self.estimator_class = estimator_class
        self.estimator_kwargs = estimator_kwargs if estimator_kwargs is not None else {}
        self.k_min = k_min
        self.k_max = k_max
        self.budget = budget
        self.max_search_samples = max_search_samples
        self.random_state = random_state

    def _evaluate_k(self, X, k):
        """Helper to fit a model and compute the Davies-Bouldin score."""
        model = self.estimator_class(n_clusters=k, **self.estimator_kwargs)
        labels = model.fit_predict(X)
        
        n_labels = len(set(labels)) - (1 if -1 in labels else 0)
        if 1 < n_labels < X.shape[0]:
            # Davies-Bouldin is O(N * k), exceptionally fast for large N
            return davies_bouldin_score(X, labels)
            
        return float('inf') 

    def _get_elbow(self, k_values, scores):
        """
        Computes the geometric elbow of the DBI curve. 
        Filters out invalid (infinity) scores and uses Min-Max scaling 
        to ensure the axes carry equal geometric weight.
        """
        # 1. Filter out invalid clusterings (inf or NaN scores)
        valid_k = []
        valid_scores = []
        for k, s in zip(k_values, scores):
            if s != float('inf') and not np.isnan(s):
                valid_k.append(k)
                valid_scores.append(s)
                
        # 2. Safety check: Ensure we have enough valid points to draw a curve
        if len(valid_k) < 3:
            return valid_k[0] if len(valid_k) > 0 else k_values[0]
            
        k_arr = np.array(valid_k, dtype=np.float64)
        s_arr = np.array(valid_scores, dtype=np.float64)
        
        # 3. Scale both axes to [0, 1]
        k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min() + 1e-9)
        s_norm = (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min() + 1e-9)
        
        p1 = np.array([k_norm[0], s_norm[0]])
        p2 = np.array([k_norm[-1], s_norm[-1]])
        
        best_k = valid_k[0]
        max_dist = -1
        
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return best_k
            
        # 4. Find the point furthest from the chord
        for idx in range(len(valid_k)):
            p0 = np.array([k_norm[idx], s_norm[idx]])
            dist = np.abs(np.cross(line_vec, p1 - p0)) / line_len
            if dist > max_dist:
                max_dist = dist
                best_k = valid_k[idx]
                
        return best_k

    def fit_predict(self, X, y=None):
        # Establish reproducible RNG
        rng = np.random.default_rng(self.random_state)
        
        # --- 0. Subsample for Search Phase ---
        if X.shape[0] > self.max_search_samples:
            indices = rng.choice(X.shape[0], size=self.max_search_samples, replace=False)
            X_search = X[indices]
        else:
            X_search = X

        actual_k_max = min(self.k_max, X_search.shape[0] - 1)
        evaluated_k = {}

        # --- 1. Coarse Random Search (80% of budget) ---
        coarse_budget = int(self.budget * 0.8)
        log_k_min = np.log(self.k_min)
        log_k_max = np.log(actual_k_max)
        
        for _ in range(coarse_budget):
            random_k = int(np.exp(rng.uniform(log_k_min, log_k_max)))
            if random_k not in evaluated_k:
                evaluated_k[random_k] = self._evaluate_k(X_search, random_k)

        # Build the initial curve and find the rough elbow
        coarse_k_values = sorted(list(evaluated_k.keys()))
        coarse_scores = [evaluated_k[k] for k in coarse_k_values]
        coarse_best_k = self._get_elbow(coarse_k_values, coarse_scores)

        # --- 2. Fine Local Search (20% of budget) ---
        fine_budget = self.budget - coarse_budget
        
        # Define a tight search window centered around the coarse elbow
        fine_half_window = fine_budget // 2
        fine_min = max(self.k_min, coarse_best_k - fine_half_window)
        fine_max = min(actual_k_max, coarse_best_k + fine_half_window)
        
        # Determine candidates in the local window
        fine_candidates = list(range(fine_min, fine_max + 1))
        
        # If the window is larger than the fine budget, sample uniformly within it
        if len(fine_candidates) > fine_budget:
            rng.shuffle(fine_candidates)
            fine_candidates = fine_candidates[:fine_budget]
            
        for k in fine_candidates:
            if k not in evaluated_k:
                evaluated_k[k] = self._evaluate_k(X_search, k)

        # --- 3. Final Elbow Calculation ---
        # Re-evaluate the geometric elbow on the finalized, dense curve
        all_k_values = sorted(list(evaluated_k.keys()))
        all_scores = [evaluated_k[k] for k in all_k_values]
        final_best_k = self._get_elbow(all_k_values, all_scores)

        # --- 4. Final Fit on Full Dataset ---
        self.model_ = self.estimator_class(n_clusters=final_best_k, **self.estimator_kwargs)
        self.labels_ = self.model_.fit_predict(X)
        return self.labels_
        
    def fit(self, X, y=None):
        self.fit_predict(X)
        return self

class SpectralHDBSCAN(BaseEstimator, ClusterMixin):
    """
    A hybrid clustering algorithm that uses Spectral Embedding for non-linear 
    dimensionality reduction (manifold learning), followed by HDBSCAN to find 
    density-based clusters without requiring an apriori cluster count.
    """
    def __init__(self, n_components=10, n_neighbors=10, 
                 min_cluster_size=5, min_samples=None, 
                 random_state=42, n_jobs=-1):
        
        # Spectral Embedding Parameters
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        
        # HDBSCAN Parameters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        
        # Global Parameters
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit_predict(self, X, y=None):
        # --- 1. Spectral Dimensionality Reduction ---
        # This builds the k-NN graph, computes the normalized Laplacian, 
        # and extracts the first `n_components` eigenvectors.
        spectral = SpectralEmbedding(
            n_components=self.n_components,
            affinity='nearest_neighbors',
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # The data is now projected into the spectral manifold space
        self.spectral_embeddings_ = spectral.fit_transform(X)

        # --- 2. HDBSCAN Clustering ---
        # We run density-based clustering on the unwrapped eigenvectors.
        # No 'k' is required here.
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            core_dist_n_jobs=self.n_jobs
        )
        
        self.labels_ = clusterer.fit_predict(self.spectral_embeddings_)
        
        # Expose HDBSCAN's soft-clustering probabilities for downstream use
        self.probabilities_ = clusterer.probabilities_
        
        return self.labels_
        
    def fit(self, X, y=None):
        self.fit_predict(X)
        return self
    

class SpectralFINCH(BaseEstimator, ClusterMixin):
    """
    A hybrid clustering algorithm that uses Spectral Embedding for non-linear
    dimensionality reduction (manifold learning), followed by FINCH to find
    clusters hierarchically without requiring an apriori cluster count.
    """
    def __init__(self, n_components=10, n_neighbors=10,
                 distance='cosine', partition_idx=2,
                 random_state=42, n_jobs=-1):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.partition_idx = partition_idx
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit_predict(self, X, _y=None):
        spectral = SpectralEmbedding(
            n_components=self.n_components,
            affinity='nearest_neighbors',
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        self.spectral_embeddings_ = spectral.fit_transform(X)

        c, _, _ = FINCH(self.spectral_embeddings_, req_clust=None,
                        distance=self.distance, verbose=False)
        idx = min(self.partition_idx, c.shape[1] - 1)
        self.labels_ = c[:, idx].flatten()
        return self.labels_

    def fit(self, X, _y=None):
        self.fit_predict(X)
        return self


class MeanShiftK(BaseEstimator, ClusterMixin):
    """
    Mean-shift parameterised by a target number of clusters instead of a bandwidth.
    The bandwidth is sklearn's estimate_bandwidth with quantile = 1 / n_clusters,
    i.e. the mean distance to each point's (N / n_clusters)-th nearest neighbour:
    the neighbourhood size a cluster would have if the N points were split evenly
    into n_clusters. The resulting cluster count is data-dependent, so it only
    tracks n_clusters approximately.
    """
    def __init__(self, n_clusters=8, n_jobs=-1):
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs

    def fit_predict(self, X, y=None):
        # Floor at 2 neighbours: with 1, the only neighbour is the point itself
        # (distance 0), which gives a zero bandwidth that MeanShift rejects.
        quantile = min(1.0, max(1.0 / self.n_clusters, 2.0 / X.shape[0]))
        self.bandwidth_ = estimate_bandwidth(X, quantile=quantile, n_jobs=self.n_jobs)
        self.model_ = MeanShift(bandwidth=self.bandwidth_, n_jobs=self.n_jobs)
        self.labels_ = self.model_.fit_predict(X)
        return self.labels_

    def fit(self, X, y=None):
        self.fit_predict(X)
        return self


class AffinityPropagationQuantile(BaseEstimator, ClusterMixin):
    """
    Affinity propagation with the preference given as a quantile in [0, 1] of the
    pairwise similarities (negative squared Euclidean distances, sklearn's default
    affinity) rather than as a raw similarity value. 0.5 is ~sklearn's default
    (the median similarity); lower values give fewer exemplars, higher values more.
    With 1 - q = m / N, each point has on average m other points more similar than
    the preference, which roughly favours clusters of ~m points.
    Needs several N x N float64 matrices, so memory grows quadratically with N.
    """
    def __init__(self, preference_quantile=0.5, damping=0.5, max_iter=200, random_state=42):
        self.preference_quantile = preference_quantile
        self.damping = damping
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_predict(self, X, y=None):
        S = -euclidean_distances(X, squared=True)
        # Quantile over pairs of distinct points only: the diagonal (self-similarity,
        # 0) is the top 1/N of S, so including it makes any q > 1 - 1/N give
        # preference 0, i.e. every point its own cluster.
        off_diag = S[~np.eye(S.shape[0], dtype=bool)]
        self.preference_ = np.quantile(off_diag, self.preference_quantile, overwrite_input=True)
        del off_diag
        self.model_ = AffinityPropagation(
            affinity='precomputed', preference=self.preference_,
            damping=self.damping, max_iter=self.max_iter,
            copy=False, random_state=self.random_state
        )
        self.labels_ = self.model_.fit_predict(S)
        return self.labels_

    def fit(self, X, y=None):
        self.fit_predict(X)
        return self


class DPGMMScaledPrior(BaseEstimator, ClusterMixin):
    """
    Dirichlet-process GMM (diagonal covariances) with the covariance prior set to
    covariance_prior_scale times the per-dimension variance of X. sklearn's default
    prior is that variance itself (scale 1), so a given scale means the same thing on
    every dataset. Smaller scales favour tighter, more numerous clusters and larger
    ones fewer. Unlike the weight concentration prior, which had no effect here
    (cars: 137 -> 135 clusters over 1e-3 to 1e6), scales 1e-4 to 10 moved cars from
    203 to 120 clusters and giraffe_zebra from ~200 to 26.
    """
    def __init__(self, covariance_prior_scale=1.0, n_components=5000, random_state=42):
        self.covariance_prior_scale = covariance_prior_scale
        self.n_components = n_components
        self.random_state = random_state

    def fit_predict(self, X, y=None):
        self.model_ = BayesianGaussianMixture(
            n_components=self.n_components, covariance_type='diag',
            weight_concentration_prior_type='dirichlet_process',
            covariance_prior=self.covariance_prior_scale * np.var(X, axis=0, ddof=1),
            random_state=self.random_state
        )
        self.labels_ = self.model_.fit_predict(X)
        return self.labels_

    def fit(self, X, y=None):
        self.fit_predict(X)
        return self


class OPTICSReachabilityCut(BaseEstimator, ClusterMixin):
    """
    OPTICS with DBSCAN-style extraction (sklearn's cluster_method='dbscan'): a flat
    cut of the reachability plot at eps, with eps given as a quantile of the finite
    reachability distances rather than as a raw distance, so one grid works across
    datasets and embeddings. Needs the reachability plot before eps is known, so it
    fits OPTICS first and then cuts with cluster_optics_dbscan.
    """
    def __init__(self, min_samples=5, reach_quantile=0.95, n_jobs=-1):
        self.min_samples = min_samples
        self.reach_quantile = reach_quantile
        self.n_jobs = n_jobs

    def fit_predict(self, X, y=None):
        self.model_ = OPTICS(min_samples=self.min_samples, n_jobs=self.n_jobs).fit(X)
        reach = self.model_.reachability_
        self.eps_ = float(np.quantile(reach[np.isfinite(reach)], self.reach_quantile))
        self.labels_ = cluster_optics_dbscan(
            reachability=reach, core_distances=self.model_.core_distances_,
            ordering=self.model_.ordering_, eps=self.eps_
        )
        return self.labels_

    def fit(self, X, y=None):
        self.fit_predict(X)
        return self


class BirchAdaptive(BaseEstimator, ClusterMixin):
    """
    Birch with no global clustering step (n_clusters=None): every leaf subcluster is a
    cluster, so the number of clusters is set by the threshold (the maximum subcluster
    radius). The threshold is set relative to the data's own distance scale, so one
    grid works across datasets and embeddings: it is log-interpolated, by `position`
    in [0, 1], between two quantiles of the pairwise distances -- q = 2 / N (about each
    point's 2nd-nearest-neighbour distance, i.e. near-singleton clusters) at 0, and
    q = coarse_quantile (a coarse clustering of a few to tens of clusters) at 1.
    Spacing evenly in (log) distance rather than in quantile matters: embeddings with
    tight, well-separated identities have few pairwise distances between the within-
    and between-identity scales, so evenly spaced quantiles jump straight over the
    thresholds that separate identities (on ox_flower they skipped from 0.013 to 0.45).
    Quantiles are estimated from the distances of up to n_query random points to all points.
    """
    def __init__(self, position=0.5, coarse_quantile=0.02, branching_factor=50, n_query=1000, random_state=42):
        self.position = position
        self.coarse_quantile = coarse_quantile
        self.branching_factor = branching_factor
        self.n_query = n_query
        self.random_state = random_state

    def fit_predict(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        query = rng.choice(X.shape[0], size=min(self.n_query, X.shape[0]), replace=False)
        D = euclidean_distances(X[query], X)
        # Drop each query point's distance to itself (0)
        not_self = np.ones(D.shape, dtype=bool)
        not_self[np.arange(len(query)), query] = False
        fine_quantile = min(2 / X.shape[0], self.coarse_quantile)
        # Floored above 0: Birch needs threshold > 0, and log-interpolation needs both
        # ends > 0, which a quantile of exact-duplicate distances isn't
        lo, hi = np.maximum(np.quantile(D[not_self], [fine_quantile, self.coarse_quantile]), 1e-12)
        self.threshold_ = float(np.exp(np.log(lo) + self.position * (np.log(hi) - np.log(lo))))
        self.model_ = Birch(threshold=self.threshold_, branching_factor=self.branching_factor, n_clusters=None)
        self.labels_ = self.model_.fit_predict(X)
        return self.labels_

    def fit(self, X, y=None):
        self.fit_predict(X)
        return self


def get_clustering_algorithms(N_TRUE_CLUSTERS, db_only=False, skip_algo=None, grid_search=False, n_points=None, only_algo=None):
    # SEARCH_STEPS = (100, 10, 1) # Coarse to fine search steps for unknown-k algorithms
    ESTIMATED_MAX_CLUSTERS = 2000
    clustering_algorithms = {
        # "KMeans": KMeans(n_clusters=N_TRUE_CLUSTERS, n_init='auto', random_state=42, max_iter=20, tol=0.0),
        # "Agglomerative": AgglomerativeClustering(n_clusters=N_TRUE_CLUSTERS),
        # 'Spectral': SpectralClustering(n_clusters=N_TRUE_CLUSTERS, n_components=128, affinity='nearest_neighbors', n_neighbors=10, random_state=42, n_jobs=-1),
        # 'DBSCAN': DBSCAN(eps=0.5, min_samples=10, n_jobs=-1), 
        'HDBSCAN': hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, core_dist_n_jobs=-1),
        'FINCH': FINCHWrapper(req_clust=None, distance='cosine', partition_idx=2), # Mid-level granularity
        # 'SpectralHDBSCAN': SpectralHDBSCAN(n_components=128, n_neighbors=10, min_cluster_size=5, min_samples=5, random_state=42, n_jobs=-1),
        # 'SpectralFINCH': SpectralFINCH(n_components=128, n_neighbors=10, distance='cosine', partition_idx=2, random_state=42, n_jobs=-1),
        # 'KMeans_Unknown': KMeansUnknown(k_min=1, k_max=2000, step=30, random_state=42),
        # 'Agglomerative_Unknown': AgglomerativeUnknown(k_min=1, k_max=2000),
        # 'Spectral_Unknown': SpectralUnknown(k_min=1, k_max=1000, n_components=128, n_neighbors=10, random_state=42),
        # 'KMeans_Search': ClusterSearchWrapper(
        #     estimator_class=KMeans,
        #     estimator_kwargs={'n_init': 'auto', 'random_state': 42, 'max_iter': 20, 'tol': 0.0},
        #     k_min=1, k_max=ESTIMATED_MAX_CLUSTERS,
        #     budget=100
        # ),
        # 'Agglomerative_Search': ClusterSearchWrapper(
        #     estimator_class=AgglomerativeClustering,
        #     estimator_kwargs={},
        #     k_min=1, k_max=ESTIMATED_MAX_CLUSTERS,
        #     budget=100
        # ),
        # 'Spectral_Search': ClusterSearchWrapper(
        #     estimator_class=SpectralClustering,
        #     estimator_kwargs={'affinity': 'nearest_neighbors', 'n_neighbors': 10, 'random_state': 42, 'n_jobs': -1},
        #     k_min=1, k_max=ESTIMATED_MAX_CLUSTERS,
        #     budget=100
        # ),
    }

    # if grid_search:
    #     clustering_algorithms = {}
    #     # 4 x 5 = 20 combos, matching the size of the K_GRID below so HDBSCAN,
    #     # KMeans, Agglomerative, and Spectral each get about the same number of trials.
    #     HDBSCAN_MIN_CLUSTER_SIZES = [2, 5, 10, 20, 50]
    #     HDBSCAN_MIN_SAMPLES = [1, 2, 3, 5]
    #     for mcs in HDBSCAN_MIN_CLUSTER_SIZES:
    #         for ms in HDBSCAN_MIN_SAMPLES:
    #             clustering_algorithms[f'HDBSCAN_mcs{mcs}_ms{ms}'] = hdbscan.HDBSCAN(
    #                 min_cluster_size=mcs, min_samples=ms, core_dist_n_jobs=-1
    #             )

    #     # Shared k grid for the algorithms that take an explicit n_clusters.
    #     # Exponential (geometric) spacing so the grid is dense at small k and
    #     # sparse toward the ceiling, which matches how quickly the clustering
    #     # metrics change as k grows.
    #     N_K = 15
    #     K_GRID = sorted(set(
    #         int(round(k)) for k in np.geomspace(5, 4000, num=N_K)
    #     ))
    #     if n_points is not None and n_points <= 4000:
    #         # Too few points for the fixed grid above (its k values would exceed
    #         # n_points); replace it with the same number of values, geometrically
    #         # spaced up to n_points - 1 instead of up to 2000. Capped at
    #         # n_points - 1 (not n_points) because silhouette_score requires
    #         # n_labels < n_samples (k == n_points puts every point in its own
    #         # cluster, which is invalid).
    #         k_max = max(2, n_points - 1)
    #         K_GRID = sorted(set(
    #             int(round(k)) for k in np.geomspace(2, k_max, num=N_K)
    #         ))
    #     for k in K_GRID:
    #         clustering_algorithms[f'KMeans_k{k}'] = KMeans(
    #             n_clusters=k, n_init='auto', random_state=42, max_iter=20, tol=0.0
    #         )
    #         clustering_algorithms[f'Agglomerative_k{k}'] = AgglomerativeClustering(n_clusters=k)


    #     # FINCH's own hierarchy stands in for a hyperparameter grid: sweep the
    #     # partition level instead of a model hyperparameter (idx is clamped to
    #     # the deepest level FINCH actually finds, see FINCHWrapper.fit_predict).
    #     # Kept small since FINCH typically only surfaces a handful of levels,
    #     # so more values here would just duplicate the same clustering.
    #     FINCH_PARTITION_IDXS = [0, 1, 2, 3, 4]
    #     for idx in FINCH_PARTITION_IDXS:
    #         clustering_algorithms[f'FINCH_p{idx}'] = FINCHWrapper(
    #             req_clust=None, distance='cosine', partition_idx=idx
    #         )


    #     # Eigendecomposition cost grows quickly with k, so k is capped at a fixed
    #     # ceiling (further lowered to n_points - 1 on small datasets) rather than
    #     # the (unknown, in practice) true cluster count. n_components is left at
    #     # sklearn's default (== n_clusters): spectral theory wants ~k eigenvectors
    #     # of the N x N Laplacian to separate k clusters (this is bounded by
    #     # n_points, not by the input feature dim), so the grid is kept modest
    #     # (512) to keep that k-vector eigensolve affordable.
    #     SPECTRAL_MAX_K = 512

    #     # Highest k to try: the fixed ceiling, but never >= n_points, since
    #     # SpectralClustering's internal KMeans needs n_clusters < n_samples
    #     # (same constraint as the KMeans/Agglomerative grid above).
    #     spectral_k_max = SPECTRAL_MAX_K
    #     if n_points is not None:
    #         spectral_k_max = min(spectral_k_max, n_points - 1)
    #     spectral_k_max = max(spectral_k_max, 2)

    #     # Same geometric spacing as the main K_GRID, from k=5 up to that ceiling.
    #     # (Start below 5 only when the ceiling itself is that small.)
    #     spectral_k_min = min(5, spectral_k_max)
    #     K_GRID = sorted(set(
    #         int(round(k))
    #         for k in np.geomspace(spectral_k_min, spectral_k_max, num=N_K)
    #     ))
    #     # affinity='nearest_neighbors' needs n_neighbors <= n_samples (sklearn raises
    #     # ValueError otherwise), so this fixed 10 needs the same n_points-aware cap as
    #     # n_clusters below -- otherwise this loop crashes on any dataset with fewer
    #     # than 10 points regardless of how small k has been rescaled to.
    #     spectral_k_n_neighbors = min(10, n_points - 1) if n_points is not None else 10
    #     for k in K_GRID:
    #         clustering_algorithms[f'Spectral_k{k}'] = SpectralClustering(
    #             n_clusters=k, affinity='nearest_neighbors',
    #             n_neighbors=spectral_k_n_neighbors, random_state=42, n_jobs=-1
    #         )

    #     # # Separate one-off sweep over the kNN graph density at a fixed, reasonable k,
    #     # # rather than crossing it with the full K_GRID above (which would multiply
    #     # # the number of expensive eigendecompositions).
    #     # SPECTRAL_N_NEIGHBORS_GRID = [10, 15, 20, 30]
    #     # spectral_sweep_k = 200
    #     # spectral_sweep_k = min(spectral_sweep_k, n_points - 1) if n_points is not None else spectral_sweep_k
    #     # for nn in SPECTRAL_N_NEIGHBORS_GRID:
    #     #     # Same n_neighbors <= n_samples constraint as above; floored at 1 so this
    #     #     # never asks for 0 neighbors on a near-degenerate (n_points<=1) problem.
    #     #     nn_capped = max(1, min(nn, n_points - 1)) if n_points is not None else nn
    #     #     clustering_algorithms[f'Spectral_nn{nn}'] = SpectralClustering(
    #     #         n_clusters=spectral_sweep_k, n_components=min(128, spectral_sweep_k),
    #     #         affinity='nearest_neighbors', n_neighbors=nn_capped, random_state=42, n_jobs=-1
    #     #     )

    if grid_search:
        clustering_algorithms = {}

        # Shared k grid for the algorithms parameterised by # clusters (KMeans,
        # Agglomerative, Spectral, GMM, Mean-shift): 10 log-spaced values from 5
        # to 5000. Capped at n_points - 1 on small datasets, since silhouette_score
        # requires n_labels < n_samples (and KMeans/GMM need n_clusters <= n_samples).
        N_K = 10
        k_min, k_max = 5, 5000
        if n_points is not None:
            k_max = min(k_max, max(2, n_points - 1))
            k_min = min(k_min, k_max)
        K_GRID = sorted(set(
            int(round(k)) for k in np.geomspace(k_min, k_max, num=N_K)
        ))

        # affinity='nearest_neighbors' needs n_neighbors <= n_samples (sklearn raises
        # ValueError otherwise), so this fixed 10 needs the same n_points-aware cap.
        spectral_n_neighbors = min(10, n_points - 1) if n_points is not None else 10

        for k in K_GRID:
            clustering_algorithms[f'KMeans_k{k}'] = KMeans(
                n_clusters=k, n_init='auto', random_state=42, max_iter=20, tol=0.0
            )
            clustering_algorithms[f'Agglomerative_k{k}'] = AgglomerativeClustering(n_clusters=k)
            # n_components (eigenvectors) capped at 128 rather than sklearn's default
            # of n_clusters: at k in the thousands the k-vector eigensolve dominated
            # (hours per fit), and on ox_flower / giraffe_zebra 128 eigenvectors were
            # 20-240x faster with equal or better NMI/ARI at large k (within ~0.015
            # NMI of all k eigenvectors). n_init=1 (sklearn's default is 10) for the
            # k-means on the embedding, matching the KMeans grid; 10 restarts gave
            # identical scores at twice the time.
            clustering_algorithms[f'Spectral_k{k}'] = SpectralClustering(
                n_clusters=k, n_components=min(k, 128), n_init=1,
                affinity='nearest_neighbors', n_neighbors=spectral_n_neighbors,
                random_state=42, n_jobs=-1
            )
            # Diagonal covariances: full ones need k x D x D parameters, which is
            # infeasible for high-dim embeddings at k in the thousands.
            clustering_algorithms[f'GMM_k{k}'] = GaussianMixture(
                n_components=k, covariance_type='diag', random_state=42
            )
            clustering_algorithms[f'MeanShift_k{k}'] = MeanShiftK(n_clusters=k)

        # FINCH partition level: 4 linearly spaced values over partitions 1-5
        # (1-indexed), i.e. partitions [1, 2, 4, 5] -> partition_idx [0, 1, 3, 4].
        # idx is clamped to the deepest level FINCH actually finds, see
        # FINCHWrapper.fit_predict.
        FINCH_PARTITIONS = sorted(set(
            int(round(p)) for p in np.linspace(1, 5, num=4)
        ))
        for p in FINCH_PARTITIONS:
            clustering_algorithms[f'FINCH_p{p - 1}'] = FINCHWrapper(
                req_clust=None, distance='cosine', partition_idx=p - 1
            )

        # 4 x 4 = 16 combos over min_cluster_size x min_samples.
        HDBSCAN_MIN_CLUSTER_SIZES = [2, 5, 10, 20]
        HDBSCAN_MIN_SAMPLES = [1, 3, 5, 10]
        for mcs in HDBSCAN_MIN_CLUSTER_SIZES:
            for ms in HDBSCAN_MIN_SAMPLES:
                clustering_algorithms[f'HDBSCAN_mcs{mcs}_ms{ms}'] = hdbscan.HDBSCAN(
                    min_cluster_size=mcs, min_samples=ms, core_dist_n_jobs=-1
                )

        # OPTICS with DBSCAN-style extraction: eps at 10 quantiles of each dataset's
        # reachability distances, 1 - q log-spaced from 0.2 to 0.005 (q = 0.800 ...
        # 0.995), see OPTICSReachabilityCut. sklearn's other (xi) extraction labels only
        # the smallest (leaf) clusters, so identities broke into tiny clusters plus
        # 40-65% noise (cars: ARI <= 0.04 at min_samples=5, 0.57 at best with
        # min_samples=50); the eps cut reached ARI 0.70 on cars (at q ~0.98; 0.40 at
        # 0.90, 0.30 at 0.995 where eps jumps to the between-identity scale). Neither
        # works on datasets with 2-3 images per identity (giraffe_zebra: ARI ~0).
        for q in 1 - np.geomspace(0.2, 0.005, num=10):
            clustering_algorithms[f'OPTICS_eps_q{q:.3f}'] = OPTICSReachabilityCut(
                min_samples=5, reach_quantile=float(q), n_jobs=-1
            )

        # Affinity propagation: commented out for now (~6-10 h per sweep and ~117 GiB at
        # N=54k on 1 CPU); Birch below replaces it.
        # # Affinity propagation preference, read as a quantile q of the pairwise
        # # similarities (see AffinityPropagationQuantile), since a raw preference >= 0
        # # is at or above every similarity and would make every point its own exemplar.
        # # Only pairs near the top of the similarity distribution (q close to 1) change
        # # the result much, so 1 - q = m / N is log-spaced in m, the average number of
        # # neighbours more similar than the preference: from 2 (clusters of a few
        # # points, e.g. happy_whale's median of 3 images per individual) to N / 2 (the
        # # median similarity, ~sklearn's default). The values of q therefore depend on
        # # N, so the names use the grid position (p0 = most clusters, p9 = fewest) to
        # # line up across datasets. damping=0.9 rather than sklearn's 0.5, which
        # # oscillates and hits max_iter without converging even on well-separated blobs.
        # if n_points is None:
        #     raise ValueError("grid_search needs n_points to build the affinity propagation grid")
        # ap_one_minus_q = np.geomspace(min(2 / n_points, 0.5), 0.5, num=10)
        # for i, one_minus_q in enumerate(ap_one_minus_q):
        #     clustering_algorithms[f'AffinityPropagation_p{i}'] = AffinityPropagationQuantile(
        #         preference_quantile=float(1 - one_minus_q), damping=0.9, random_state=42
        #     )

        # Birch threshold: 10 values log-spaced in distance between each dataset's own
        # fine end (the 2 / N quantile of pairwise distances, ~2nd-nearest-neighbour
        # distance) and coarse end (the 2% quantile; at 5% the coarsest setting was a
        # single cluster on ox_flower), see BirchAdaptive. The thresholds
        # therefore depend on the data, so the names use the grid position
        # (p0 = finest / most clusters, p9 = coarsest) to line up across datasets.
        for i, position in enumerate(np.linspace(0, 1, num=10)):
            clustering_algorithms[f'Birch_p{i}'] = BirchAdaptive(position=float(position), random_state=42)

        # Dirichlet-process GMM: covariance prior scale (see DPGMMScaledPrior), 10
        # log-spaced values from 1e-3 to 100. Below 1e-3 results stopped changing
        # (cars: 203 vs 202 clusters at 1e-4 / 1e-3); the weight concentration prior
        # is left at sklearn's default since it had no effect. n_components is only a
        # truncation level (unused components get ~0 weight), so it is set to the top
        # of the k grid. Diagonal covariances for the same reason as GMM above.
        for scale in np.geomspace(1e-3, 1e2, num=10):
            clustering_algorithms[f'DPGMM_cp{scale:.3g}'] = DPGMMScaledPrior(
                covariance_prior_scale=float(scale), n_components=K_GRID[-1], random_state=42
            )

    if db_only:
        clustering_algorithms = {
            name: model for name, model in clustering_algorithms.items()
            if name == 'DBSCAN' or name.startswith('HDBSCAN')
        }

    # skip_algo / only_algo: comma-separated, case-insensitive name prefixes, e.g.
    # "AffinityPropagation" matches every AffinityPropagation_p* grid entry.
    if skip_algo:
        skip_prefixes = _parse_prefixes(skip_algo)
        clustering_algorithms = {
            name: model for name, model in clustering_algorithms.items()
            if not name.lower().startswith(skip_prefixes)
        }

    if only_algo:
        only_prefixes = _parse_prefixes(only_algo)
        clustering_algorithms = {
            name: model for name, model in clustering_algorithms.items()
            if name.lower().startswith(only_prefixes)
        }

    return clustering_algorithms


def _parse_prefixes(csv):
    # Empty entries (e.g. from a trailing comma) are dropped: "" is a prefix of
    # every name, so it would match everything.
    return tuple(p for p in (alg.strip().lower() for alg in csv.split(",")) if p)