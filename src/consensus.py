"""Consensus clusterings of an ensemble of base clusterings ("raters"), used by the consensus priors
(src/priors.py). Every method takes a raw (R, N) label matrix -- R raters labeling the same N points,
with arbitrary cluster ids per rater and ABSTAIN (-1, e.g. HDBSCAN noise) meaning "no opinion" -- and
returns one (N,) consensus labeling:

    cooc_closure           connected components of the pairwise majority vote A~ (Eq. 2)
    cooc_recluster         HDBSCAN on the co-association matrix, distance 1 - (fraction of raters
                           co-clustering the pair)                            [dense N x N]
    vote                   align every rater's ids to a reference rater (many-to-one), then a
                           per-point majority vote
    matrixfac              spectral clustering of the rater x cluster indicator matrix H
                           (Fern & Brodley, 2004)
    matrixfac_recluster    truncated SVD of H, then k-means
    bce                    Bayesian Cluster Ensembles (Wang, Shan & Banerjee, 2011): variational EM
                           of a latent-class model, K from the HDBSCAN/FINCH raters ("single") or
                           selected by ICL over a range ("grid")

matrixfac, matrixfac_recluster and bce need a number of consensus clusters; by default it is the median
cluster count of the HDBSCAN/FINCH raters (default_single_k), which find K without being told one.
"""
import numpy as np
import hdbscan
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import digamma, gammaln
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import TruncatedSVD

ABSTAIN = -1
MAX_DENSE_POINTS = 30_000  # cooc_recluster holds N x N matrices


def n_clusters_of(row):
    return len(np.unique(row[row != ABSTAIN]))


def default_single_k(raw_matrix, rater_names, name_filter=("hdbscan", "finch")):
    """Median cluster count of the raters whose name contains one of name_filter (all raters if none
    do). A hyperparameter sweep (KMeans_k5 ... KMeans_k5000) is told its k, so its counts say nothing
    about the data; HDBSCAN and FINCH find theirs."""
    matches = [r for r, name in enumerate(rater_names) if any(f in str(name).lower() for f in name_filter)]
    ks = [n_clusters_of(raw_matrix[r]) for r in (matches or range(len(rater_names)))]
    ks = [k for k in ks if k > 0]
    return int(np.median(ks)) if ks else 2


# ---------------------------------------------------------------------------
# Co-occurrence
# ---------------------------------------------------------------------------

def closure_of_pairs(N, i, l):
    """Consensus clusters = connected components of the graph whose edges are the given pairs, e.g. the
    consensus-positive pairs of Eq. 2 (the transitive closure of A~)."""
    graph = csr_matrix((np.ones(len(i), dtype=np.int8), (i, l)), shape=(N, N))
    return connected_components(graph, directed=False)[1]


def co_association_matrix(raw_matrix):
    """C[i, l] = fraction of raters that put i and l in the same (non-abstain) cluster; dense N x N."""
    R, N = raw_matrix.shape
    if N > MAX_DENSE_POINTS:
        raise ValueError(f"co-association matrix of {N} points needs {4 * N * N / 1e9:.0f}GB per copy; "
                         f"only supported up to {MAX_DENSE_POINTS} points")
    C = np.zeros((N, N), dtype=np.float32)
    for row in raw_matrix:
        C += (row[:, None] == row[None, :]) & (row[:, None] != ABSTAIN)
    return C / R


def cooccurrence_recluster(C, min_cluster_size=None, min_samples=None):
    """HDBSCAN on distances 1 - C, so no co-association threshold (and no chaining through barely-over-
    threshold pairs) is needed. Defaults scale with N."""
    N = C.shape[0]
    min_cluster_size = min_cluster_size or max(5, N // 100)
    min_samples = min_samples or max(2, min_cluster_size // 5)
    D = 100 * (1 - C.astype(np.float64))
    np.fill_diagonal(D, 0)
    return hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=min_cluster_size,
                           min_samples=min_samples).fit_predict(D)


# ---------------------------------------------------------------------------
# ID matching + majority vote
# ---------------------------------------------------------------------------

def match_to_reference(ref_row, other_row):
    """Relabel other_row with ref_row's ids: each of its clusters maps to the reference cluster it
    overlaps most. Many-to-one on purpose -- a rater that splits a true cluster in pieces should have
    every piece map to it. A cluster with no overlap becomes ABSTAIN."""
    ref_ids = np.unique(ref_row[ref_row != ABSTAIN])
    other_ids = np.unique(other_row[other_row != ABSTAIN])
    out = np.full(other_row.shape, ABSTAIN, dtype=ref_row.dtype)
    if len(other_ids) == 0 or len(ref_ids) == 0:
        return out

    both = (other_row != ABSTAIN) & (ref_row != ABSTAIN)
    overlap = np.bincount(np.searchsorted(other_ids, other_row[both]) * len(ref_ids)
                          + np.searchsorted(ref_ids, ref_row[both]),
                          minlength=len(other_ids) * len(ref_ids)).reshape(len(other_ids), len(ref_ids))
    best = overlap.argmax(axis=1)
    mapped = np.where(overlap[np.arange(len(other_ids)), best] > 0, ref_ids[best], ABSTAIN)

    has_label = other_row != ABSTAIN
    out[has_label] = mapped[np.searchsorted(other_ids, other_row[has_label])]
    return out


def consensus_voting(raw_matrix, rater_names=None, ref_rater=None):
    """Align every rater to a reference rater (default: the one whose cluster count is closest to the
    ensemble median), then give each point the reference id most raters voted for."""
    if ref_rater is None:
        k = np.array([n_clusters_of(row) for row in raw_matrix])
        valid = np.flatnonzero(k > 0)
        ref_rater = int(valid[np.argmin(np.abs(k[valid] - np.median(k[valid])))])
    ref_row = raw_matrix[ref_rater]
    aligned = np.stack([match_to_reference(ref_row, row) for row in raw_matrix])  # (R, N)

    ref_ids = np.unique(ref_row[ref_row != ABSTAIN])
    N = raw_matrix.shape[1]
    points = np.broadcast_to(np.arange(N), aligned.shape)
    voted = aligned != ABSTAIN
    votes = np.bincount(points[voted] * len(ref_ids) + np.searchsorted(ref_ids, aligned[voted]),
                        minlength=N * len(ref_ids)).reshape(N, len(ref_ids))
    return np.where(votes.sum(axis=1) > 0, ref_ids[votes.argmax(axis=1)], ABSTAIN)


# ---------------------------------------------------------------------------
# Matrix factorization of the indicator matrix H = [H_1 | ... | H_R]
# ---------------------------------------------------------------------------

def indicator_matrix(raw_matrix):
    """Sparse (N, sum_r K_r) point x cluster incidence matrix of every rater, H_r[i, c] = 1 iff rater r
    puts i in its cluster c (abstained points get an all-zero row in that rater's block)."""
    R, N = raw_matrix.shape
    rows, cols, offset = [], [], 0
    for row in raw_matrix:
        has_label = np.flatnonzero(row != ABSTAIN)
        ids, codes = np.unique(row[has_label], return_inverse=True)
        rows.append(has_label)
        cols.append(codes + offset)
        offset += len(ids)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, offset))


def matrix_factorization_spectral(raw_matrix, k, seed=0):
    """Spectral clustering of H itself (nearest-neighbor affinity) into k clusters."""
    H = indicator_matrix(raw_matrix)
    return SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=min(10, H.shape[0] - 1),
                              random_state=seed, n_jobs=-1).fit_predict(H)


def matrix_factorization_svd(raw_matrix, k, n_components=None, seed=0):
    """Truncated SVD of H (embedding @ embedding.T ~ H H^T, a low-rank co-association matrix), then
    k-means on the embedding."""
    H = indicator_matrix(raw_matrix)
    n_components = n_components or max(2, min(10, H.shape[1] - 1))
    embedding = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(H)
    return KMeans(n_clusters=k, n_init="auto", random_state=seed).fit_predict(embedding)


# ---------------------------------------------------------------------------
# Bayesian Cluster Ensembles (Wang, Shan & Banerjee, 2011)
#
#   theta ~ Dir(alpha_0),  Phi_k^(r) ~ Dir(beta_0)     for consensus cluster k, rater r
#   z_i ~ Discrete(theta),  x_i^(r) ~ Discrete(Phi_{z_i}^(r))
#
# x_i^(r) is rater r's raw label of point i (ABSTAIN is just one more symbol), so no rater is ever
# aligned to another: EM learns which of its clusters each consensus cluster emits. Mean-field
# variational EM, q(theta) q(Phi) q(z) with Dirichlet / Dirichlet / Discrete factors.
# Cost per iteration is O(N * K * R) time and O(K * sum_r K_r) memory.
# ---------------------------------------------------------------------------

def _dirichlet_kl(a, a0):
    """KL(Dir(a) || Dir(a0)) over the last axis of a."""
    a0 = np.broadcast_to(a0, a.shape).astype(float)
    return (gammaln(a.sum(-1)) - gammaln(a).sum(-1) - gammaln(a0.sum(-1)) + gammaln(a0).sum(-1)
            + np.sum((a - a0) * (digamma(a) - digamma(a.sum(-1, keepdims=True))), axis=-1))


def _bce_fit(codes, n_symbols, K, rng, n_iter=50, prior=0.5):
    """One variational EM run at K consensus clusters. codes[r] are rater r's labels as 0..K_r-1.
    Returns tau (N, K) = q(z) and the ELBO (a lower bound on log p(X | K))."""
    R, N = codes.shape
    one_hots = [csr_matrix((np.ones(N), (codes[r], np.arange(N))), shape=(n_symbols[r], N)) for r in range(R)]
    tau = rng.dirichlet(np.full(K, 2.0), size=N)
    for _ in range(n_iter):
        # M-step: posterior Dirichlet parameters = prior + expected counts under q(z)
        alpha = prior + tau.sum(axis=0)
        betas = [prior + (one_hots[r] @ tau).T for r in range(R)]  # (K, K_r) each
        # E-step: log tau_ik = E[log theta_k] + sum_r E[log Phi_k^(r)[x_i^(r)]]
        log_tau = np.tile(digamma(alpha) - digamma(alpha.sum()), (N, 1))
        for r in range(R):
            e_log_phi = digamma(betas[r]) - digamma(betas[r].sum(axis=1, keepdims=True))
            log_tau += e_log_phi[:, codes[r]].T
        log_norm = np.logaddexp.reduce(log_tau, axis=1)
        tau = np.exp(log_tau - log_norm[:, None])
    kl = _dirichlet_kl(alpha, prior) + sum(_dirichlet_kl(b, prior).sum() for b in betas)
    return tau, float(log_norm.sum() - kl)


def bayesian_cluster_ensemble(raw_matrix, k_range, n_restarts=1, n_iter=50, prior=0.5, seed=0):
    """Fit at every K in k_range (best ELBO of n_restarts) and return the argmax labeling at the K with
    the lowest ICL = BIC + 2 * entropy(q(z)) (Biernacki et al., 2000), with the ELBO standing in for
    the log-likelihood."""
    codes, n_symbols = [], []
    for row in raw_matrix:
        ids, code = np.unique(row, return_inverse=True)
        codes.append(code)
        n_symbols.append(len(ids))
    codes = np.stack(codes)
    N = codes.shape[1]
    rng = np.random.default_rng(seed)

    best_icl, best_labels = np.inf, None
    for K in k_range:
        tau, elbo = max((_bce_fit(codes, n_symbols, K, rng, n_iter, prior) for _ in range(n_restarts)),
                        key=lambda fit: fit[1])
        n_params = (K - 1) + sum(K * (n - 1) for n in n_symbols)
        entropy = -np.sum(tau * np.log(np.clip(tau, 1e-300, 1)))
        icl = -2 * elbo + n_params * np.log(N) + 2 * entropy
        if icl < best_icl:
            best_icl, best_labels = icl, tau.argmax(axis=1)
    return best_labels


def bce_k_range(raw_matrix, margin=2, max_values=20):
    """K from the smallest to the largest rater cluster count (+ margin), at most max_values values."""
    ks = [n_clusters_of(row) for row in raw_matrix]
    k_min, k_max = max(2, min(ks) - margin), max(ks) + margin
    return range(k_min, k_max + 1, (k_max - k_min) // max_values + 1)
