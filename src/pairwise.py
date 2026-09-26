"""Pairwise co-occurrence view of the candidate clusterings (Sec 3.2).

Each clustering j becomes a binary label for every unordered pair of points (i, l), i < l:

    A^j_il = I(z^j_i = z^j_l)                                   (Eq. 1)

A per-pair quantity over all N(N-1)/2 pairs is stored as a flat array indexed by the pair's position in
the strict upper triangle of the N x N pair matrix (see pair_offsets). The central one is the number of
candidates that co-cluster each pair, S_il = sum_j A^j_il, from which the consensus (Eq. 2) and the
pool of pairs worth querying are read off.
"""
import os
import time

import numba
import numpy as np

from .io_utils import cached_npz


# ---------------------------------------------------------------------------
# Flat indexing of unordered pairs
# ---------------------------------------------------------------------------

def pair_offsets(N):
    """offset[i] such that pair (i, l), i < l, has flat index offset[i] + l."""
    i = np.arange(N, dtype=np.int64)
    return i * (2 * N - i - 1) // 2 - i - 1


def unravel_pairs(idx, N):
    """Flat pair indices -> (i, l) arrays."""
    row_start = pair_offsets(N) + np.arange(1, N + 1)  # flat index of (i, i + 1)
    i = np.searchsorted(row_start, idx, side="right") - 1
    return i, idx - pair_offsets(N)[i]


def n_co_clustered_pairs(labels):
    """|A^j| = sum_il A^j_il = sum over clusters c of C(n_c, 2)."""
    sizes = np.unique(labels, return_counts=True)[1].astype(np.int64)
    return int((sizes * (sizes - 1) // 2).sum())


def pair_votes(labels_T, i, l, chunk=1 << 26):
    """(P, M) bool, row p = (A^1_p, ..., A^M_p) for pairs p = (i[p], l[p]); labels_T is (N, M)."""
    M = labels_T.shape[1]
    step = max(1, chunk // M)
    return np.concatenate([labels_T[i[s:s + step]] == labels_T[l[s:s + step]]
                           for s in range(0, len(i), step)]) if len(i) else np.zeros((0, M), bool)


# ---------------------------------------------------------------------------
# Co-clustering counts S_il over all pairs
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _add_co_clustered_pairs(counts, members, bounds, offset):
    """counts[pair] += 1 for every pair inside each cluster. members lists the points cluster by
    cluster (ascending within a cluster), cluster c being members[bounds[c]:bounds[c + 1]]."""
    for c in range(len(bounds) - 1):
        for a in range(bounds[c], bounds[c + 1] - 1):
            base = offset[members[a]]
            for b in range(a + 1, bounds[c + 1]):
                counts[base + members[b]] += 1


def co_cluster_counts(label_matrix):
    """S_il = sum_j A^j_il for all N(N-1)/2 pairs, as a flat uint16 array (N(N-1) bytes: ~2.8GB at
    N = 53K). Cost is the total number of co-clustered pairs over all candidates."""
    M, N = label_matrix.shape
    assert M < np.iinfo(np.uint16).max
    counts = np.zeros(N * (N - 1) // 2, dtype=np.uint16)
    offset = pair_offsets(N)
    for labels in label_matrix:
        members = np.argsort(labels, kind="stable")  # stable: ascending point index within a cluster
        bounds = np.concatenate([[0], np.flatnonzero(np.diff(labels[members])) + 1, [N]])
        _add_co_clustered_pairs(counts, members, bounds, offset)
    return counts


def _chunks(n, size=1 << 27):
    return ((s, min(n, s + size)) for s in range(0, n, size))


def consensus_pairs(counts, M, N, tau):
    """Eq. 2: the pairs with A~_il = I(S_il / M > tau) = 1, as (i, l) arrays."""
    idx = np.concatenate([np.flatnonzero(counts[s:e] > tau * M) + s for s, e in _chunks(len(counts))])
    return unravel_pairs(idx, N)


def pairwise_f1(labels_T, cons_i, cons_l, n_positive):
    """Eq. 3-4, s_j = 2 P_j R_j / (P_j + R_j) = 2 TP_j / (|A^j| + |A~|), where TP_j = sum_il A^j_il A~_il
    is counted over the consensus pairs and n_positive[j] = |A^j| (n_co_clustered_pairs)."""
    tp = np.zeros(labels_T.shape[1], dtype=np.int64)
    step = max(1, (1 << 26) // labels_T.shape[1])
    for s in range(0, len(cons_i), step):
        tp += pair_votes(labels_T, cons_i[s:s + step], cons_l[s:s + step]).sum(axis=0)
    denom = np.asarray(n_positive, dtype=np.float64) + len(cons_i)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 1.0)


def sample_disagreement_pairs(counts, M, N, n, rng):
    """About n pairs on which the candidates disagree (0 < S_il < M), stratified by S_il: level
    c = 1..M-1 contributes min(#pairs at level c, k) pairs, k chosen so ~n are drawn in total.
    Unanimous pairs are never worth querying (their predictive entropy is the minimum, h(eps)), and a
    uniform sample would be swamped by pairs only a few coarse candidates co-cluster; stratifying
    keeps every kind of disagreement represented for the entropy search."""
    level_sizes = np.zeros(M + 1, dtype=np.int64)
    for s, e in _chunks(len(counts)):
        level_sizes += np.bincount(counts[s:e], minlength=M + 1)
    sizes = level_sizes[1:M].astype(float)
    lo, hi = 0.0, max(sizes.max(), 1.0)
    for _ in range(100):  # largest k with sum_c min(size_c, k) <= n
        k = (lo + hi) / 2
        lo, hi = (k, hi) if np.minimum(sizes, k).sum() <= n else (lo, k)
    keep_prob = np.zeros(M + 1, dtype=np.float32)
    keep_prob[1:M] = np.minimum(1.0, lo / np.maximum(sizes, 1))

    idx = np.concatenate([np.flatnonzero(rng.random(e - s, dtype=np.float32) < keep_prob[counts[s:e]]) + s
                          for s, e in _chunks(len(counts))])
    return unravel_pairs(idx, N)


class PairwiseContext:
    """The co-clustering counts of one candidate set, computed lazily (only when some prior or selector
    needs them) and never twice; everything derived from them is cached under cache_dir."""

    def __init__(self, label_matrix, cache_dir):
        self.label_matrix = label_matrix
        self.M, self.N = label_matrix.shape
        self.cache_dir = cache_dir
        self._counts = None

    @property
    def counts(self):
        if self._counts is None:
            start = time.time()
            self._counts = co_cluster_counts(self.label_matrix)
            print(f"  co-clustering counts of {self.N * (self.N - 1) // 2:,} pairs over {self.M} candidates "
                  f"({time.time() - start:.0f}s)")
        return self._counts

    def consensus_pairs(self, tau):
        path = os.path.join(self.cache_dir, f"consensus_pairs_tau{tau:g}.npz")
        pairs = cached_npz(path, lambda: dict(zip("il", consensus_pairs(self.counts, self.M, self.N, tau))))
        return pairs["i"], pairs["l"]

    def disagreement_pairs(self, n, seed):
        path = os.path.join(self.cache_dir, f"disagreement_pairs_n{n}_seed{seed}.npz")
        rng = np.random.default_rng(seed)
        pairs = cached_npz(path, lambda: dict(zip("il", sample_disagreement_pairs(
            self.counts, self.M, self.N, n, rng))))
        return pairs["i"], pairs["l"]
