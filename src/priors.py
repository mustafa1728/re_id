"""Priors over which candidate clustering is best (Sec 3.3).

A prior scores every candidate j with s_j (higher = more likely the best) and the initial belief is

    w_j^(0) = exp(s_j / T) / sum_k exp(s_k / T)                                  (Eq. 5)

Scores:
    uniform              s_j = 0
    silhouette, dbi, chi the candidate's unsupervised index (DBI negated: lower is better)
    pairwise             F1 agreement of A^j with the pairwise majority-vote consensus A~ (Eq. 1-4)
    cooc_closure, cooc_recluster, vote, matrixfac, matrixfac_recluster, bce, bce_grid
                         agreement (--consensus_agreement: nmi / ari / id_f1 / pairwise_f1) of the
                         candidate with that consensus clustering (src/consensus.py)

To add a prior, write a function (cands, ctx, cfg) -> (scores, info) and register it in PRIORS.
"""
import os

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, rankdata

from . import consensus
from .io_utils import cached_json
from .metrics import agreement, supervised_metrics
from .pairwise import n_co_clustered_pairs, pairwise_f1


def uniform_prior(cands, ctx, cfg):
    return np.zeros(cands.M), {}


def silhouette_prior(cands, ctx, cfg):
    return cands.unsup["silhouette"], {}


def dbi_prior(cands, ctx, cfg):
    return -cands.unsup["dbi"], {}


def chi_prior(cands, ctx, cfg):
    return cands.unsup["calinski_harabasz"], {}


def pairwise_prior(cands, ctx, cfg):
    """Sec 3.2: s_j = F1(A^j, A~) with A~_il = I(mean_j A^j_il > tau), summed over all pairs."""
    cons_i, cons_l = ctx.consensus_pairs(cfg.tau)
    n_positive = [n_co_clustered_pairs(row) for row in cands.labels]
    scores = pairwise_f1(cands.labels_T, cons_i, cons_l, n_positive)
    return scores, {"n_consensus_pairs": len(cons_i), "n_pairs": cands.N * (cands.N - 1) // 2}


def build_consensus(method, cands, ctx, cfg):
    """One consensus labeling of the candidates (see src/consensus.py). cooc_closure uses every
    candidate (it is read off the pairwise counts); the others use cfg.consensus_n_raters randomly
    chosen ones if set, as some scale with the number of raters."""
    if method == "cooc_closure":
        return consensus.closure_of_pairs(cands.N, *ctx.consensus_pairs(cfg.tau))

    raters = np.arange(cands.M)
    if cfg.consensus_n_raters and cfg.consensus_n_raters < cands.M:
        raters = np.sort(np.random.default_rng(0).choice(cands.M, cfg.consensus_n_raters, replace=False))
    raw, names = cands.labels[raters], [cands.keys[r] for r in raters]

    if method == "cooc_recluster":
        return consensus.cooccurrence_recluster(consensus.co_association_matrix(raw))
    if method == "vote":
        return consensus.consensus_voting(raw)
    k = consensus.default_single_k(raw, names)
    if method == "matrixfac":
        return consensus.matrix_factorization_spectral(raw, k)
    if method == "matrixfac_recluster":
        return consensus.matrix_factorization_svd(raw, k)
    if method == "bce":
        return consensus.bayesian_cluster_ensemble(raw, k_range=[k])
    if method == "bce_grid":
        return consensus.bayesian_cluster_ensemble(raw, k_range=consensus.bce_k_range(raw))
    raise ValueError(f"Unknown consensus method {method!r}")


def consensus_clustering_prior(method):
    def prior(cands, ctx, cfg):
        labels = build_consensus(method, cands, ctx, cfg)
        scores = np.array([agreement(labels, row, cfg.consensus_agreement) for row in cands.labels])
        info = {"consensus_n_clusters": consensus.n_clusters_of(labels),
                "consensus_metrics": supervised_metrics(cands.gt, labels)}
        return scores, info
    return prior


CONSENSUS_METHODS = ["cooc_closure", "cooc_recluster", "vote", "matrixfac", "matrixfac_recluster", "bce", "bce_grid"]
PRIORS = {
    "uniform": uniform_prior,
    "silhouette": silhouette_prior,
    "dbi": dbi_prior,
    "chi": chi_prior,
    "pairwise": pairwise_prior,
    **{method: consensus_clustering_prior(method) for method in CONSENSUS_METHODS},
}


def prior_cache_name(cfg):
    """Everything a prior's scores depend on (besides the candidates)."""
    if cfg.prior == "pairwise":
        return f"pairwise_tau{cfg.tau:g}"
    if cfg.prior == "cooc_closure":
        return f"cooc_closure_tau{cfg.tau:g}_{cfg.consensus_agreement}"
    if cfg.prior in CONSENSUS_METHODS:
        return f"{cfg.prior}_{cfg.consensus_agreement}_raters{cfg.consensus_n_raters or 'all'}"
    return None  # cheap, not cached


def compute_prior_scores(cfg, cands, ctx, cache_dir):
    """(scores, info) of cfg.prior, cached under cache_dir for the expensive ones."""
    compute = lambda: dict(zip(("scores", "info"), PRIORS[cfg.prior](cands, ctx, cfg)))
    name = prior_cache_name(cfg)
    result = compute() if name is None else cached_json(os.path.join(cache_dir, f"prior_{name}.json"), compute)
    return np.array(result["scores"], dtype=float), result["info"]


def prior_log_weights(scores, T, transform="raw"):
    """log w^(0) = log softmax(s / T) (Eq. 5), after optionally standardizing s across candidates:
        raw     the scores as they are (as in the paper)
        zscore  (s - mean) / std
        rank    normal quantiles of the ranks, so one outlying score (e.g. an unbounded CHI) cannot take
                all the mass
    A missing score (NaN, e.g. silhouette of a 1-cluster candidate) is put 5 std below the worst one."""
    s = np.asarray(scores, dtype=float).copy()
    valid = np.isfinite(s)
    if not valid.any():
        return np.full(len(s), -np.log(len(s)))
    if transform == "zscore":
        s[valid] = (s[valid] - s[valid].mean()) / (s[valid].std() or 1.0)
    elif transform == "rank":
        s[valid] = norm.ppf((rankdata(s[valid]) - 0.5) / valid.sum())
    elif transform != "raw":
        raise ValueError(f"Unknown prior transform {transform!r}")
    s[~valid] = s[valid].min() - 5 * (s[valid].std() or 1.0)
    log_w = s / T
    return log_w - logsumexp(log_w)
