"""Supervised clustering metrics, the per-query record saved by active_selection.py, and the
evaluation metrics of Sec 4.2 computed from those records by plot_saved_results.py."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure,
)

# Supervised metrics of a clustering, all higher-is-better
SUPERVISED_METRICS = ["id_f1", "id_f1_macro", "nmi", "ari", "homogeneity", "completeness", "v_measure"]
# Also recorded for every candidate: K^j and the cluster discovery error EK (Eq. 13, lower is better)
METRIC_LABELS = {
    "id_f1": "ID-F1", "id_f1_macro": "ID-F1 (macro)", "nmi": "NMI", "ari": "ARI",
    "homogeneity": "Homogeneity", "completeness": "Completeness", "v_measure": "V-measure",
    "ek": "EK", "n_clusters": "# clusters",
}


# ---------------------------------------------------------------------------
# Metrics of one clustering
# ---------------------------------------------------------------------------

def _contingency(a, b):
    a_ids, a_inv = np.unique(a, return_inverse=True)
    b_ids, b_inv = np.unique(b, return_inverse=True)
    flat = np.bincount(a_inv * len(b_ids) + b_inv, minlength=len(a_ids) * len(b_ids))
    return flat.reshape(len(a_ids), len(b_ids))


def id_f1(gt, pred):
    """ID-F1 (Sec 4.2), same definition as cluster_benchmark.compute_id_f1: one-to-one Hungarian
    matching of predicted to true clusters, F1 of each true cluster with its match (0 if unmatched),
    averaged over true clusters (macro) or weighted by true cluster size (weighted)."""
    C = _contingency(gt, pred)
    rows, cols = linear_sum_assignment(-C)
    gt_sizes, pred_sizes = C.sum(1), C.sum(0)
    tp = C[rows, cols]
    precision, recall = tp / pred_sizes[cols], tp / gt_sizes[rows]
    f1 = np.zeros(len(gt_sizes))
    with np.errstate(invalid="ignore"):
        f1[rows] = np.nan_to_num(2 * precision * recall / (precision + recall))
    return float(f1.mean()), float(np.average(f1, weights=gt_sizes))


def supervised_metrics(gt, pred):
    macro, weighted = id_f1(gt, pred)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(gt, pred)
    return {
        "id_f1": weighted, "id_f1_macro": macro,
        "nmi": float(normalized_mutual_info_score(gt, pred)),
        "ari": float(adjusted_rand_score(gt, pred)),
        "homogeneity": float(homogeneity), "completeness": float(completeness), "v_measure": float(v_measure),
    }


def pairwise_f1_between(a, b):
    """F1 between the co-occurrence matrices of two clusterings (Eq. 3-4 with b as the reference),
    exact from their contingency table: TP = sum_ab C(n_ab, 2), |A| = sum_a C(n_a, 2), ..."""
    C = _contingency(a, b).astype(np.float64)
    pairs = lambda n: n * (n - 1) / 2
    tp, n_a, n_b = pairs(C).sum(), pairs(C.sum(1)).sum(), pairs(C.sum(0)).sum()
    return 2 * tp / (n_a + n_b) if n_a + n_b > 0 else 1.0


def agreement(consensus, labels, metric):
    """How well one clustering agrees with a consensus clustering (used by the consensus priors)."""
    if metric == "nmi":
        return float(normalized_mutual_info_score(consensus, labels))
    if metric == "ari":
        return float(adjusted_rand_score(consensus, labels))
    if metric == "id_f1":
        return id_f1(consensus, labels)[1]
    if metric == "pairwise_f1":
        return pairwise_f1_between(labels, consensus)
    raise ValueError(f"Unknown agreement metric {metric!r}")


# ---------------------------------------------------------------------------
# What active_selection.py records after every query
# ---------------------------------------------------------------------------

def rank_by_belief(log_w, rng):
    """Candidates by decreasing belief, ties broken at random (e.g. every candidate under a uniform
    prior), so a tie never silently favors whichever candidate happens to be listed first."""
    return np.lexsort((rng.random(len(log_w)), -log_w))


def _srocc(x_ranks, y):
    """Spearman correlation given x's ranks; None when either side is constant."""
    y_ranks = rankdata(y)
    if np.ptp(x_ranks) == 0 or np.ptp(y_ranks) == 0:
        return None
    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def belief_record(cands, log_w, rng, top_k):
    """Summary of the belief w^(t) (log_w normalized so that logsumexp(log_w) = 0):
      selected       j_hat = argmax_j w_j, the candidate chosen if we stopped now
      metrics        every metric of j_hat
      top_k_mean     mean of every metric over the top_k candidates by w
      srocc          Spearman correlation between w and each supervised metric across candidates"""
    order = rank_by_belief(log_w, rng)
    j_hat, top = order[0], order[:top_k]
    w_ranks = rankdata(log_w)
    return {
        "selected": cands.keys[j_hat],
        "max_weight": float(np.exp(log_w[j_hat])),
        "metrics": {m: float(v[j_hat]) for m, v in cands.metrics.items()},
        "top_k_mean": {m: float(v[top].mean()) for m, v in cands.metrics.items()},
        "srocc": {m: _srocc(w_ranks, cands.metrics[m]) for m in SUPERVISED_METRICS},
    }


# ---------------------------------------------------------------------------
# Evaluation metrics (Sec 4.2), computed from saved results
# ---------------------------------------------------------------------------

def fractional_regret(value, best):
    """Eq. 12: R = 1 - metric(selected) / max_j metric(candidate j)."""
    return 1.0 - value / best if best > 0 else 0.0


def regret_auc(regrets):
    """Area under the regret-vs-#queries curve, t = 0..T, normalized by T (i.e. the mean regret),
    so it is comparable across budgets and 0 means optimal from the very first query."""
    regrets = np.asarray(regrets, dtype=float)
    if len(regrets) < 2:
        return float(regrets[0])
    return float(np.trapezoid(regrets) / (len(regrets) - 1))


def cluster_discovery_error(k_selected, k_true):
    """Eq. 13: EK = |1 - K^j_hat / K|."""
    return abs(1.0 - k_selected / k_true)
