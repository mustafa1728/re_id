"""Active model selection (Sec 3.3).

Belief w_j^(t) = P(J* = j | L_t) that candidate j is the best, kept in log space. The oracle's label
y_p of a queried pair p is modeled as agreeing with the best candidate's prediction A^j_p with
probability 1 - eps (Eq. 6), which gives the Bayes update (Eq. 7) and the posterior predictive
(Eq. 10) below. Selectors choose which pair to query next:

    entropy            p* = argmax_p H(y_p | L_t) (Eq. 9) -- maximizes the information gain in J*
    uniform            a uniformly random pair of distinct points
    uniform_disagree   a uniformly random pair the candidates disagree on
"""
import numpy as np
from scipy.special import logsumexp

from .pairwise import pair_votes


def posterior_predictive(w, votes, eps):
    """Eq. 10: q = P(y_p = 1 | L_t) = sum_j w_j [(1 - eps) A^j_p + eps (1 - A^j_p)]; votes (..., M)."""
    return eps + (1 - 2 * eps) * (votes @ w)


def binary_entropy(q):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return -q * np.log(q) - (1 - q) * np.log(1 - q)


def random_argmax(values, rng, weights=None, tol=1e-6):
    """Index of the maximum with ties (within tol, to absorb float rounding) broken at random, with
    probability proportional to `weights` if given -- never just the first or last tied index."""
    tied = np.flatnonzero(values >= values.max() - tol)
    p = None if weights is None else weights[tied] / weights[tied].sum()
    return int(rng.choice(tied, p=p))


def bayes_update(log_w, votes, y, eps):
    """Eq. 7: w_j <- w_j (1 - eps)^I(A^j_p = y) eps^I(A^j_p != y), normalized."""
    log_w = log_w + np.where(votes == bool(y), np.log(1 - eps), np.log(eps))
    return log_w - logsumexp(log_w)


class EntropySelector:
    """Max-entropy queries (Eq. 9) over a pool of disagreement pairs. Pairs with the same votes
    (A^1_p, ..., A^M_p) always have the same entropy, so the pool is grouped by vote signature: the
    arg max is taken over signatures, and the chosen signature's pairs are used up in random order
    (a signature stays available until all its pairs have been queried). Signatures tied in entropy
    (e.g. every signature with the same number of votes under a uniform prior) are chosen at random,
    in proportion to their remaining pairs, i.e. uniformly over the tied pairs."""

    def __init__(self, labels_T, pool_i, pool_l, eps, rng):
        self.eps, self.rng = eps, rng
        votes = pair_votes(labels_T, pool_i, pool_l)
        _, first, signature = np.unique(np.packbits(votes, axis=1), axis=0, return_index=True, return_inverse=True)
        signature = signature.reshape(-1)
        self.signatures = votes[first].astype(np.float32)  # (S, M)

        # Pairs grouped by signature, in random order within a group
        shuffled = rng.permutation(len(signature))
        self.pairs = np.stack([pool_i, pool_l], axis=1)[shuffled[np.argsort(signature[shuffled], kind="stable")]]
        self.remaining = np.bincount(signature, minlength=len(first))
        self.group_start = np.cumsum(self.remaining) - self.remaining
        self.stats = {"pool_pairs": len(signature), "pool_signatures": len(first)}

    def select(self, w):
        h = binary_entropy(posterior_predictive(w.astype(np.float32), self.signatures, self.eps))
        h[self.remaining == 0] = -np.inf
        if not np.isfinite(h.max()):
            return None  # pool used up
        s = random_argmax(h, self.rng, weights=self.remaining)
        self.remaining[s] -= 1  # the group is in random order, so this pops a random pair of it
        i, l = self.pairs[self.group_start[s] + self.remaining[s]]
        return int(i), int(l), self.signatures[s] > 0.5, float(h[s] - binary_entropy(self.eps))


class UniformSelector:
    """Uniformly random unqueried pairs of distinct points; with disagreement_only, only pairs on which
    the candidates disagree (0 < sum_j A^j_p < M)."""

    def __init__(self, labels_T, eps, rng, disagreement_only=False, max_tries=1_000_000):
        self.labels_T, self.eps, self.rng = labels_T, eps, rng
        self.disagreement_only, self.max_tries = disagreement_only, max_tries
        self.queried = set()
        self.stats = {}

    def select(self, w):
        N = len(self.labels_T)
        for _ in range(self.max_tries):
            i, l = sorted(self.rng.integers(N, size=2).tolist())
            if i == l or (i, l) in self.queried:
                continue
            votes = self.labels_T[i] == self.labels_T[l]
            if self.disagreement_only and (votes.all() or not votes.any()):
                continue
            self.queried.add((i, l))
            h = binary_entropy(posterior_predictive(w, votes, self.eps))
            return i, l, votes, float(h - binary_entropy(self.eps))
        return None


SELECTORS = ["entropy", "uniform", "uniform_disagree"]


def make_selector(cfg, cands, ctx, rng):
    if cfg.select == "entropy":
        pool_i, pool_l = ctx.disagreement_pairs(cfg.pool_size, cfg.seed)
        return EntropySelector(cands.labels_T, pool_i, pool_l, cfg.epsilon, rng)
    if cfg.select in ("uniform", "uniform_disagree"):
        return UniformSelector(cands.labels_T, cfg.epsilon, rng, disagreement_only=cfg.select == "uniform_disagree")
    raise ValueError(f"Unknown selector {cfg.select!r}")


def run_active_selection(log_w, selector, oracle, eps, budget, on_query):
    """Query up to `budget` pairs, each chosen by `selector` given the current belief and labeled by
    `oracle`, updating the belief after every label (Eq. 7). on_query(t, log_w, query) is called after
    the t-th update. Returns the final log belief; the selected candidate is its arg max."""
    for t in range(1, budget + 1):
        choice = selector.select(np.exp(log_w))
        if choice is None:
            print(f"  no pairs left to query after {t - 1} queries")
            break
        i, l, votes, info_gain = choice
        y = oracle(i, l)
        log_w = bayes_update(log_w, votes, y, eps)
        on_query(t, log_w, {"i": i, "l": l, "y": int(y), "info_gain": info_gain})
    return log_w
