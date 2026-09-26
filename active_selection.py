"""Active model selection for clustering: run experiments.

For every dataset and every combination of --prior / --select / --epsilon / --seed, starts from the
prior belief w^(0) over the candidate clusterings, queries --budget pairs and saves, under
<results_root>/<experiment>/<dataset>/:

    config.json      the experiment's arguments, pool statistics and run time
    candidates.json  every candidate: key, feature, algorithm, supervised and unsupervised metrics
    prior.json       prior scores s_j and w^(0)
    results.json     one record per query t = 0..B (t = 0 is the prior alone): the queried pair, the
                     oracle's and the true label, the selected candidate and its metrics, the mean
                     metrics of the top-k candidates, and the SROCC between w^(t) and each metric
    weights.json     w^(t) for t = 0..B

Examples:
    python active_selection.py --datasets cars --prior pairwise,uniform,silhouette,dbi,chi \\
        --select entropy,uniform --epsilon 0.1,0.4 --budget 1000
    python active_selection.py --datasets all --prior pairwise --oracle qwen --budget 100
"""
import os
import time
import warnings
import traceback
import concurrent.futures

import numpy as np

from src.args import parse_args, experiment_configs, experiment_name
from src.active import make_selector, run_active_selection
from src.candidates import load_candidates, list_datasets
from src.io_utils import write_json
from src.metrics import SUPERVISED_METRICS, belief_record
from src.oracles import make_oracle
from src.pairwise import PairwiseContext
from src.priors import compute_prior_scores, prior_log_weights

# sklearn's clustering metrics warn whenever a clustering has more clusters than half the points
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")


def run_experiment(cfg, cands, ctx, out_dir):
    start = time.time()
    cache_dir = os.path.join(cfg.cache_root, cands.dataset)

    scores, prior_info = compute_prior_scores(cfg, cands, ctx, ctx.cache_dir)
    log_w = prior_log_weights(scores, cfg.prior_T, cfg.prior_transform)
    selector = make_selector(cfg, cands, ctx, np.random.default_rng(cfg.seed))
    oracle = make_oracle(cfg, cands, cfg.processed_root, cache_dir)

    tiebreak_rng = np.random.default_rng([cfg.seed, 1])
    records, weights = [], []

    def record(t, log_w, query):
        if query is not None:
            i, l = query.pop("i"), query.pop("l")
            query = {"pair": [cands.img_names[i], cands.img_names[l]], **query,
                     "y_true": int(cands.gt[i] == cands.gt[l])}
        records.append({"t": t, "query": query, **belief_record(cands, log_w, tiebreak_rng, cfg.top_k)})
        weights.append([float(f"{w:.4g}") for w in np.exp(log_w)])

    record(0, log_w, None)
    run_active_selection(log_w, selector, oracle, cfg.epsilon, cfg.budget, record)

    final = records[-1]
    print(f"[{cands.dataset}] {os.path.basename(os.path.dirname(out_dir))}: ID-F1 "
          f"{records[0]['metrics']['id_f1']:.3f} (prior) -> {final['metrics']['id_f1']:.3f} after "
          f"{final['t']} queries; best {cands.metrics['id_f1'].max():.3f}  ({time.time() - start:.0f}s)")

    write_json(os.path.join(out_dir, "results.json"), {
        "experiment": experiment_name(cfg), "dataset": cands.dataset,
        "n_candidates": cands.M, "n_points": cands.N, "true_n_clusters": cands.true_n_clusters,
        "top_k": cfg.top_k,
        # oracle-best candidate, and the mean of the true top-k, per supervised metric
        "best": {m: cands.metrics[m].max() for m in SUPERVISED_METRICS},
        "top_k_best": {m: np.sort(cands.metrics[m])[::-1][:cfg.top_k].mean() for m in SUPERVISED_METRICS},
        "queries": records,
    })
    write_json(os.path.join(out_dir, "weights.json"), {"candidates": cands.keys, "weights": weights})
    write_json(os.path.join(out_dir, "prior.json"), {
        "prior": cfg.prior, "transform": cfg.prior_transform, "T": cfg.prior_T,
        "scores": scores, "weights": np.exp(prior_log_weights(scores, cfg.prior_T, cfg.prior_transform)),
        "info": prior_info,
    })
    write_json(os.path.join(out_dir, "candidates.json"), cands.to_json())
    config = {k: v for k, v in vars(cfg).items() if k not in ("experiments", "metrics", "table_metrics",
                                                              "eval_at", "max_queries", "ema_alpha")}
    write_json(os.path.join(out_dir, "config.json"), {
        **config, "experiment": experiment_name(cfg), "dataset": cands.dataset,
        "candidates_fingerprint": cands.fingerprint, "selector_stats": selector.stats,
        "n_queries": final["t"], "runtime_s": time.time() - start,
    }, indent=2)


def run_dataset(dataset, args):
    todo = []
    for cfg in experiment_configs(args):
        out_dir = os.path.join(args.results_root, experiment_name(cfg), dataset)
        if args.overwrite or not os.path.exists(os.path.join(out_dir, "results.json")):
            todo.append((cfg, out_dir))
    if not todo:
        print(f"[{dataset}] every experiment already has results (--overwrite to re-run)")
        return

    cands = load_candidates(dataset, args.predictions_root, args.processed_root)
    print(f"[{dataset}] {cands.M} candidates, {cands.N:,} points, {cands.true_n_clusters} true clusters")
    ctx = PairwiseContext(cands.labels, os.path.join(args.cache_root, dataset, cands.fingerprint))
    for cfg, out_dir in todo:
        try:
            run_experiment(cfg, cands, ctx, out_dir)
        except Exception:
            print(f"[{dataset}] {experiment_name(cfg)} FAILED:\n{traceback.format_exc()}")


def main():
    args = parse_args(__doc__.splitlines()[0])
    datasets = list_datasets(args.predictions_root) if args.datasets == ["all"] else args.datasets
    print(f"{len(datasets)} dataset(s): {', '.join(datasets)}")
    if args.n_jobs <= 1:
        for dataset in datasets:
            run_dataset(dataset, args)
    else:
        with concurrent.futures.ProcessPoolExecutor(args.n_jobs) as pool:
            list(pool.map(run_dataset, datasets, [args] * len(datasets)))


if __name__ == "__main__":
    main()
