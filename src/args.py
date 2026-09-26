"""Arguments shared by active_selection.py (runs experiments) and plot_saved_results.py (metrics and
plots from their saved results). Path defaults come from src/config.py.

The experiment configuration arguments --prior, --select, --epsilon and --seed take comma-separated
lists; active_selection.py runs every combination (experiment_configs). Each experiment is saved to
<results_root>/<experiment_name>/<dataset>/.
"""
import copy
import argparse
import itertools

from . import config
from .active import SELECTORS
from .oracles import ORACLES
from .priors import PRIORS, CONSENSUS_METHODS

DEFAULT_T = 0.1


def _csv(type_=str, choices=None):
    def parse(text):
        values = [type_(v) for v in text.split(",") if v.strip()]
        bad = [v for v in values if choices is not None and v not in choices]
        if bad:
            raise argparse.ArgumentTypeError(f"{bad} not in {list(choices)}")
        return values
    return parse


def get_parser(description=None):
    parser = argparse.ArgumentParser(description=description)

    paths = parser.add_argument_group("paths")
    paths.add_argument("--processed_root", default=config.PROCESSED_DATA_ROOT,
                       help="Datasets processed by src/process_datasets.py (img_name2label.json, images/)")
    paths.add_argument("--predictions_root", default=config.PREDICTIONS_ROOT,
                       help="cluster_benchmark.py output: <run>/predictions/<dataset>/<dataset>_<features>_preds.json")
    paths.add_argument("--results_root", default=config.ACTIVE_RESULTS_ROOT,
                       help="Where experiments are saved: <results_root>/<experiment>/<dataset>/*.json")
    paths.add_argument("--plots_root", default=config.PLOTS_ROOT, help="Where tables and plots are saved")
    paths.add_argument("--cache_root", default=None,
                       help="Cache of co-clustering products, prior scores and LLM answers "
                            "(default: <results_root>/_cache)")
    paths.add_argument("--datasets", type=_csv(), default=["all"],
                       help="Comma-separated datasets, or 'all' (every dataset with predictions / results)")

    exp = parser.add_argument_group("experiment configuration (lists run as a cross product)")
    exp.add_argument("--prior", type=_csv(choices=PRIORS), default=["pairwise"],
                     help=f"Prior over the best candidate, w^(0) (Eq. 5): {', '.join(PRIORS)}")
    exp.add_argument("--select", type=_csv(choices=SELECTORS), default=["entropy"],
                     help=f"Query selection: {', '.join(SELECTORS)}")
    exp.add_argument("--epsilon", type=_csv(float), default=[0.1],
                     help="Oracle error probability eps in (0, 0.5) of the observation model (Eq. 6)")
    exp.add_argument("--seed", type=_csv(int), default=[0],
                     help="Seed of the query pool and of random pair selection / tie-breaking")
    exp.add_argument("--budget", type=int, default=100, help="Number of pairwise queries B")
    exp.add_argument("--prior_T", type=float, default=DEFAULT_T, help="Softmax temperature T of the prior (Eq. 5)")
    exp.add_argument("--prior_transform", choices=["raw", "zscore", "rank"], default="raw",
                     help="Standardization of prior scores before the softmax (raw = as in Eq. 5)")
    exp.add_argument("--tau", type=float, default=0.5, help="Consensus threshold tau (Eq. 2)")
    exp.add_argument("--consensus_agreement", choices=["nmi", "ari", "id_f1", "pairwise_f1"], default="nmi",
                     help="How candidates are scored against a consensus clustering "
                          f"(priors {', '.join(CONSENSUS_METHODS)})")
    exp.add_argument("--consensus_n_raters", type=int, default=None,
                     help="Build consensus clusterings from this many random candidates (default: all)")
    exp.add_argument("--pool_size", type=int, default=200_000,
                     help="Disagreement pairs sampled as the pool for --select entropy")
    exp.add_argument("--oracle", choices=ORACLES, default="ground_truth", help="Who answers the queries")
    exp.add_argument("--oracle_model", default=None, help="Override the LLM oracle's default model")
    exp.add_argument("--exp_tag", default="", help="Suffix appended to the experiment name")
    exp.add_argument("--top_k", type=int, default=10, help="k of the mean-top-k metrics")

    run = parser.add_argument_group("running")
    run.add_argument("--overwrite", action="store_true", help="Re-run experiments that already have results")
    run.add_argument("--n_jobs", type=int, default=1, help="Datasets processed in parallel")

    analysis = parser.add_argument_group("analysis (plot_saved_results.py)")
    analysis.add_argument("--experiments", type=_csv(), default=["prior_*"],
                          help="Experiment directories (glob patterns allowed) under --results_root")
    analysis.add_argument("--metrics", type=_csv(),
                          default=["id_f1", "nmi", "ari", "id_f1_macro", "homogeneity", "completeness", "v_measure"],
                          help="Metrics to plot")
    analysis.add_argument("--table_metrics", type=_csv(), default=["id_f1", "nmi"],
                          help="Metrics whose regret / AUC go into the tables")
    analysis.add_argument("--eval_at", type=_csv(int), default=[10, 50, 100],
                          help="Numbers of queries at which the tables are computed")
    analysis.add_argument("--max_queries", type=int, default=None, help="x-axis limit of the line plots")
    analysis.add_argument("--ema_alpha", type=float, default=None,
                          help="Also save line plots smoothed by an exponential moving average")
    return parser


def parse_args(description=None):
    args = get_parser(description).parse_args()
    if args.cache_root is None:
        args.cache_root = f"{args.results_root}/_cache"
    return args


def experiment_configs(args):
    """One namespace per combination of the list-valued experiment arguments."""
    for prior, select, epsilon, seed in itertools.product(args.prior, args.select, args.epsilon, args.seed):
        cfg = copy.copy(args)
        cfg.prior, cfg.select, cfg.epsilon, cfg.seed = prior, select, epsilon, seed
        yield cfg


def experiment_name(cfg):
    """prior_<prior>_select_<select>_eps_<eps>, where <prior> also names the consensus agreement metric,
    a non-raw transform and a non-default T; non-default oracle, seed and --exp_tag are appended."""
    prior = cfg.prior
    if cfg.prior in CONSENSUS_METHODS:
        prior += f"-{cfg.consensus_agreement}"
    if cfg.prior != "uniform":
        if cfg.prior_transform != "raw":
            prior += f"-{cfg.prior_transform}"
        if cfg.prior_T != DEFAULT_T:
            prior += f"-T{cfg.prior_T:g}"
    name = f"prior_{prior}_select_{cfg.select}_eps_{cfg.epsilon:g}"
    if cfg.oracle != "ground_truth":
        name += f"_oracle_{cfg.oracle}"
    if cfg.seed != 0:
        name += f"_seed_{cfg.seed}"
    if cfg.exp_tag:
        name += f"_{cfg.exp_tag}"
    return name
