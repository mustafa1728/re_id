import os

_WORKSPACE = "/scratch3/workspace/mchasmai_umass_edu-re_id"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Original downloads of every dataset
DATA_ROOT = f"{_WORKSPACE}/data"
# Datasets converted to one common format (and their features) by src/process_datasets.py
PROCESSED_DATA_ROOT = f"{_WORKSPACE}/processed_datasets"
# Candidate clusterings written by cluster_benchmark.py:
#   <PREDICTIONS_ROOT>/<run>/predictions/<dataset>/<dataset>_<features>_preds.json
PREDICTIONS_ROOT = f"{_WORKSPACE}/results/full_benchmark_results/09_25"
# Experiments written by active_selection.py:
#   <ACTIVE_RESULTS_ROOT>/<experiment>/<dataset>/{config,candidates,prior,results,weights}.json
ACTIVE_RESULTS_ROOT = f"{_WORKSPACE}/results/active_selection"
# Tables and plots written by plot_saved_results.py
PLOTS_ROOT = os.path.join(_REPO_ROOT, "plots", "active_selection")
