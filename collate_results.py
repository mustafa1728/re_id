import matplotlib.pyplot as plt
import os
import json
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Collate clustering results and generate plots")
parser.add_argument("--feature_type", type=str, default="Original")
args = parser.parse_args()

if args.feature_type == "Original":

    results_dir = "./benchmark_results/latest"
    PLOT_DIR = "./plots/latest"
    FEATURE_NAME = "Original"

elif args.feature_type == "UMAP_128D":

    results_dir = "./benchmark_results/latest/umap_128"
    PLOT_DIR = "./plots/latest/umap_128"
    FEATURE_NAME = "UMAP_128D"

elif args.feature_type == "PCA_128D":

    results_dir = "./benchmark_results/latest/pca_128"
    PLOT_DIR = "./plots/latest/pca_128"
    FEATURE_NAME = "PCA_128D"

DATASETS = ["cars", "aircraft", "stanford_products_micro", "imat_products_nano_class", "vehicle_reid", "more"]
DATASET_LABELS = {
    "cars": "Stanford Cars",
    "aircraft": "Aircraft",
    "stanford_products_nano": "St Products",
    "stanford_products_mini": "St Products",
    "stanford_products_micro": "St Products",
    "imat_products_nano": "IMAT Products",
    "imat_products_nano_class": "IMAT Products",
    "vehicle_reid": "AI City Vehicle ReID",
    "more": "Motorcycle ReID",
}
ALGORITHMS = ["KMeans", "KMeans_Unknown", "Spectral", "Spectral_Unknown", "Agglomerative", "Agglomerative_Unknown", "HDBSCAN", "FINCH"]
UMAP_DIMS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
os.makedirs(PLOT_DIR, exist_ok=True)


all_results = {}
for dset in DATASETS:
    for fname in os.listdir(os.path.join(results_dir, dset)):
        if fname.endswith(".json"):
            if "dim128" in fname or "db_only" in fname: continue
            with open(os.path.join(results_dir, dset, fname), "r") as f:
                result = json.load(f)
                all_results[fname] = result

def parse_filename(fname):
    # "stanford_products_mini", "stanford_products_micro", "stanford_products_nano",
    datasets = DATASETS
    models = ["dinov2", "dinov3"]
    backbones = ["vitb16", "vitb14", "vitl14", "vitg14"]

    dataset = None
    model = None
    backbone = None

    for d in datasets:
        if d in fname:
            wrong_mod = False
            for mod in ["mini", "micro", "nano"]:
                if mod in fname and not mod in d:
                    wrong_mod = True
            if not wrong_mod:
                dataset = d
            break
    for m in models:
        if m in fname:
            model = m
            break
    for b in backbones:
        if b in fname:
            backbone = b
            break
    return dataset, model, backbone

# # DINOv2 vs DINOv3
# dino2_results = {}
# dino3_results = {}
# for fname, result in all_results.items():
#     dataset, model, backbone = parse_filename(fname)
#     print(fname, dataset, model, backbone)
#     if model == "dinov2" and backbone == "vitb14":
#         dino2_results[dataset] = result
#     elif model == "dinov3":
#         dino3_results[dataset] = result

# for algorithm in ALGORITHMS:

#     # x = [dataset for dataset in dino2_results.keys() if dataset in dino3_results]
#     x = DATASETS
#     x_i = np.arange(len(x))
#     dino2_y = [dino2_results[dataset][FEATURE_NAME][algorithm]["v_measure"] for dataset in x]
#     dino3_y = [dino3_results[dataset][FEATURE_NAME][algorithm]["v_measure"] for dataset in x]

#     plt.figure(figsize=(4, 3))
#     plt.bar(x_i - 0.2, dino2_y, width=0.4, label="DINOv2")
#     plt.bar(x_i + 0.2, dino3_y, width=0.4, label="DINOv3")
#     plt.ylabel("V-measure")
#     plt.title("DINOv2 vs DINOv3 " + algorithm + " Clustering")
#     plt.legend()
#     plt.xticks(x_i, [DATASET_LABELS.get(dataset, dataset) for dataset in x], rotation=15, ha='right')
#     plt.tight_layout()
#     plt.savefig(os.path.join(PLOT_DIR, f"dino_comparison_{algorithm.lower()}.png"), dpi=300, bbox_inches='tight')




# print ressults in a table format
for dataset in DATASETS:
    fname = f"{dataset}_dinov3_vitb16.json"
    if fname not in all_results: continue
    result = all_results[fname]
    print(DATASET_LABELS.get(dataset, dataset), "="*20)
    print("Alorithm,Time(s),Homogeneity,Completeness,V-Measure,N_Clusters")
    for alg, metrics in result[FEATURE_NAME].items():
        print(f"{alg},{metrics['time']:.4f},{metrics['homogeneity']:.3f},{metrics['completeness']:.3f},{metrics['v_measure']:.3f},{metrics['n_clusters']}")



# # variation of DINOV3 with different UMAP dimensions
# for dataset in DATASETS:
#     dino3_umap_scores = {}
#     dino3_umap_times = {}
#     fname = f"{dataset}_dinov3_vitb16.json"
#     if fname not in all_results: continue
#     result = all_results[fname]
#     for algorithm in ALGORITHMS:
#         dino3_umap_scores[algorithm] = {}
#         dino3_umap_times[algorithm] = {}
#         for dim in UMAP_DIMS:
#             dim_key = f"UMAP_{dim}D"
#             if dim_key in result and algorithm in result[dim_key]:
#                 dino3_umap_scores[algorithm][dim] = result[dim_key][algorithm]["v_measure"]
#                 dino3_umap_times[algorithm][dim] = result[dim_key][algorithm]["time"]
#         dino3_umap_scores[algorithm][768] = result[FEATURE_NAME][algorithm]["v_measure"]
#         dino3_umap_times[algorithm][768] = result[FEATURE_NAME][algorithm]["time"]
#     # Plotting
#     plt.figure(figsize=(4, 3))
#     max_y = 0
#     min_y = 1.0
#     for algorithm, scores in dino3_umap_scores.items():
#         dims = list(scores.keys())
#         v_measures = list(scores.values())
#         max_y = max(max_y, max(v_measures))
#         min_y = min(min_y, min(v_measures))
#         plt.plot(dims, v_measures, label=algorithm)
#     plt.xscale('log', base=2)
#     plt.xlabel("UMAP Dimensions")
#     plt.ylabel("V-measure")
#     plt.title(f"DINOv3 UMAP - {DATASET_LABELS.get(dataset, dataset)}")
#     plt.legend()
#     min_y = max(min_y, 0.5)
#     plt.ylim(min_y - 0.02, max_y + 0.02)
#     plt.tight_layout()
#     plt.savefig(os.path.join(PLOT_DIR, f"dino3_umap_{dataset}.png"), dpi=300, bbox_inches='tight')

#     # Plotting
#     plt.figure(figsize=(4, 3))
#     max_y = 0
#     min_y = 1.0
#     for algorithm, times in dino3_umap_times.items():
#         dims = list(times.keys())
#         t = list(times.values())
#         plt.plot(dims, t, label=algorithm)
#     plt.xscale('log', base=2)
#     plt.yscale('log')
#     plt.xlabel("UMAP Dimensions")
#     plt.ylabel("Time (s)")
#     plt.title(f"DINOv3 UMAP - {DATASET_LABELS.get(dataset, dataset)}")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(PLOT_DIR, f"dino3_umap_time_{dataset}.png"), dpi=300, bbox_inches='tight')


# Make table with this format

# Algorithm, Dataset1, , , Dataset2, , ,
# , V-Measure, ARI, NMI, V-Measure, ARI, NMI

with open(os.path.join(PLOT_DIR, "clustering_quality.csv"), "w") as f:
    header = "Algorithm"
    for dataset in DATASETS:
        header += f",{DATASET_LABELS.get(dataset, dataset)},,"
    f.write(header + "\n")
    subheader = ""
    for dataset in DATASETS:
        subheader += ",V-Measure,ARI,NMI"
    f.write(subheader + "\n")
    for algorithm in ALGORITHMS:
        row = algorithm
        for dataset in DATASETS:
            fname = f"{dataset}_dinov3_vitb16.json"
            if fname not in all_results:
                row += ",,,"
                continue
            result = all_results[fname]
            if algorithm in result[FEATURE_NAME]:
                metrics = result[FEATURE_NAME][algorithm]
                row += f",{metrics['v_measure']:.3f},{metrics['ari']:.3f},{metrics['nmi']:.3f}"
            else:
                row += ",,,"
        f.write(row + "\n")



# Make table with this format where each row has times for each algo

# Algorithm,Dataset1,Dataset2,

with open(os.path.join(PLOT_DIR, "clustering_runtime.csv"), "w") as f:
    header = "Algorithm"
    for dataset in DATASETS:
        header += f",{DATASET_LABELS.get(dataset, dataset)}"
    f.write(header + "\n")
    for algorithm in ALGORITHMS:
        row = algorithm
        for dataset in DATASETS:
            fname = f"{dataset}_dinov3_vitb16.json"
            if fname not in all_results:
                row += ","
                continue
            result = all_results[fname]
            if algorithm in result[FEATURE_NAME]:
                metrics = result[FEATURE_NAME][algorithm]
                row += f",{metrics['time']:.1f}"
            else:
                row += ","
        f.write(row + "\n")


# make bar plots
for dataset in DATASETS:
    fname = f"{dataset}_dinov3_vitb16.json"
    if fname not in all_results: continue
    result = all_results[fname]
    for metric in ["time", "v_measure", "ari", "nmi"]:
        # Build base->pair mapping and ordered display list
        pairs = {}
        seen = set()
        display = []
        for alg in ALGORITHMS:
            base = alg[:-len("_Unknown")] if alg.endswith("_Unknown") else alg
            if base not in pairs:
                pairs[base] = {"known": None, "unknown": None}
            if alg.endswith("_Unknown"):
                pairs[base]["unknown"] = alg
            else:
                pairs[base]["known"] = alg
            if base not in seen:
                seen.add(base)
                display.append(base)

        valid = [b for b in display if
                 (pairs[b]["unknown"] and pairs[b]["unknown"] in result[FEATURE_NAME]) or
                 (pairs[b]["known"]   and pairs[b]["known"]   in result[FEATURE_NAME])]

        plt.figure(figsize=(5, 3))
        legend_added = set()
        for i, base in enumerate(valid):
            pair = pairs[base]
            has_both = pair["known"] and pair["unknown"] and \
                       pair["known"] in result[FEATURE_NAME] and \
                       pair["unknown"] in result[FEATURE_NAME]

            if has_both:
                y_known = result[FEATURE_NAME][pair["known"]][metric]
                lbl = "Known k" if "known" not in legend_added else "_nolegend_"
                legend_added.add("known")
                plt.bar(i, y_known, width=0.8, color="tab:blue", alpha=0.3, label=lbl)
                plt.hlines(y_known, i - 0.4, i + 0.4, colors="black", linestyles="--", linewidth=1.5, zorder=3)

            primary = pair["unknown"] or pair["known"]
            if primary and primary in result[FEATURE_NAME]:
                lbl = "Unknown k" if (has_both and "unknown" not in legend_added) else "_nolegend_"
                if has_both: legend_added.add("unknown")
                plt.bar(i, result[FEATURE_NAME][primary][metric],
                        width=0.8, color="tab:blue", alpha=1.0, label=lbl)

        plt.xticks(range(len(valid)), valid, rotation=15, ha='right')
        if "known" in legend_added:
            plt.legend(fontsize=8, loc="lower right")
        plt.ylabel(metric)
        plt.title(f"{DATASET_LABELS.get(dataset, dataset)}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{dataset}_{metric}.png"), dpi=300, bbox_inches='tight')
        plt.close()



# make bar plots
# for metric in ["time", "v_measure", "ari", "nmi"]:
#     plt.figure(figsize=(5, 3))
#     x = np.arange(len(ALGORITHMS))
#     bar_width = 0.8
#     for d_i, dataset in enumerate(DATASETS):
#         x_cur = x + (d_i - len(DATASETS)/2) * (bar_width / len(DATASETS))
#         fname = f"{dataset}_dinov3_vitb16.json"
#         if fname not in all_results: continue
#         result = all_results[fname]

#         ys = []
#         algs = []
#         x_cur = []
#         for alg in ALGORITHMS:
#             if alg not  in result[FEATURE_NAME]: continue
#             ys.append(result[FEATURE_NAME][alg][metric])
#             algs.append(alg)
#             x_cur.append(ALGORITHMS.index(alg) + (d_i - len(DATASETS)/2) * (bar_width / len(DATASETS)))
#         max_yi, max_y = max(enumerate(ys), key=lambda x: x[1])

#         plt.bar(x_cur, ys, width=bar_width/len(DATASETS), label=DATASET_LABELS.get(dataset, dataset))

#     plt.xticks(x, ALGORITHMS)
#     # plt.bar(ALGORITHMS, ys)
#     # plt.bar(ALGORITHMS[max_yi], max_y)
#     plt.ylabel(metric)
#     plt.legend()
#     plt.title(f"Clustering {metric.capitalize()} Comparison")
#     plt.tight_layout()
#     plt.savefig(os.path.join(PLOT_DIR, f"{metric}.png"), dpi=300, bbox_inches='tight')
#     plt.close()




# make bar plots, dataset-mean
for metric in ["time", "v_measure", "ari", "nmi"]:
    plt.figure(figsize=(5, 3))
    bar_width = 0.8

    # Group algorithms: base_name -> {known: alg_or_None, unknown: alg_or_None}
    alg_pairs = {}
    for alg in ALGORITHMS:
        base = alg[:-len("_Unknown")] if alg.endswith("_Unknown") else alg
        if base not in alg_pairs:
            alg_pairs[base] = {"known": None, "unknown": None}
        if alg.endswith("_Unknown"):
            alg_pairs[base]["unknown"] = alg
        else:
            alg_pairs[base]["known"] = alg

    # Preserve ordering from ALGORITHMS
    seen = set()
    display_algs = []
    for alg in ALGORITHMS:
        base = alg[:-len("_Unknown")] if alg.endswith("_Unknown") else alg
        if base not in seen:
            seen.add(base)
            display_algs.append(base)

    # Compute dataset-mean score for each algorithm
    algo2score = {}
    for dataset in DATASETS:
        fname = f"{dataset}_dinov3_vitb16.json"
        if fname not in all_results: continue
        result = all_results[fname]
        for alg in ALGORITHMS:
            if alg not in algo2score:
                algo2score[alg] = []
            if alg not in result[FEATURE_NAME]:
                algo2score[alg].append(None)
            else:
                algo2score[alg].append(result[FEATURE_NAME][alg][metric])

    def get_mean(alg):
        if alg is None or alg not in algo2score: return None
        scores = algo2score[alg]
        if any(s is None for s in scores): return None
        return np.mean(scores)

    # Only show base algs where the primary (unknown or known) has full data
    valid_display = [b for b in display_algs if get_mean(alg_pairs[b]["unknown"] or alg_pairs[b]["known"]) is not None]
    x = np.arange(len(valid_display))

    legend_added = set()
    for i, base in enumerate(valid_display):
        pair = alg_pairs[base]
        has_both = pair["known"] is not None and pair["unknown"] is not None

        # Faded bar for known-k variant (drawn first so unknown overlaps it)
        if has_both:
            y_known = get_mean(pair["known"])
            if y_known is not None:
                label = "Known k" if "known" not in legend_added else "_nolegend_"
                legend_added.add("known")
                plt.bar(i, y_known, width=bar_width, color="tab:blue", alpha=0.3, label=label)
                plt.hlines(y_known, i - bar_width/2, i + bar_width/2, colors="black", linestyles="--", linewidth=1.5, zorder=3)

        # Solid bar for unknown-k variant (or standalone algorithm)
        primary = pair["unknown"] or pair["known"]
        y_primary = get_mean(primary)
        if y_primary is not None:
            label = "Unknown k" if (has_both and "unknown" not in legend_added) else "_nolegend_"
            if has_both: legend_added.add("unknown")
            plt.bar(i, y_primary, width=bar_width, color="tab:blue", alpha=1.0, label=label)

    plt.xticks(x, valid_display, rotation=15, ha='right')
    plt.ylabel(metric.capitalize())
    plt.title(f"Clustering {metric.capitalize()} Comparison")
    if "known" in legend_added:
        plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"data_mean_{metric}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Spectral_Unknown vs HDBSCAN vs FINCH: dataset-mean comparison
_cmp_three = ["Spectral_Unknown", "HDBSCAN", "FINCH"]
_cmp_means = {alg: {} for alg in _cmp_three}
for metric in ["v_measure", "ari", "nmi", "time"]:
    for alg in _cmp_three:
        scores = []
        for dataset in DATASETS:
            fname = f"{dataset}_dinov3_vitb16.json"
            if fname not in all_results: continue
            r = all_results[fname][FEATURE_NAME]
            if alg in r: scores.append(r[alg][metric])
        _cmp_means[alg][metric] = np.mean(scores) if scores else None

print(f"\nDataset-mean comparison (dinov3_vitb16, {FEATURE_NAME})")
for metric in ["v_measure", "ari", "nmi", "time"]:
    vals = {alg: _cmp_means[alg][metric] for alg in _cmp_three}
    if any(v is None for v in vals.values()): continue
    print(f"\n  {metric.upper()}")
    for alg, v in vals.items():
        print(f"    {alg:<20} {v:.4f}")
    pairs = [("Spectral_Unknown", "HDBSCAN"), ("Spectral_Unknown", "FINCH"), ("HDBSCAN", "FINCH")]
    for a, b in pairs:
        diff = vals[a] - vals[b]
        rel  = diff / vals[b] if vals[b] != 0 else float("nan")
        print(f"    {a} - {b}: {diff:+.4f} ({rel:+.1%})")


# make bar plots, found clusters vs true clusters
unknown_algs = [alg for alg in ALGORITHMS if alg.endswith("_Unknown") or alg in ("HDBSCAN", "FINCH")]
unknown_labels = [alg.replace("_Unknown", "") for alg in unknown_algs]

valid_datasets = [d for d in DATASETS if f"{d}_dinov3_vitb16.json" in all_results]
n_algs = len(unknown_algs)
n_dsets = len(valid_datasets)
bar_width = 0.8 / n_algs
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(figsize=(8, 3))
x = np.arange(n_dsets)

for a_i, (alg, alg_label) in enumerate(zip(unknown_algs, unknown_labels)):
    color = colors[a_i % len(colors)]
    x_cur, ys = [], []
    for d_i, dataset in enumerate(valid_datasets):
        result = all_results[f"{dataset}_dinov3_vitb16.json"]
        if alg in result[FEATURE_NAME]:
            x_cur.append(d_i + (a_i - n_algs / 2 + 0.5) * bar_width)
            ys.append(result[FEATURE_NAME][alg]["n_clusters"])
    ax.bar(x_cur, ys, width=bar_width, color=color, label=alg_label)

# Draw true_k as a short dashed line spanning only each dataset's group
true_k_legend_added = False
for d_i, dataset in enumerate(valid_datasets):
    result = all_results[f"{dataset}_dinov3_vitb16.json"]
    true_k = next(
        (result[FEATURE_NAME][alg]["true_clusters"] for alg in unknown_algs if alg in result[FEATURE_NAME]),
        None
    )
    if true_k is not None:
        label = "True k" if not true_k_legend_added else "_nolegend_"
        true_k_legend_added = True
        ax.hlines(true_k, d_i - 0.4, d_i + 0.4, colors="black", linestyles="--", linewidth=1.5, label=label)

ax.set_yscale("log")
ax.set_ylim(bottom=10)
ax.set_xticks(x)
ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in valid_datasets], rotation=15, ha='right')
ax.set_ylabel("# Clusters (log scale)")
ax.set_title("Predicted vs True Number of Clusters")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "n_clusters_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()



# architecture variations for DINOv3
from matplotlib.patches import Patch

MODELS = ["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16plus"]
MODEL_LABELS = {
    "dinov3_vits16":    "ViT-S/16",
    "dinov3_vitb16":    "ViT-B/16",
    "dinov3_vitl16":    "ViT-L/16",
    "dinov3_vith16plus":"ViT-H/16+",
}

# Group algorithms into base -> {known, unknown} pairs (reused each metric loop)
_alg_pairs = {}
for _alg in ALGORITHMS:
    _base = _alg[:-len("_Unknown")] if _alg.endswith("_Unknown") else _alg
    if _base not in _alg_pairs:
        _alg_pairs[_base] = {"known": None, "unknown": None}
    if _alg.endswith("_Unknown"):
        _alg_pairs[_base]["unknown"] = _alg
    else:
        _alg_pairs[_base]["known"] = _alg

_seen = set()
_display_algs = []
for _alg in ALGORITHMS:
    _base = _alg[:-len("_Unknown")] if _alg.endswith("_Unknown") else _alg
    if _base not in _seen:
        _seen.add(_base)
        _display_algs.append(_base)

for metric in ["time", "v_measure", "ari", "nmi"]:
    fig, ax = plt.subplots(figsize=(9, 3))
    bar_width = 0.8
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_models = len(MODELS)
    sub_width = bar_width / n_models
    x = np.arange(len(_display_algs))

    # Compute dataset-mean score for each (model, alg)
    model_algo_scores = {}
    for model in MODELS:
        for dataset in DATASETS:
            fname = f"{dataset}_{model}.json"
            if fname not in all_results: continue
            result = all_results[fname]
            for alg in ALGORITHMS:
                key = (model, alg)
                if key not in model_algo_scores:
                    model_algo_scores[key] = []
                if alg not in result[FEATURE_NAME]:
                    model_algo_scores[key].append(None)
                else:
                    model_algo_scores[key].append(result[FEATURE_NAME][alg][metric])

    def get_arch_mean(model, alg):
        key = (model, alg)
        if key not in model_algo_scores: return None
        scores = model_algo_scores[key]
        if any(s is None for s in scores): return None
        return np.mean(scores)

    model_legend_added = set()
    for m_i, model in enumerate(MODELS):
        color = colors[m_i % len(colors)]
        for b_i, base in enumerate(_display_algs):
            pair = _alg_pairs[base]
            has_both = pair["known"] is not None and pair["unknown"] is not None
            x_pos = b_i + (m_i - n_models / 2 + 0.5) * sub_width

            # Faded bar for known-k (drawn first so unknown overlaps it)
            if has_both:
                y_known = get_arch_mean(model, pair["known"])
                if y_known is not None:
                    ax.bar(x_pos, y_known, width=sub_width, color=color, alpha=0.3)
                    ax.hlines(y_known, x_pos - sub_width/2, x_pos + sub_width/2, colors="black", linestyles="--", linewidth=1.5, zorder=3)

            # Solid bar for unknown-k (or standalone)
            primary = pair["unknown"] or pair["known"]
            y_primary = get_arch_mean(model, primary)
            if y_primary is not None:
                label = MODEL_LABELS.get(model, model) if model not in model_legend_added else "_nolegend_"
                model_legend_added.add(model)
                ax.bar(x_pos, y_primary, width=sub_width, color=color, alpha=1.0, label=label)

    # Append known/unknown proxy entries to legend
    arch_handles, arch_labels = ax.get_legend_handles_labels()
    # arch_handles += [Patch(facecolor='gray', alpha=1.0), Patch(facecolor='gray', alpha=0.3)]
    # arch_labels  += ["Unknown k", "Known k"]
    ax.legend(handles=arch_handles, labels=arch_labels, fontsize=8, loc="lower right")

    ax.set_xticks(x)
    ax.set_xticklabels(_display_algs, rotation=15, ha='right')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Architecture Comparison -- {metric.capitalize()}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"arch_{metric}.png"), dpi=300, bbox_inches='tight')
    plt.close()



# For HDBSCAN and FINCH, compare performance with original features, PCA-128D, UMAP-128D, and SpectralHDBSCAN
_COMPARE_CONFIGS = [
    ("./benchmark_results/latest",             "Original",  "Original"),
    ("./benchmark_results/latest/pca_128_v1",  "PCA_128D",  "PCA-128D"),
    ("./benchmark_results/latest/umap_128_v1", "UMAP_128D", "UMAP-128D"),
]

_compare_results = {}
for _rdir, _fkey, _ in _COMPARE_CONFIGS:
    _res = {}
    for _dset in DATASETS:
        _dpath = os.path.join(_rdir, _dset)
        if not os.path.exists(_dpath): continue
        for _fname in os.listdir(_dpath):
            if _fname.endswith(".json") and "dim128" not in _fname and "db_only" not in _fname:
                with open(os.path.join(_dpath, _fname)) as _f:
                    _res[_fname] = json.load(_f)
    _compare_results[_fkey] = _res

def _cmean(feat_key, alg, metric):
    scores = []
    for dset in DATASETS:
        fname = f"{dset}_dinov3_vitb16.json"
        r = _compare_results.get(feat_key, {}).get(fname, {})
        if feat_key in r and alg in r[feat_key]:
            scores.append(r[feat_key][alg][metric])
    return np.mean(scores) if scores else None

# Per-algorithm bar specs: (feat_key, alg_in_results, display_label)
# SpectralHDBSCAN is treated as a dim-reduction for HDBSCAN, read from Original results
_alg_bars = {
    "HDBSCAN": [
        ("Original",  "HDBSCAN",        "Original"),
        ("PCA_128D",  "HDBSCAN",        "PCA-128D"),
        ("UMAP_128D", "HDBSCAN",        "UMAP-128D"),
        ("Original",  "SpectralHDBSCAN","Spectral-128D"),
    ],
    "FINCH": [
        ("Original",  "FINCH", "Original"),
        ("PCA_128D",  "FINCH", "PCA-128D"),
        ("UMAP_128D", "FINCH", "UMAP-128D"),
        ("Original",  "SpectralFINCH","Spectral-128D"),
    ],
}
_bar_w    = 0.25
_group_gap = 0.1
_dim_red_labels = ["Original", "PCA-128D", "UMAP-128D", "Spectral-128D"]

for metric in ["v_measure", "ari", "nmi", "time"]:
    fig, ax = plt.subplots(figsize=(4, 3))
    _ccolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    _color_map = {lbl: _ccolors[i] for i, lbl in enumerate(_dim_red_labels)}
    _legend_added = set()

    _group_centers = []
    _x_cursor = _group_gap
    for alg, bars in _alg_bars.items():
        _group_start = _x_cursor
        for feat_key, result_alg, dim_label in bars:
            y = _cmean(feat_key, result_alg, metric)
            if y is not None:
                lbl = dim_label if dim_label not in _legend_added else "_nolegend_"
                _legend_added.add(dim_label)
                ax.bar(_x_cursor, y, width=_bar_w, color=_color_map[dim_label], label=lbl)
            _x_cursor += _bar_w
        _group_centers.append((_group_start + _x_cursor - _bar_w) / 2)
        _x_cursor += _group_gap

    ax.set_xticks(_group_centers)
    ax.set_xticklabels(list(_alg_bars.keys()))
    ax.set_ylabel(metric.capitalize())
    ax.set_title("HDBSCAN & FINCH -- Dim Reduction Comparison")
    ax.legend(title="Dim Reduction", fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join("./plots", f"hdbscan_finch_{metric}.png"), dpi=300, bbox_inches='tight')
    plt.close()


# HDBSCAN and FINCH variation across UMAP / PCA dimensions
_DIM_PLOT_DIMS = [8, 16, 32, 64, 128, 256, 512]
_DIM_ALGS = ["HDBSCAN", "FINCH"]
_dim_colors = {"HDBSCAN": "tab:blue", "FINCH": "tab:orange"}

_dim_reduction_configs = [
    ("umap", "UMAP"),
    ("pca",  "PCA"),
]

def _load_dim_results(reduction, dims):
    out = {}
    for dim in dims:
        _dir = f"./benchmark_results/latest/{reduction}_{dim}"
        _key = f"{reduction.upper()}_{dim}D"
        _res = {}
        for dataset in DATASETS:
            _dpath = os.path.join(_dir, dataset)
            if not os.path.exists(_dpath):
                continue
            for _fn in os.listdir(_dpath):
                if _fn.endswith(".json") and "db_only" not in _fn:
                    with open(os.path.join(_dpath, _fn)) as _f:
                        _res[_fn] = json.load(_f)
        out[dim] = (_key, _res)
    return out

def _compute_dim_means(dim_results, dims, alg, metric):
    xs, ys = [], []
    for dim in dims:
        dim_key, dim_res = dim_results[dim]
        scores = []
        for dataset in DATASETS:
            fname = f"{dataset}_dinov3_vitb16.json"
            r = dim_res.get(fname, {})
            if dim_key in r and alg in r[dim_key]:
                scores.append(r[dim_key][alg][metric])
        if scores:
            xs.append(dim)
            ys.append(np.mean(scores))
    return xs, ys

# Pre-load all results for both reductions
_all_dim_results = {red: _load_dim_results(red, _DIM_PLOT_DIMS) for red, _ in _dim_reduction_configs}

# Compute shared y-limits per (metric, alg) across both reductions
_dim_ylims = {}
for metric in ["v_measure", "ari", "nmi"]:
    _dim_ylims[metric] = {}
    for alg in _DIM_ALGS:
        all_vals = []
        for red, _ in _dim_reduction_configs:
            _, ys = _compute_dim_means(_all_dim_results[red], _DIM_PLOT_DIMS, alg, metric)
            all_vals.extend(ys)
        if all_vals:
            pad = (max(all_vals) - min(all_vals)) * 0.08 or 0.02
            _dim_ylims[metric][alg] = (min(all_vals) - pad, max(all_vals) + pad)

for red, red_label in _dim_reduction_configs:
    dim_results = _all_dim_results[red]
    for metric in ["v_measure", "ari", "nmi"]:
        fig, ax1 = plt.subplots(figsize=(5, 3))
        ax2 = ax1.twinx()
        axes = {"HDBSCAN": ax1, "FINCH": ax2}

        lines, labels = [], []
        for alg in _DIM_ALGS:
            ax = axes[alg]
            xs, ys = _compute_dim_means(dim_results, _DIM_PLOT_DIMS, alg, metric)
            if xs:
                (line,) = ax.plot(xs, ys, marker="o", label=alg, color=_dim_colors[alg])
                lines.append(line)
                labels.append(alg)
            # if alg in _dim_ylims[metric]:
            #     ax.set_ylim(_dim_ylims[metric][alg])

        ax1.set_xscale("log", base=2)
        ax1.set_xticks(_DIM_PLOT_DIMS)
        ax1.set_xticklabels([str(d) for d in _DIM_PLOT_DIMS])
        ax1.set_xlabel(f"{red_label} Dimensions")
        ax1.set_ylabel("HDBSCAN -- " + metric.replace("_", "-"), color=_dim_colors["HDBSCAN"])
        ax2.set_ylabel("FINCH -- " + metric.replace("_", "-"), color=_dim_colors["FINCH"])
        ax1.tick_params(axis="y", labelcolor=_dim_colors["HDBSCAN"])
        ax2.tick_params(axis="y", labelcolor=_dim_colors["FINCH"])
        ax1.set_title(f"HDBSCAN & FINCH -- {red_label} Dims (Dataset Mean)")
        ax1.legend(lines, labels, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{red}_dim_{metric}_mean.png"), dpi=300, bbox_inches="tight")
        plt.close()
