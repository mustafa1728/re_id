"""Evaluation metrics (Sec 4.2) and plots from active_selection.py's saved results.

Reads <results_root>/<experiment>/<dataset>/{config,results,candidates}.json and writes to --plots_root:

    tables/metrics_at_<T>.{csv,md}           one row per method, averaged over the datasets every
    tables/metrics_at_<T>_per_dataset.csv    method has results for: fractional regret (Eq. 12) of
                                             the selected candidate ("top1") and of the mean of the
                                             top-k ("topk") for each --table_metrics, the regret AUC
                                             over queries 0..T, EK (Eq. 13), the SROCC between the
                                             belief and the metric, and the LLM oracle's accuracy
    lines/<metric>.png                       metric of the selected candidate vs number of queries,
    lines/<metric>_avg.png                   one panel per dataset / averaged over datasets, with the
    lines/regret_<metric>_avg.png            no-query baselines as horizontal lines
    violins/<metric>_by_{dataset,algorithm,feature}.png
                                             distribution of the metric over the candidates

Methods: every experiment at T queries; "prior only: <prior>" (an experiment's t = 0 record); and the
no-query baselines computed from candidates.json: the median candidate and the best candidate by
silhouette, DBI and CHI.

    python plot_saved_results.py --experiments "prior_*_select_entropy_eps_0.1" --eval_at 10,50
"""
import os
import re
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.args import parse_args
from src.io_utils import read_json
from src.metrics import METRIC_LABELS, fractional_regret, regret_auc

# Validated categorical palette (dataviz reference palette), assigned in this fixed order
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# No-query baselines: recessive gray lines told apart by dash style (and the legend)
BASELINE_STYLES = {
    "Oracle best": dict(color=INK, ls="-", lw=1.2),
    "Median candidate": dict(color=INK_SECONDARY, ls="--", lw=1.2),
    "Silhouette": dict(color=INK_SECONDARY, ls=":", lw=1.5),
    "DBI": dict(color=INK_SECONDARY, ls="-.", lw=1.2),
    "CHI": dict(color=INK_SECONDARY, ls=(0, (6, 2, 1, 2, 1, 2)), lw=1.2),
}
UNSUP_CRITERIA = {"Silhouette": ("silhouette", 1), "DBI": ("dbi", -1), "CHI": ("calinski_harabasz", 1)}

plt.rcParams.update({
    "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 9,
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def experiment_label(name):
    """prior_pairwise_select_entropy_eps_0.1_oracle_qwen -> "pairwise / entropy / ε=0.1 / oracle_qwen"."""
    m = re.match(r"prior_(.+)_select_(.+?)_eps_([^_]+)_?(.*)", name)
    return name if m is None else " / ".join(p for p in (m[1], m[2], f"ε={m[3]}", m[4]) if p)


def load(args):
    """runs[experiment][dataset] = results.json (+ its config); candidates[dataset] = candidates.json."""
    exp_dirs = sorted({d for pattern in args.experiments for d in glob(os.path.join(args.results_root, pattern))
                       if os.path.isdir(d)})
    runs, candidates = defaultdict(dict), {}
    for exp_dir in exp_dirs:
        for path in sorted(glob(os.path.join(exp_dir, "*", "results.json"))):
            ds_dir = os.path.dirname(path)
            dataset = os.path.basename(ds_dir)
            if args.datasets != ["all"] and dataset not in args.datasets:
                continue
            results = read_json(path)
            results["config"] = read_json(os.path.join(ds_dir, "config.json"))
            runs[os.path.basename(exp_dir)][dataset] = results
            if dataset not in candidates:
                candidates[dataset] = read_json(os.path.join(ds_dir, "candidates.json"))
    if not runs:
        raise SystemExit(f"No results matching {args.experiments} under {args.results_root}")
    print(f"{len(runs)} experiment(s), {len(candidates)} dataset(s)")
    return dict(runs), candidates


def curve(results, key, metric, T):
    """results["queries"][t][key][metric] for t = 0..T; a run that stopped early keeps its last value
    (with no new labels the selection does not change)."""
    values = [q[key][metric] for q in results["queries"][:T + 1]]
    return np.array(values + values[-1:] * (T + 1 - len(values)), dtype=float)


# ---------------------------------------------------------------------------
# No-query baselines, from candidates.json
# ---------------------------------------------------------------------------

def candidate_arrays(cands):
    rows = cands["candidates"]
    return {k: np.array([np.nan if r[k] is None else r[k] for r in rows], dtype=float)
            for k in rows[0] if not isinstance(rows[0][k], str)}


def baselines(cands, top_k):
    """{baseline: {"top1": {metric: value}, "topk": {metric: value}}} for one dataset."""
    values = candidate_arrays(cands)
    metrics = [m for m in METRIC_LABELS if m in values]
    out = {"Median candidate": {"top1": {m: np.median(values[m]) for m in metrics},
                                "topk": {m: np.median(values[m]) for m in metrics}}}
    for name, (field, sign) in UNSUP_CRITERIA.items():
        score = sign * values[field]
        order = np.argsort(np.where(np.isnan(score), -np.inf, -score), kind="stable")
        out[name] = {"top1": {m: values[m][order[0]] for m in metrics},
                     "topk": {m: values[m][order[:top_k]].mean() for m in metrics}}
    return out


def best_values(cands, top_k):
    values = candidate_arrays(cands)
    best = {m: np.nanmax(values[m]) for m in METRIC_LABELS if m in values}
    top_k_best = {m: np.sort(values[m])[::-1][:top_k].mean() for m in METRIC_LABELS if m in values}
    return best, top_k_best


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def method_row(top1, topk, best, top_k_best, table_metrics, regret_curves=None, srocc=None, oracle_acc=None):
    row = {}
    for m in table_metrics:
        row[f"regret_{m}_top1"] = fractional_regret(top1[m], best[m])
        row[f"regret_{m}_topk"] = fractional_regret(topk[m], top_k_best[m])
    for m in table_metrics:
        row[f"auc_{m}"] = regret_auc(regret_curves[m]) if regret_curves else np.nan
    row["ek"] = top1["ek"]
    for m in table_metrics:
        row[f"srocc_{m}"] = np.nan if srocc is None or srocc.get(m) is None else srocc[m]
    row["oracle_acc"] = np.nan if oracle_acc is None else oracle_acc
    return row


def metrics_table(runs, candidates, T, args):
    """Per-dataset rows for every method at T queries, and their mean over the datasets that every
    experiment has results for."""
    rows = []
    prior_only_done = set()
    for exp, per_dataset in sorted(runs.items()):
        for dataset, res in per_dataset.items():
            best, top_k_best = res["best"], res["top_k_best"]
            q_T = res["queries"][min(T, len(res["queries"]) - 1)]
            regret_curves = {m: [fractional_regret(v, best[m]) for v in curve(res, "metrics", m, T)]
                             for m in args.table_metrics}
            answered = [q["query"] for q in res["queries"][1:T + 1]]
            oracle_acc = (np.mean([a["y"] == a["y_true"] for a in answered])
                          if res["config"]["oracle"] != "ground_truth" and answered else None)
            rows.append({"method": experiment_label(exp), "dataset": dataset, **method_row(
                q_T["metrics"], q_T["top_k_mean"], best, top_k_best, args.table_metrics,
                regret_curves, q_T["srocc"], oracle_acc)})

            prior = experiment_label(exp).split(" / ")[0]
            if (prior, dataset) not in prior_only_done and res["config"]["oracle"] == "ground_truth":
                prior_only_done.add((prior, dataset))
                q0 = res["queries"][0]
                rows.append({"method": f"prior only: {prior}", "dataset": dataset, **method_row(
                    q0["metrics"], q0["top_k_mean"], best, top_k_best, args.table_metrics, srocc=q0["srocc"])})

    for dataset, cands in candidates.items():
        best, top_k_best = best_values(cands, args.top_k)
        for name, sel in baselines(cands, args.top_k).items():
            rows.append({"method": name, "dataset": dataset,
                         **method_row(sel["top1"], sel["topk"], best, top_k_best, args.table_metrics)})

    per_dataset = pd.DataFrame(rows)
    datasets = sorted(set.intersection(*(set(d) for d in runs.values())))
    mean = (per_dataset[per_dataset.dataset.isin(datasets)].groupby("method", sort=False)
            .mean(numeric_only=True))
    return per_dataset, mean, datasets


def to_markdown(df, floatfmt="{:.3f}"):
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = ["–" if pd.isna(v) else floatfmt.format(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def save_tables(runs, candidates, args):
    out_dir = os.path.join(args.plots_root, "tables")
    os.makedirs(out_dir, exist_ok=True)
    for T in args.eval_at:
        per_dataset, mean, datasets = metrics_table(runs, candidates, T, args)
        mean = mean.dropna(axis=1, how="all")
        per_dataset.to_csv(os.path.join(out_dir, f"metrics_at_{T}_per_dataset.csv"), index=False)
        mean.to_csv(os.path.join(out_dir, f"metrics_at_{T}.csv"))
        md = (f"Metrics after {T} queries, mean over {len(datasets)} datasets "
              f"({', '.join(datasets)}); regret: lower is better\n\n{to_markdown(mean)}\n")
        with open(os.path.join(out_dir, f"metrics_at_{T}.md"), "w") as f:
            f.write(md)
        print(md)


# ---------------------------------------------------------------------------
# Line plots
# ---------------------------------------------------------------------------

def ema(values, alpha):
    out = np.array(values, dtype=float)
    for t in range(1, len(out)):
        out[t] = alpha * out[t] + (1 - alpha) * out[t - 1]
    return out


def baseline_values(cands, metric, top_k):
    best, _ = best_values(cands, top_k)
    return {"Oracle best": best[metric],
            **{name: sel["top1"][metric] for name, sel in baselines(cands, top_k).items()}}


def draw_panel(ax, lines, refs, title, ylabel):
    """lines: [(label, color, x, y)], refs: {baseline: value}."""
    for name, value in refs.items():
        if np.isfinite(value):
            ax.axhline(value, label=name, zorder=1, **BASELINE_STYLES[name])
    for label, color, x, y in lines:
        ax.plot(x, y, color=color, lw=2, label=label, zorder=3)
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xlabel("Number of pairwise queries")
    ax.set_ylabel(ylabel)


def save_figure(fig, axes, path):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def experiment_groups(experiments):
    """At most len(SERIES_COLORS) experiments per figure, so hues are never reused within a figure."""
    n = len(SERIES_COLORS)
    groups = [experiments[i:i + n] for i in range(0, len(experiments), n)]
    if len(groups) > 1:
        print(f"{len(experiments)} experiments -> {len(groups)} figures of <= {n} lines each "
              f"(use --experiments to choose what to compare)")
    return groups


def line_plots(runs, candidates, args):
    T = args.max_queries or max(len(r["queries"]) - 1 for per_ds in runs.values() for r in per_ds.values())
    x = np.arange(T + 1)
    smooth = (lambda y: ema(y, args.ema_alpha)) if args.ema_alpha else None
    datasets_common = sorted(set.intersection(*(set(d) for d in runs.values())))

    for g, experiments in enumerate(experiment_groups(sorted(runs))):
        part = f"_part{g + 1}" if len(runs) > len(SERIES_COLORS) else ""
        colors = dict(zip(experiments, SERIES_COLORS))
        for metric in args.metrics:
            label = METRIC_LABELS.get(metric, metric)
            for suffix, transform in [("", None)] + ([(f"_ema{args.ema_alpha:g}", smooth)] if smooth else []):
                tf = transform or (lambda y: y)
                # One panel per dataset
                datasets = sorted({d for e in experiments for d in runs[e]})
                n_cols = min(4, len(datasets))
                n_rows = int(np.ceil(len(datasets) / n_cols))
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.8 * n_rows), squeeze=False)
                axes = axes.ravel()
                for ax, dataset in zip(axes, datasets):
                    lines = [(experiment_label(e), colors[e], x, tf(curve(runs[e][dataset], "metrics", metric, T)))
                             for e in experiments if dataset in runs[e]]
                    draw_panel(ax, lines, baseline_values(candidates[dataset], metric, args.top_k), dataset, label)
                for ax in axes[len(datasets):]:
                    ax.axis("off")
                save_figure(fig, axes, os.path.join(args.plots_root, "lines", f"{metric}{suffix}{part}.png"))

                # Mean over the datasets every experiment has, as the metric and as fractional regret
                for kind in ("value", "regret"):
                    fig, ax = plt.subplots(figsize=(5.5, 3.6))
                    lines, refs = [], defaultdict(list)
                    for e in experiments:
                        ys = []
                        for d in datasets_common:
                            y = curve(runs[e][d], "metrics", metric, T)
                            ys.append(y if kind == "value" else 1 - y / runs[e][d]["best"][metric])
                        lines.append((experiment_label(e), colors[e], x, tf(np.mean(ys, axis=0))))
                    for d in datasets_common:
                        base = baseline_values(candidates[d], metric, args.top_k)
                        for name, v in base.items():
                            refs[name].append(v if kind == "value" else 1 - v / base["Oracle best"])
                    ylabel = label if kind == "value" else f"Fractional regret ({label})"
                    draw_panel(ax, lines, {k: np.mean(v) for k, v in refs.items()},
                               f"Mean over {len(datasets_common)} datasets", ylabel)
                    name = f"{metric}_avg" if kind == "value" else f"regret_{metric}_avg"
                    save_figure(fig, [ax], os.path.join(args.plots_root, "lines", f"{name}{suffix}{part}.png"))


# ---------------------------------------------------------------------------
# Violin plots of the candidates
# ---------------------------------------------------------------------------

def draw_violins(ax, groups, labels):
    """One violin per group, all in one hue (the x axis carries identity); a constant group is drawn as
    a tick since it has no density."""
    for pos, values in enumerate(groups, start=1):
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        if np.ptp(values) == 0:
            ax.hlines(values[0], pos - 0.3, pos + 0.3, color=SERIES_COLORS[0], lw=2)
            continue
        parts = ax.violinplot(values, positions=[pos], widths=0.8, showextrema=False, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(SERIES_COLORS[0])
            body.set_edgecolor("none")
            body.set_alpha(0.55)
        parts["cmedians"].set_color(INK)
        parts["cmedians"].set_linewidth(1.5)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right")
    ax.set_xlim(0.4, len(labels) + 0.6)
    ax.grid(axis="x", visible=False)


def violin_plots(candidates, args):
    table = pd.DataFrame([{**row, "dataset": d} for d, c in candidates.items() for row in c["candidates"]])
    datasets = sorted(candidates)
    for metric in args.metrics:
        label = METRIC_LABELS.get(metric, metric)

        fig, ax = plt.subplots(figsize=(max(5.0, 0.45 * len(datasets) + 2), 3.4))
        draw_violins(ax, [table.loc[table.dataset == d, metric].to_numpy(float) for d in datasets], datasets)
        ax.set_ylabel(label)
        ax.set_title(f"{label} of all candidate clusterings", fontsize=9)
        path = os.path.join(args.plots_root, "violins", f"{metric}_by_dataset.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")

        for by, column in [("algorithm", "family"), ("feature", "feature")]:
            categories = sorted(table[column].unique())
            panels = ["All datasets"] + datasets
            n_cols = min(4, len(panels))
            n_rows = int(np.ceil(len(panels) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(0.4 * len(categories) * n_cols + 2, 3.0 * n_rows),
                                     squeeze=False)
            axes = axes.ravel()
            for ax, panel in zip(axes, panels):
                sub = table if panel == "All datasets" else table[table.dataset == panel]
                draw_violins(ax, [sub.loc[sub[column] == c, metric].to_numpy(float) for c in categories], categories)
                ax.set_title(panel, fontsize=9)
                ax.set_ylabel(label)
            for ax in axes[len(panels):]:
                ax.axis("off")
            fig.suptitle(f"{label} of the candidate clusterings by {by}", fontsize=10)
            fig.tight_layout()
            path = os.path.join(args.plots_root, "violins", f"{metric}_by_{by}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {path}")


def main():
    args = parse_args(__doc__.splitlines()[0])
    runs, candidates = load(args)
    save_tables(runs, candidates, args)
    line_plots(runs, candidates, args)
    violin_plots(candidates, args)


if __name__ == "__main__":
    main()
