import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from study_configs import PRIMARY_MATCHED_PAIR


MODEL_LABELS = {
    "lstm_small_matched": "LSTM (small matched)",
    "transformer_small": "Transformer (small matched)",
    "markov_baseline": "Markov",
    "lstm_large": "LSTM (large)",
    "transformer_large_matched": "Transformer (large matched)",
}
MODEL_COLORS = {
    "lstm_small_matched": "#1f77b4",
    "transformer_small": "#d62728",
    "markov_baseline": "#6b7280",
    "lstm_large": "#1f77b4",
    "transformer_large_matched": "#d62728",
}
FRACTION_ORDER = [0.1, 0.3, 1.0]
FRACTION_LABELS = {0.1: "10%", 0.3: "30%", 1.0: "100%"}


def load_summary_rows(summary_csv: Path) -> List[Dict[str, str]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def _model_color(model_name: str) -> str:
    return MODEL_COLORS.get(model_name, "#4b5563")


def filter_rows(
    rows: Sequence[Dict[str, str]],
    model_names: Iterable[str],
    fractions: Iterable[float] | None = None,
    temperature: float = 0.8,
) -> List[Dict[str, str]]:
    model_names = set(model_names)
    fractions_set = None if fractions is None else {float(value) for value in fractions}
    filtered = []
    for row in rows:
        if row["model_spec"] not in model_names:
            continue
        if fractions_set is not None and float(row["fraction"]) not in fractions_set:
            continue
        if abs(float(row["temperature"]) - temperature) > 1e-9:
            continue
        filtered.append(row)
    return filtered


def _group_metric(rows: Sequence[Dict[str, str]], metric: str) -> Dict[str, List[float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = _to_float(row.get(metric))
        if value is not None:
            grouped[row["model_spec"]].append(value)
    return grouped


def plot_perplexity_bars(rows: Sequence[Dict[str, str]], output_path: Path, title: str) -> None:
    grouped = _group_metric(rows, "test_perplexity")
    labels = list(grouped.keys())
    means = [np.mean(grouped[label]) for label in labels]
    stds = [np.std(grouped[label]) for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [_model_label(label) for label in labels],
        means,
        yerr=stds,
        color=[_model_color(label) for label in labels],
        alpha=0.9,
        capsize=6,
    )
    ax.set_title(title)
    ax.set_ylabel("Test perplexity")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_metric_group(
    rows: Sequence[Dict[str, str]],
    metrics: Sequence[str],
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in metrics:
            value = _to_float(row.get(metric))
            if value is not None:
                grouped[row["model_spec"]][metric].append(value)

    labels = list(grouped.keys())
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, label in enumerate(labels):
        means = [np.mean(grouped[label][metric]) for metric in metrics]
        stds = [np.std(grouped[label][metric]) for metric in metrics]
        ax.bar(
            x + (offset - (len(labels) - 1) / 2) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=_model_label(label),
            color=_model_color(label),
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fraction_metric_lines(
    rows: Sequence[Dict[str, str]],
    model_names: Sequence[str],
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(FRACTION_ORDER))

    for model_name in model_names:
        means = []
        stds = []
        for fraction in FRACTION_ORDER:
            values = [
                float(row[metric])
                for row in rows
                if row["model_spec"] == model_name and abs(float(row["fraction"]) - fraction) < 1e-9
            ]
            if not values:
                means.append(np.nan)
                stds.append(0.0)
                continue
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))

        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker="o",
            linewidth=2.2,
            capsize=4,
            label=_model_label(model_name),
            color=_model_color(model_name),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([FRACTION_LABELS[fraction] for fraction in FRACTION_ORDER])
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fraction_metric_panels(
    rows: Sequence[Dict[str, str]],
    model_names: Sequence[str],
    metrics: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.8), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(FRACTION_ORDER))

    for axis, metric in zip(axes, metrics):
        for model_name in model_names:
            means = []
            stds = []
            for fraction in FRACTION_ORDER:
                values = [
                    float(row[metric])
                    for row in rows
                    if row["model_spec"] == model_name and abs(float(row["fraction"]) - fraction) < 1e-9
                ]
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values)))

            axis.errorbar(
                x,
                means,
                yerr=stds,
                marker="o",
                linewidth=2.0,
                capsize=4,
                label=_model_label(model_name),
                color=_model_color(model_name),
            )

        axis.set_title(metric)
        axis.set_xticks(x)
        axis.set_xticklabels([FRACTION_LABELS[fraction] for fraction in FRACTION_ORDER])
        axis.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Ratio")
    axes[len(axes) // 2].set_xlabel("Training data fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(model_names), frameon=False)
    fig.suptitle(title, y=1.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_histories(
    result_root: Path,
    run_id: str,
    model_names: Sequence[str],
    seeds: Sequence[int],
    fraction_tag: str = "fraction_100",
) -> Dict[str, List[List[Dict[str, float]]]]:
    history_map: Dict[str, List[List[Dict[str, float]]]] = defaultdict(list)
    for model_name in model_names:
        for seed in seeds:
            history_path = result_root / run_id / model_name / f"seed_{seed}" / "split_cbdb_fixed_v1" / fraction_tag / "history.csv"
            if not history_path.exists():
                continue
            with history_path.open("r", encoding="utf-8", newline="") as handle:
                history_map[model_name].append(
                    [
                        {
                            "epoch": int(row["epoch"]),
                            "train_perplexity": float(row["train_perplexity"]),
                            "val_perplexity": float(row["val_perplexity"]),
                        }
                        for row in csv.DictReader(handle)
                    ]
                )
    return history_map


def plot_convergence(
    histories: Dict[str, List[List[Dict[str, float]]]],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for model_name, runs in histories.items():
        if not runs:
            continue
        max_epoch = max(len(run) for run in runs)
        train_means = []
        val_means = []

        for epoch_index in range(max_epoch):
            train_values = [run[epoch_index]["train_perplexity"] for run in runs if epoch_index < len(run)]
            val_values = [run[epoch_index]["val_perplexity"] for run in runs if epoch_index < len(run)]
            train_means.append(np.mean(train_values))
            val_means.append(np.mean(val_values))

        epochs = np.arange(1, len(train_means) + 1)
        axes[0].plot(epochs, train_means, label=_model_label(model_name), color=_model_color(model_name))
        axes[1].plot(epochs, val_means, label=_model_label(model_name), color=_model_color(model_name))

    axes[0].set_title(f"{title}: train")
    axes[1].set_title(f"{title}: validation")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Perplexity")
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_human_eval(human_eval_csv: Path, key_csv: Path, output_path: Path) -> bool:
    with human_eval_csv.open("r", encoding="utf-8", newline="") as handle:
        scored_rows = [row for row in csv.DictReader(handle) if row["naturalness"].strip()]

    if not scored_rows:
        return False

    with key_csv.open("r", encoding="utf-8", newline="") as handle:
        key_map = {row["candidate_id"]: row["model_spec"] for row in csv.DictReader(handle)}

    metrics = ["naturalness", "realism", "phonetic_flow", "novelty_balance"]
    grouped = defaultdict(lambda: defaultdict(list))
    for row in scored_rows:
        model_name = key_map[row["candidate_id"]]
        for metric in metrics:
            grouped[model_name][metric].append(float(row[metric]))

    labels = list(grouped.keys())
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, label in enumerate(labels):
        means = [np.mean(grouped[label][metric]) for metric in metrics]
        ax.bar(
            x + (offset - (len(labels) - 1) / 2) * width,
            means,
            width,
            label=_model_label(label),
            color=_model_color(label),
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Average rating")
    ax.set_title("Human evaluation")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot study artifacts for the fairness-first experiment.")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--run-id", default="fairness_v1")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--seeds", default="13,37,73")
    args = parser.parse_args()

    result_root = Path(args.output_root)
    summary_path = Path(args.summary_path) if args.summary_path else result_root / args.run_id / "summary" / "summary.csv"
    rows = load_summary_rows(summary_path)
    figure_dir = result_root / args.run_id / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    primary_models = list(PRIMARY_MATCHED_PAIR)
    full_data_rows = filter_rows(rows, primary_models, fractions=[1.0], temperature=0.8)
    data_scale_rows = filter_rows(rows, primary_models, fractions=FRACTION_ORDER, temperature=0.8)

    plot_perplexity_bars(
        full_data_rows,
        figure_dir / "stage1_primary_test_perplexity.png",
        "Stage 1: primary matched pair at full data",
    )
    plot_metric_group(
        full_data_rows,
        metrics=["unique_name_ratio", "distinct_2", "novelty_ratio", "train_overlap_ratio"],
        output_path=figure_dir / "stage1_distribution_metrics.png",
        title="Stage 1: full-data generation metrics",
        ylabel="Ratio",
    )
    plot_metric_group(
        full_data_rows,
        metrics=["consecutive_duplicate_ratio", "any_duplicate_ratio", "invalid_output_ratio", "compound_surname_compatibility"],
        output_path=figure_dir / "stage1_constraint_metrics.png",
        title="Stage 1: full-data constraint metrics",
        ylabel="Ratio",
    )

    histories = load_histories(
        result_root=result_root,
        run_id=args.run_id,
        model_names=primary_models,
        seeds=seeds,
        fraction_tag="fraction_100",
    )
    plot_convergence(
        histories=histories,
        output_path=figure_dir / "stage1_primary_convergence.png",
        title="Stage 1 full-data convergence",
    )

    plot_fraction_metric_lines(
        data_scale_rows,
        model_names=primary_models,
        metric="test_perplexity",
        output_path=figure_dir / "stage2_data_scale_perplexity.png",
        title="Stage 2: test perplexity across data scales",
        ylabel="Test perplexity",
    )
    plot_fraction_metric_panels(
        data_scale_rows,
        model_names=primary_models,
        metrics=["distinct_2", "novelty_ratio", "train_overlap_ratio"],
        output_path=figure_dir / "stage2_data_scale_distribution.png",
        title="Stage 2: distribution changes across data scales",
    )
    plot_fraction_metric_panels(
        data_scale_rows,
        model_names=primary_models,
        metrics=["unique_name_ratio", "any_duplicate_ratio", "invalid_output_ratio"],
        output_path=figure_dir / "stage2_data_scale_quality.png",
        title="Stage 2: quality and validity across data scales",
    )

    human_eval_dir = result_root / args.run_id / "human_eval"
    human_eval_csv = human_eval_dir / "human_eval.csv"
    human_eval_key = human_eval_dir / "human_eval_key.csv"
    has_human_eval = False
    if human_eval_csv.exists() and human_eval_key.exists():
        has_human_eval = plot_human_eval(human_eval_csv, human_eval_key, figure_dir / "human_eval.png")

    manifest = {
        "summary_path": str(summary_path),
        "figure_dir": str(figure_dir),
        "primary_pair": primary_models,
        "generated_figures": [
            "stage1_primary_test_perplexity.png",
            "stage1_distribution_metrics.png",
            "stage1_constraint_metrics.png",
            "stage1_primary_convergence.png",
            "stage2_data_scale_perplexity.png",
            "stage2_data_scale_distribution.png",
            "stage2_data_scale_quality.png",
        ],
        "has_human_eval": has_human_eval,
    }
    (figure_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Figures written to {figure_dir}")


if __name__ == "__main__":
    main()
