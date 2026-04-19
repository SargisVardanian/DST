from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np


BRIEF_DISPLAY_ORDER = [
    "RF",
    "FOIL:dsgd_dempster",
    "RIPPER:dsgd_dempster",
    "FOIL:native_ordered_rule",
    "FOIL:weighted_vote",
    "FOIL:first_hit_laplace",
    "RIPPER:native_ordered_rule",
    "RIPPER:weighted_vote",
    "RIPPER:first_hit_laplace",
]
BEST_METHOD_TIEBREAK_ORDER = [
    "RF",
    "RIPPER:native_ordered_rule",
    "RIPPER:first_hit_laplace",
    "RIPPER:weighted_vote",
    "RIPPER:dsgd_dempster",
    "FOIL:native_ordered_rule",
    "FOIL:first_hit_laplace",
    "FOIL:weighted_vote",
    "FOIL:dsgd_dempster",
]
LOWER_IS_BETTER = {"nll", "ece", "unc_mean", "unc_comb"}


def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | float | int | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def canonical_method(algo: str | None, method: str | None) -> str:
    algo_u = str(algo or "").strip().upper()
    method_u = str(method or "").strip().lower().replace("-", "_")
    if method_u == "rf" or algo_u == "RF":
        return "RF"
    method_map = {
        "dempster": "dsgd_dempster",
        "dsgd_dempster": "dsgd_dempster",
        "vote": "weighted_vote",
        "weighted_vote": "weighted_vote",
        "native_ordered_rule": "native_ordered_rule",
        "ordered_rule": "native_ordered_rule",
        "ordered": "native_ordered_rule",
        "first_hit": "first_hit_laplace",
        "first_hit_laplace": "first_hit_laplace",
    }
    suffix = method_map.get(method_u, method_u)
    return suffix if not algo_u else f"{algo_u}:{suffix}"


def normalize_row(row: dict[str, str]) -> dict[str, object] | None:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    dataset = str(lowered.get("dataset", "")).strip()
    if not dataset:
        return None
    if "algo" in lowered and "method" in lowered:
        method = canonical_method(lowered.get("algo"), lowered.get("method"))
    else:
        method = str(lowered.get("method", "")).strip()
        if not method:
            return None
    return {
        "dataset": dataset,
        "method": method,
        "seed": to_float(lowered.get("seed")),
        "n_runs": to_float(lowered.get("n_runs")),
        "n_seeds": to_float(lowered.get("n_seeds")),
        "acc": to_float(lowered.get("acc")),
        "std_acc": to_float(lowered.get("std_acc")),
        "macro_f1": to_float(lowered.get("macro_f1", lowered.get("f1"))),
        "std_macro_f1": to_float(lowered.get("std_macro_f1", lowered.get("std_f1"))),
        "nll": to_float(lowered.get("nll")),
        "std_nll": to_float(lowered.get("std_nll")),
        "ece": to_float(lowered.get("ece")),
        "std_ece": to_float(lowered.get("std_ece")),
        "unc_mean": to_float(lowered.get("unc_mean")),
        "std_unc_mean": to_float(lowered.get("std_unc_mean")),
        "unc_comb": to_float(lowered.get("unc_comb", lowered.get("omega"))),
        "std_unc_comb": to_float(lowered.get("std_unc_comb")),
    }


def mean_rank(per_dataset: dict[str, list[tuple[str, float]]], *, higher_is_better: bool) -> dict[str, float]:
    ranks: dict[str, list[float]] = defaultdict(list)
    for dataset_rows in per_dataset.values():
        ordered = sorted(dataset_rows, key=lambda item: item[1], reverse=higher_is_better)
        i = 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and abs(ordered[j][1] - ordered[i][1]) <= 1e-12:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for method, _ in ordered[i:j]:
                ranks[method].append(avg_rank)
            i = j
    return {method: mean(values) for method, values in ranks.items() if values}


def _brief_method_sort_key(method: str) -> tuple[int, str]:
    if method in BRIEF_DISPLAY_ORDER:
        return (BRIEF_DISPLAY_ORDER.index(method), method)
    return (len(BRIEF_DISPLAY_ORDER), method)


def _best_method_sort_key(method: str) -> tuple[int, str]:
    if method in BEST_METHOD_TIEBREAK_ORDER:
        return (BEST_METHOD_TIEBREAK_ORDER.index(method), method)
    return (len(BEST_METHOD_TIEBREAK_ORDER), method)


def _pick_best(rows: list[dict[str, object]], metric: str) -> dict[str, object] | None:
    candidates = [row for row in rows if row.get(metric) is not None]
    if not candidates:
        return None
    if metric in LOWER_IS_BETTER:
        ordered = sorted(
            candidates,
            key=lambda item: (float(item[metric]),) + _best_method_sort_key(str(item["method"])),
        )
    else:
        ordered = sorted(
            candidates,
            key=lambda item: (-float(item[metric]),) + _best_method_sort_key(str(item["method"])),
        )
    return ordered[0]


def build_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    normalized_rows = [item for item in (normalize_row(row) for row in rows) if item is not None]
    by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_method_std: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_dataset_acc: dict[str, list[tuple[str, float]]] = defaultdict(list)
    by_dataset_f1: dict[str, list[tuple[str, float]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    seeds_seen: set[int] = set()
    total_runs = 0
    method_support: dict[str, dict[str, int]] = defaultdict(lambda: {"dataset_rows": 0, "n_runs": 0, "max_n_seeds": 0})
    for row in normalized_rows:
        method = str(row["method"])
        dataset = str(row["dataset"])
        by_dataset[dataset].append(row)
        method_support[method]["dataset_rows"] += 1
        seed = row.get("seed")
        if seed is not None:
            seeds_seen.add(int(seed))
        row_runs = int(row.get("n_runs")) if row.get("n_runs") is not None else 1
        row_seeds = int(row.get("n_seeds")) if row.get("n_seeds") is not None else (1 if seed is not None else 0)
        total_runs += row_runs
        method_support[method]["n_runs"] += row_runs
        method_support[method]["max_n_seeds"] = max(method_support[method]["max_n_seeds"], row_seeds)
        for col in ("acc", "macro_f1", "nll", "ece", "unc_mean", "unc_comb"):
            value = row.get(col)
            if value is not None:
                by_method[method][col].append(float(value))
            std_value = row.get(f"std_{col}")
            if std_value is not None:
                by_method_std[method][col].append(float(std_value))
        if row.get("acc") is not None:
            by_dataset_acc[dataset].append((method, float(row["acc"])))
        if row.get("macro_f1") is not None:
            by_dataset_f1[dataset].append((method, float(row["macro_f1"])))
    mean_metrics = {method: {metric: mean(values) for metric, values in metric_map.items() if values} for method, metric_map in by_method.items()}
    mean_std_metrics = {method: {metric: mean(values) for metric, values in metric_map.items() if values} for method, metric_map in by_method_std.items()}
    methods_present = sorted(mean_metrics.keys(), key=_brief_method_sort_key)
    metric_bests: dict[str, set[str]] = {metric: set() for metric in ("acc", "macro_f1", "nll", "ece", "unc_mean", "unc_comb")}
    for metric in metric_bests:
        values = {method: metrics.get(metric) for method, metrics in mean_metrics.items() if metrics.get(metric) is not None}
        if not values:
            continue
        best_value = min(values.values()) if metric in LOWER_IS_BETTER else max(values.values())
        metric_bests[metric] = {method for method, value in values.items() if value is not None and abs(value - best_value) <= 1e-12}
    dataset_summary = []
    for dataset in sorted(by_dataset):
        rows_for_dataset = by_dataset[dataset]
        rf_row = next((row for row in rows_for_dataset if row["method"] == "RF"), None)
        rule_rows = [row for row in rows_for_dataset if row["method"] != "RF"]
        dataset_summary.append({"dataset": dataset, "rf": rf_row, "best_f1": _pick_best(rule_rows, "macro_f1"), "best_acc": _pick_best(rule_rows, "acc")})
    return {
        "datasets": sorted(by_dataset),
        "mean_metrics": mean_metrics,
        "mean_std_metrics": mean_std_metrics,
        "metric_bests": metric_bests,
        "acc_ranks": mean_rank(by_dataset_acc, higher_is_better=True),
        "f1_ranks": mean_rank(by_dataset_f1, higher_is_better=True),
        "dataset_summary": dataset_summary,
        "method_support": dict(method_support),
        "n_rows": len(normalized_rows),
        "n_total_runs": total_runs,
        "n_unique_seeds": len(seeds_seen),
        "n_methods": len(mean_metrics),
        "methods_present": methods_present,
    }


def _maybe_bold(method: str, metric: str, text: str, summary: dict[str, object]) -> str:
    return f"**{text}**" if method in set(summary["metric_bests"].get(metric, set())) else text


def method_row(summary: dict[str, object], method: str) -> str:
    metrics = summary["mean_metrics"].get(method, {})
    std_metrics = summary.get("mean_std_metrics", {}).get(method, {})
    support = summary.get("method_support", {}).get(method, {})

    def fmt_pm(metric: str) -> str:
        mean_val = metrics.get(metric)
        std_val = std_metrics.get(metric)
        if mean_val is None:
            return "—"
        if std_val is None or not np.isfinite(float(std_val)) or float(std_val) <= 1e-12:
            return f"{float(mean_val):.4f}"
        return f"{float(mean_val):.4f} ± {float(std_val):.4f}"

    return (
        f"| {method} | {_maybe_bold(method, 'acc', fmt_pm('acc'), summary)} | "
        f"{_maybe_bold(method, 'macro_f1', fmt_pm('macro_f1'), summary)} | "
        f"{_maybe_bold(method, 'nll', fmt_pm('nll'), summary)} | "
        f"{_maybe_bold(method, 'ece', fmt_pm('ece'), summary)} | "
        f"{_maybe_bold(method, 'unc_mean', fmt_pm('unc_mean'), summary)} | "
        f"{_maybe_bold(method, 'unc_comb', fmt_pm('unc_comb'), summary)} | "
        f"{fmt(summary['f1_ranks'].get(method), digits=3)} | {fmt(summary['acc_ranks'].get(method), digits=3)} | "
        f"{int(support.get('n_runs', 0))} / {int(support.get('max_n_seeds', 0))} |"
    )


def dataset_table_rows(summary: dict[str, object]) -> list[str]:
    lines = ["| Dataset | Best rule-based (Macro-F1) | Rule Macro-F1 | RF Macro-F1 | Best rule-based (Acc) | Rule Acc | RF Acc |", "|---|---|---:|---:|---|---:|---:|"]
    dataset_items = list(summary["dataset_summary"])
    max_rule_f1 = max((float((item.get("best_f1") or {}).get("macro_f1")) for item in dataset_items if (item.get("best_f1") or {}).get("macro_f1") is not None), default=None)
    max_rf_f1 = max((float((item.get("rf") or {}).get("macro_f1")) for item in dataset_items if (item.get("rf") or {}).get("macro_f1") is not None), default=None)
    max_rule_acc = max((float((item.get("best_acc") or {}).get("acc")) for item in dataset_items if (item.get("best_acc") or {}).get("acc") is not None), default=None)
    max_rf_acc = max((float((item.get("rf") or {}).get("acc")) for item in dataset_items if (item.get("rf") or {}).get("acc") is not None), default=None)
    for item in dataset_items:
        best_f1 = item.get("best_f1") or {}
        best_acc = item.get("best_acc") or {}
        rf = item.get("rf") or {}
        rule_f1 = best_f1.get("macro_f1")
        rf_f1 = rf.get("macro_f1")
        rule_acc = best_acc.get("acc")
        rf_acc = rf.get("acc")
        rule_f1_text = fmt(rule_f1)
        rf_f1_text = fmt(rf_f1)
        rule_acc_text = fmt(rule_acc)
        rf_acc_text = fmt(rf_acc)
        if rule_f1 is not None and max_rule_f1 is not None and float(rule_f1) == max_rule_f1:
            rule_f1_text = f"**{rule_f1_text}**"
        if rf_f1 is not None and max_rf_f1 is not None and float(rf_f1) == max_rf_f1:
            rf_f1_text = f"**{rf_f1_text}**"
        if rule_acc is not None and max_rule_acc is not None and float(rule_acc) == max_rule_acc:
            rule_acc_text = f"**{rule_acc_text}**"
        if rf_acc is not None and max_rf_acc is not None and float(rf_acc) == max_rf_acc:
            rf_acc_text = f"**{rf_acc_text}**"
        lines.append(
            f"| {item['dataset']} | {str(best_f1.get('method', '—'))} | {rule_f1_text} | {rf_f1_text} | "
            f"{str(best_acc.get('method', '—'))} | {rule_acc_text} | {rf_acc_text} |"
        )
    return lines


def render_report(summary: dict[str, object], metrics_path: Path, hard_cases_path: Path) -> str:
    datasets = list(summary["datasets"])
    methods_present = list(summary["methods_present"])
    rows = [
        "# DSGD-Auto Brief Project Report",
        "",
        "## Abstract",
        "This report summarizes the current frozen-rule DSGD pipeline: induce readable rules with custom FOIL- and RIPPER-style rule generators, freeze the resulting ruleset, and then learn evidential masses on those fixed rules. The present snapshot aggregates one fixed benchmark split with five DSGD training seeds, so it is more informative than a one-run anecdote but still not a split-robust generalization claim.",
        "",
        "## What Was Built",
        "- `build_report.py` is the canonical public entry point for the full benchmark/report pipeline.",
        "- `train_test_runner.py` is the shared split/train/test engine.",
        "- `analyze_hard_cases.py` remains the hard-case analysis entry point.",
        "- FOIL and RIPPER here are custom repository-specific induced-and-shaped rule pipelines.",
        f"- Current benchmark snapshot: `{metrics_path}`",
        "",
        "Baseline definitions used below:",
        "- `native_ordered_rule`: use the first fired rule in the original emission order produced by the inducer and predict its class label directly.",
        "- `first_hit_laplace`: use the first fired rule in the ordered list and read its Laplace-smoothed class estimate.",
        "- `weighted_vote`: aggregate all fired rules with heuristic weights, without learned evidential masses.",
        "- `frozen ruleset`: rule induction is finished first, and the later mass-learning stage does not add, remove, or rewrite rules.",
        "- `hard case`: a held-out test sample that satisfies at least one conflict-oriented criterion computed from fired rules of that same frozen ruleset.",
        "",
        "## Results",
        f"The current snapshot contains {summary['n_rows']} aggregated rows built from {summary['n_total_runs']} benchmark runs across {len(datasets)} datasets and {summary['n_methods']} methods. The benchmark averages are computed over five training seeds on one fixed train/test split per dataset. In this fixed-split multi-seed snapshot, the learned Dempster rows usually rank near the top of the rule-based methods evaluated on the same frozen rulesets, while Random Forest remains the strongest overall external reference. That ordering should still be read as benchmark description, not as evidence for a broader mechanism or split-robust dominance.",
        "",
        "| Method | Acc | Macro-F1 | NLL | ECE | unc_mean | unc_comb | Mean rank F1 | Mean rank Acc | Runs / Seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows.extend(method_row(summary, method) for method in methods_present)
    rows.extend(
        [
            "",
            "Reading notes:",
            "- Bold values mark the best average value in each metric column among the methods present in this snapshot.",
            "- These averages are useful as a compact benchmark summary, but they should be read together with the dataset-level table below.",
            "- The current report summarizes five-seed averages on one fixed split per dataset; it does not replace broader split-robust evaluation.",
            "",
            "## Dataset-Level Snapshot",
            "Best rule-based result versus RF on the main report metrics:",
            *dataset_table_rows(summary),
            "",
            "## Hard-Case Note",
            f"The separate hard-case analysis in `{hard_cases_path}` is best read as diagnostic evidence on the same frozen rulesets for the same dataset/inducer pairs. It often shows learned Dempster fusion outperforming raw voting on conflict-heavy samples in this snapshot, but those tables do not by themselves establish a broad causal explanation.",
            "",
            "## Interpretation",
            "The safest interpretation of the current snapshot is narrow: on this fixed-split five-seed frozen-rule benchmark, learned evidential masses are often competitive with or better than the raw rule aggregators built from the same ruleset. The examples and hard-case tables are illustrative diagnostics, not proof of a general mechanism. The inspection examples support only a readability claim about the displayed rule objects, not a broader comparative interpretability result. The report should therefore frame the contribution as a benchmarked frozen-rule aggregation pipeline, not as proof of a rule-diversity, overlap, fired-rule-count, or interpretability mechanism.",
            "",
            "## Reproducibility",
            f"- Current benchmark snapshot: `{metrics_path}`",
            f"- Hard-case analysis: `{hard_cases_path}`",
            "- Computational cost summary: `src/results/COMPUTATIONAL_COST.md`",
            "- Pool-shaping ablation status: `src/results/POOL_SHAPING_ABLATION.md`",
            "- Standard custom run: `python3 train_test_runner.py --dataset-path ./adult.csv --inducers RIPPER,FOIL --save-root ./tmp_run --seeds 7,8 --test-size 0.25`",
            "- Frozen paper protocol: `python3 train_test_runner.py --dataset-path ./adult.csv --inducers RIPPER,FOIL --save-root ./tmp_run --paper-mode`",
            "- Regenerate the full pipeline: `python3 build_report.py --out-root src/results/raw_runs --results-dir src/results`",
            "",
            "## References",
            "- Dempster, 1968. [Upper and lower probabilities induced by a multivalued mapping](https://doi.org/10.1214/aoms/1177698950).",
            "- Shafer, 1976. [A Mathematical Theory of Evidence](https://press.princeton.edu/books/hardcover/9780691214696/a-mathematical-theory-of-evidence).",
            "- Quinlan, 1990. [Learning Logical Definitions from Relations](https://doi.org/10.1007/BF00117105).",
            "- Cohen, 1995. [Fast Effective Rule Induction](https://doi.org/10.1016/B978-1-55860-377-6.50023-2).",
            "- Sergio Peñafiel, Nelson Baloian, Hernan Sanson, and Juan Antonio Pino, 2020. [Applying Dempster-Shafer theory for developing a flexible, accurate and interpretable classifier](https://doi.org/10.1016/j.eswa.2020.113262).",
            "- Aik Tarkhanyan and Ashot Harutyunyan, 2025. [DSGD++: Performance and robustness improvements for the Dempster-Shafer Gradient Descent classifier](https://arxiv.org/abs/2507.00453).",
        ]
    )
    return "\n".join(rows) + "\n"


def write_brief_report(*, metrics_path: Path, hard_cases_path: Path, out_path: Path) -> int:
    summary = build_summary(read_rows(metrics_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(summary, metrics_path, hard_cases_path), encoding="utf-8")
    print(f"wrote: {out_path}")
    return 0
