from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


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
RAW_METHOD_NAMES = {
    "dsgd_dempster",
    "native_ordered_rule",
    "weighted_vote",
    "first_hit_laplace",
}
REPOSITORY_URL = "https://github.com/SargisVardanian/DST"
DISPLAY_NAME = {
    "RF": "RF",
    "FOIL:dsgd_dempster": "FOIL + learned Dempster fusion",
    "RIPPER:dsgd_dempster": "RIPPER + learned Dempster fusion",
    "FOIL:native_ordered_rule": "FOIL ordered-rule baseline",
    "RIPPER:native_ordered_rule": "RIPPER ordered-rule baseline",
    "FOIL:weighted_vote": "FOIL weighted rule-vote baseline",
    "RIPPER:weighted_vote": "RIPPER weighted rule-vote baseline",
    "FOIL:first_hit_laplace": "FOIL first-hit fixed-confidence baseline",
    "RIPPER:first_hit_laplace": "RIPPER first-hit fixed-confidence baseline",
}


def display_method(method: str | None) -> str:
    text = str(method or "").strip()
    return DISPLAY_NAME.get(text, text or "—")


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
        "rule_vote": "weighted_vote",
        "majority_rule_vote": "weighted_vote",
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
    dsgd_best_counts = {"acc": 0, "macro_f1": 0}
    rf_win_tie_counts = {"acc": 0, "macro_f1": 0}
    dsgd_vs_vote_wins = {"acc": defaultdict(int), "macro_f1": defaultdict(int)}
    dsgd_vs_vote_total = {"acc": defaultdict(int), "macro_f1": defaultdict(int)}
    for dataset in sorted(by_dataset):
        rows_for_dataset = by_dataset[dataset]
        rf_row = next((row for row in rows_for_dataset if row["method"] == "RF"), None)
        rule_rows = [row for row in rows_for_dataset if row["method"] != "RF"]
        dataset_summary.append({"dataset": dataset, "rf": rf_row, "best_f1": _pick_best(rule_rows, "macro_f1"), "best_acc": _pick_best(rule_rows, "acc")})
        for metric in ("acc", "macro_f1"):
            metric_rows = [row for row in rule_rows if row.get(metric) is not None]
            if metric_rows:
                best_rule_value = max(float(row[metric]) for row in metric_rows)
                if any(str(row["method"]).endswith(":dsgd_dempster") and abs(float(row[metric]) - best_rule_value) <= 1e-12 for row in metric_rows):
                    dsgd_best_counts[metric] += 1
                if rf_row is not None and rf_row.get(metric) is not None and float(rf_row[metric]) >= best_rule_value - 1e-12:
                    rf_win_tie_counts[metric] += 1
            for algo in ("FOIL", "RIPPER"):
                dem = next((row for row in rows_for_dataset if row["method"] == f"{algo}:dsgd_dempster"), None)
                vote = next((row for row in rows_for_dataset if row["method"] == f"{algo}:weighted_vote"), None)
                if dem is not None and vote is not None and dem.get(metric) is not None and vote.get(metric) is not None:
                    dsgd_vs_vote_total[metric][algo] += 1
                    if float(dem[metric]) > float(vote[metric]) + 1e-12:
                        dsgd_vs_vote_wins[metric][algo] += 1
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
        "dsgd_best_counts": dsgd_best_counts,
        "rf_win_tie_counts": rf_win_tie_counts,
        "dsgd_vs_vote_wins": {metric: dict(values) for metric, values in dsgd_vs_vote_wins.items()},
        "dsgd_vs_vote_total": {metric: dict(values) for metric, values in dsgd_vs_vote_total.items()},
    }


def read_hard_case_summary(path: Path) -> dict[str, object] | None:
    overview = path.parent / "hard_case_overview.csv"
    method_summary = path.parent / "hard_case_method_summary.csv"
    if not overview.exists() or not method_summary.exists():
        return None
    overview_rows = read_rows(overview)
    method_rows = read_rows(method_summary)
    total_hard = sum(int(float(row.get("n_hard_cases") or 0)) for row in overview_rows)
    keyed = {(row.get("dataset"), row.get("algo"), row.get("method")): row for row in method_rows}
    pairs = sorted({(row.get("dataset"), row.get("algo")) for row in method_rows})
    out = {"n_pairs": len(overview_rows), "total_hard": total_hard}
    for metric in ("acc", "macro_f1"):
        wins = ties = count = 0
        deltas = []
        for dataset, algo in pairs:
            dem = keyed.get((dataset, algo, "dsgd_dempster"))
            vote = keyed.get((dataset, algo, "weighted_vote"))
            if dem is None or vote is None:
                continue
            d_val = to_float(dem.get(metric))
            v_val = to_float(vote.get(metric))
            if d_val is None or v_val is None:
                continue
            count += 1
            delta = float(d_val) - float(v_val)
            deltas.append(delta)
            if delta > 1e-12:
                wins += 1
            elif abs(delta) <= 1e-12:
                ties += 1
        out[f"{metric}_wins"] = wins
        out[f"{metric}_ties"] = ties
        out[f"{metric}_n"] = count
        out[f"{metric}_mean_delta"] = mean(deltas) if deltas else None
    return out


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
        if std_val is None or not math.isfinite(float(std_val)) or float(std_val) <= 1e-12:
            return f"{float(mean_val):.4f}"
        return f"{float(mean_val):.4f} ± {float(std_val):.4f}"

    return (
        f"| {display_method(method)} | {_maybe_bold(method, 'acc', fmt_pm('acc'), summary)} | "
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
        # Dataset-level bolding is row-wise: the displayed rule value is the
        # best rule-based value for that dataset, not a global cross-dataset max.
        if rule_f1 is not None:
            rule_f1_text = f"**{rule_f1_text}**"
        if rule_acc is not None:
            rule_acc_text = f"**{rule_acc_text}**"
        lines.append(
            f"| {item['dataset']} | {display_method(str(best_f1.get('method', '—')))} | {rule_f1_text} | {rf_f1_text} | "
            f"{display_method(str(best_acc.get('method', '—')))} | {rule_acc_text} | {rf_acc_text} |"
        )
    return lines


def warn_if_raw_names_leak(report_text: str) -> list[str]:
    """Return warnings for raw method identifiers in article-facing sections."""
    article_text = report_text.split("## Reproducibility", 1)[0]
    warnings = []
    for raw in sorted(RAW_METHOD_NAMES):
        if raw in article_text:
            warnings.append(f"raw method identifier appears before reproducibility section: {raw}")
    if "unc_mean" in article_text and "mean activated-rule Ω" not in article_text:
        warnings.append("unc_mean appears before reproducibility section without its definition")
    if "unc_comb" in article_text and "fused Ω" not in article_text:
        warnings.append("unc_comb appears before reproducibility section without its definition")
    return warnings


def render_report(summary: dict[str, object], metrics_path: Path, hard_cases_path: Path) -> str:
    datasets = list(summary["datasets"])
    methods_present = list(summary["methods_present"])
    n_datasets = len(datasets)
    hard_summary = read_hard_case_summary(hard_cases_path)
    hard_note = (
        f"The hard-case export contains {hard_summary['total_hard']} samples across {hard_summary['n_pairs']} dataset/inducer subsets. "
        f"Learned Dempster fusion improves over weighted rule-vote on {hard_summary['acc_wins']}/{hard_summary['acc_n']} hard-case subsets by Accuracy "
        f"and {hard_summary['macro_f1_wins']}/{hard_summary['macro_f1_n']} by Macro-F1; mean deltas are "
        f"{fmt(hard_summary['acc_mean_delta'])} Accuracy and {fmt(hard_summary['macro_f1_mean_delta'])} Macro-F1."
        if hard_summary
        else "Hard-case CSV summaries were not found, so hard-case claims should be treated as criteria-only diagnostics until regenerated."
    )
    rows = [
        "# DSGD-Auto Brief Project Report",
        "",
        "## Abstract",
        f"This report summarizes the current frozen-rule DSGD pipeline: repository-specific FOIL-style and RIPPER-style inducers emit readable rules, the induced ruleset is frozen, and learned evidential masses are trained on top of that frozen representation. The current snapshot uses one fixed 80/20 train/test split per dataset, split seed 42, and five training seeds (42-46) on that same split. Code and generated artifacts: {REPOSITORY_URL}.",
        "",
        "## What Was Built",
        "- `build_report.py` is the canonical public entry point for the full benchmark/report pipeline.",
        "- `train_test_runner.py` is the shared split/train/test engine.",
        "- `analyze_hard_cases.py` remains the hard-case analysis entry point.",
        "- FOIL and RIPPER here are repository-specific induced-and-shaped rule pipelines inspired by the classical algorithms, with project-specific grow/prune/search and pool-shaping choices.",
        "- Current protocol: one stratified 80/20 train/test split per dataset; rule induction runs on train; DSGD mass learning repeats over seeds 42-46; deterministic rule baselines reuse each frozen ruleset; Random Forest uses 400 trees with the matching run seed.",
        f"- Current benchmark snapshot: `{metrics_path}`",
        "",
        "Baseline definitions used below:",
        "- Ordered-rule baseline: evaluate the induced rules in emitted order after covering, pruning, and pool shaping; the first activated rule predicts its label, with the implementation default fallback if no rule fires.",
        "- First-hit fixed-confidence baseline: use the same first activated rule, then assign a fixed high probability to its predicted class for probability-based metrics. This is not reported as a true Laplace estimate unless the baseline is reimplemented and rerun.",
        "- Weighted rule-vote baseline: every activated rule casts an unlearned vote for its consequent class, weighted by support-scaled Laplace confidence `((support + 1) / (support + neg_covered + C)) * log(1 + support)`; normalized weighted class scores produce the class probabilities.",
        "- Learned Dempster fusion: learn per-rule mass logits over the selected focal elements, namely class singletons plus the explicit ignorance state Ω, and fuse fired-rule masses with Dempster's rule before applying the pignistic transform.",
        "- Frozen ruleset: rule induction is finished first, and the later mass-learning stage does not add, remove, or rewrite rules.",
        "- Hard-rule baselines: deterministic rule-only predictors that use the induced rules directly, without learned evidential mass parameters.",
        "- Hard case: a held-out test sample satisfying at least one conflict-oriented criterion computed from fired rules of the same frozen ruleset.",
        "- Metrics: ECE uses 15 equal-width confidence bins; `unc_mean` is mean activated-rule Ω and `unc_comb` is fused Ω after Dempster aggregation.",
        "",
        "Implementation settings:",
        "- FOIL/RIPPER rule caps: 400 rules per class, 1600 total, min positive support 1, precision tiers (0.80/8), (0.60/6), (0.40/5), (0.20/4) as min precision/max literals; current artifacts enable pool shaping.",
        "- RIPPER-style pruning: grow/prune split with reduced-error pruning; no full canonical RIPPER optimization pass is claimed.",
        "- DSGD training: AdamW, lr 1e-3, weight decay 2e-4, max 100 epochs, batch size 512 or 256 on small datasets, validation split 0.2 or 0.25 by profile, class-balanced NLL.",
        "- Random Forest: scikit-learn RandomForestClassifier with 400 trees, package-default depth/class weighting, run seed as random state.",
        "",
        "## Results",
        f"The current snapshot contains {summary['n_rows']} aggregated rows built from {summary['n_total_runs']} benchmark runs across {n_datasets} datasets and {summary['n_methods']} methods. Learned Dempster fusion is the best rule-based Accuracy variant on {summary['dsgd_best_counts']['acc']}/{n_datasets} datasets and the best Macro-F1 variant on {summary['dsgd_best_counts']['macro_f1']}/{n_datasets}; Random Forest is stronger than or tied with the best rule row on {summary['rf_win_tie_counts']['acc']}/{n_datasets} datasets for Accuracy and {summary['rf_win_tie_counts']['macro_f1']}/{n_datasets} for Macro-F1. That ordering should still be read as benchmark description, not as evidence for a broader mechanism or split-robust dominance.",
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
            "- Uncertainty metrics are reported only for methods that produce an evidential mass function with an explicit Ω component; hard-rule baselines and Random Forest do not produce a directly comparable Ω mass, so those entries are not applicable.",
            "- These averages are useful as a compact benchmark summary, but they should be read together with the dataset-level table below.",
            "- The current report summarizes five-seed averages on one fixed split per dataset; it does not replace broader split-robust evaluation.",
            f"- Paired descriptive counts versus weighted rule-vote: RIPPER learned fusion improves Accuracy on {summary['dsgd_vs_vote_wins']['acc'].get('RIPPER', 0)}/{summary['dsgd_vs_vote_total']['acc'].get('RIPPER', 0)} datasets and Macro-F1 on {summary['dsgd_vs_vote_wins']['macro_f1'].get('RIPPER', 0)}/{summary['dsgd_vs_vote_total']['macro_f1'].get('RIPPER', 0)}; FOIL learned fusion improves Accuracy on {summary['dsgd_vs_vote_wins']['acc'].get('FOIL', 0)}/{summary['dsgd_vs_vote_total']['acc'].get('FOIL', 0)} and Macro-F1 on {summary['dsgd_vs_vote_wins']['macro_f1'].get('FOIL', 0)}/{summary['dsgd_vs_vote_total']['macro_f1'].get('FOIL', 0)}.",
            "",
            "## Dataset-Level Snapshot",
            "Best rule-based result versus Random Forest on the main report metrics. Bold rule values mark the best rule-based value within that dataset row, not a global cross-dataset maximum:",
            *dataset_table_rows(summary),
            "",
            "## Hard-Case Note",
            f"The separate hard-case analysis in `{hard_cases_path}` is best read as diagnostic evidence on the same frozen rulesets for the same dataset/inducer pairs. A hard case is selected by rule-label disagreement, high fired-rule depth, low weighted-vote margin, minority support for the true class, or a one-true-many-wrong fired-rule pattern. {hard_note}",
            "",
            "## Interpretation",
            "The safest interpretation of the current snapshot is narrow: on this fixed-split five-seed frozen-rule benchmark, learned Dempster rows usually rank near the top of the rule-based methods and learned evidential masses are often competitive with or better than the raw rule aggregators built from the same ruleset. Random Forest remains the stronger pure predictive reference on most datasets, so the rule-based contribution should be framed as competitive inspectable prediction with explicit uncertainty rather than RF-level accuracy dominance. The inspection examples support only a readability claim about the displayed explanation objects; the combined explanation is a post-hoc summary of activated rules, not a new induced rule, not guaranteed to be logically equivalent to the fired-rule set, and not a broader comparative interpretability result.",
            "",
            "## Limitations",
            "The current results should be read as a fixed-split, multi-seed benchmark of a frozen-rule aggregation pipeline. They do not by themselves prove split-robust superiority, rule-diversity effects, or a general interpretability advantage. Stronger claims would require repeated train/test splits, ablations of rule-pool shaping, and direct analysis of rule overlap, fired-rule counts, conflict rates, and calibration under distributional variation. The current comparison also does not include an empirical reimplementation of prior DS classifiers such as Peñafiel et al.; it is restricted to aggregation variants over this repository's induced rulesets plus Random Forest.",
            "",
            "## Reproducibility",
            f"- Current benchmark snapshot: `{metrics_path}`",
            f"- Hard-case analysis: `{hard_cases_path}`",
            f"- Repository URL: {REPOSITORY_URL}",
            "- Computational cost summary: `src/results/COMPUTATIONAL_COST.md`",
            "- Pool-shaping ablation status: `src/results/POOL_SHAPING_ABLATION.md`",
            "- Standard custom run: `python3 train_test_runner.py --dataset-path ./adult.csv --inducers RIPPER,FOIL --save-root ./tmp_run --seeds 7,8 --test-size 0.25`",
            "- Frozen paper protocol: `python3 train_test_runner.py --dataset-path ./adult.csv --inducers RIPPER,FOIL --save-root ./tmp_run --paper-mode`",
            "- Regenerate the full pipeline: `python3 build_report.py --out-root src/results/raw_runs --results-dir src/results`",
            "- Internal result identifiers retained in CSV artifacts include `dsgd_dempster`, `native_ordered_rule`, `weighted_vote`, and `first_hit_laplace`; the article-facing labels above should be used in paper tables.",
            "",
            "## References",
            "- Dempster, 1968. [Upper and lower probabilities induced by a multivalued mapping](https://doi.org/10.1214/aoms/1177698950).",
            "- Shafer, 1976. [A Mathematical Theory of Evidence](https://press.princeton.edu/books/hardcover/9780691214696/a-mathematical-theory-of-evidence).",
            "- Quinlan, 1990. [Learning Logical Definitions from Relations](https://doi.org/10.1007/BF00117105).",
            "- Cohen, 1995. [Fast Effective Rule Induction](https://doi.org/10.1016/B978-1-55860-377-6.50023-2).",
            "- Breiman, 2001. [Random Forests](https://doi.org/10.1023/A:1010933404324).",
            "- Smets and Kennes, 1994. [The transferable belief model](https://doi.org/10.1016/0004-3702(94)90026-4).",
            "- Guo et al., 2017. [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).",
            "- UCI Machine Learning Repository, 2017. [Citation guidance](https://archive.ics.uci.edu/citation).",
            "- Vanschoren et al., 2013. [OpenML: Networked science in machine learning](https://doi.org/10.1145/2641190.2641198).",
            "- Sergio Peñafiel, Nelson Baloian, Hernan Sanson, and Juan Antonio Pino, 2020. [Applying Dempster-Shafer theory for developing a flexible, accurate and interpretable classifier](https://doi.org/10.1016/j.eswa.2020.113262).",
            "- Aik Tarkhanyan and Ashot Harutyunyan, 2025. [DSGD++: Performance and robustness improvements for the Dempster-Shafer Gradient Descent classifier](https://arxiv.org/abs/2507.00453).",
        ]
    )
    return "\n".join(rows) + "\n"


def write_brief_report(*, metrics_path: Path, hard_cases_path: Path, out_path: Path) -> int:
    summary = build_summary(read_rows(metrics_path))
    text = render_report(summary, metrics_path, hard_cases_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote: {out_path}")
    for warning in warn_if_raw_names_leak(text):
        print(f"[warn] {warning}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the DSGD-Auto brief report from aggregate metrics.")
    parser.add_argument("--metrics-path", default="src/results/ALL_DATASETS_metrics.csv")
    parser.add_argument("--hard-cases-path", default="src/results/hard_cases/HARD_CASE_ANALYSIS.md")
    parser.add_argument("--out-path", default="src/results/brief_report.md")
    args = parser.parse_args(argv)
    return write_brief_report(
        metrics_path=Path(args.metrics_path),
        hard_cases_path=Path(args.hard_cases_path),
        out_path=Path(args.out_path),
    )


if __name__ == "__main__":
    raise SystemExit(main())
