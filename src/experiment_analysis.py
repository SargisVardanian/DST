#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None


def _normalize_method(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["inducer"] = out["inducer"].astype(str).str.upper()
    out["system"] = out["system"].astype(str)
    out["method"] = np.where(out["inducer"] == "RF", "RF", out["inducer"] + ":" + out["system"])
    return out


def load_runs(bench_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for bench_dir in bench_dirs:
        for path in sorted(bench_dir.glob("bench__*.csv")):
            df = pd.read_csv(path)
            if df.empty:
                continue
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "split_seed" not in df.columns:
        df["split_seed"] = df.get("seed")
    if "pool_shaping" not in df.columns:
        df["pool_shaping"] = np.nan
    return _normalize_method(df)


def _paired_metric(df: pd.DataFrame, left: str, right: str, metric: str) -> pd.DataFrame:
    cur = df[df["method"].isin([left, right])].copy()
    if cur.empty:
        return pd.DataFrame()
    keys = ["dataset", "seed", "split_seed", "method"]
    if "pool_shaping" in cur.columns and cur["pool_shaping"].notna().any():
        keys.insert(3, "pool_shaping")
    pvt = cur.pivot_table(index=[k for k in keys if k != "method"], columns="method", values=metric, aggfunc="mean")
    pvt = pvt.dropna(subset=[left, right]).reset_index()
    if pvt.empty:
        return pd.DataFrame()
    pvt["delta"] = pvt[left] - pvt[right]
    return pvt


def write_significance_report(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons = [
        ("FOIL:dsgd_dempster", "FOIL:weighted_vote"),
        ("FOIL:dsgd_dempster", "FOIL:first_hit_laplace"),
        ("FOIL:dsgd_dempster", "FOIL:native_ordered_rule"),
        ("RIPPER:dsgd_dempster", "RIPPER:weighted_vote"),
        ("RIPPER:dsgd_dempster", "RIPPER:first_hit_laplace"),
        ("RIPPER:dsgd_dempster", "RIPPER:native_ordered_rule"),
    ]
    rows = []
    lines = ["# Multi-Seed Significance Summary", ""]
    for metric in ["macro_f1", "acc", "nll", "ece"]:
        lines.extend([f"## {metric}", "", "| Left | Right | n | mean delta | median delta | wins | losses | test | p-value |", "|---|---|---:|---:|---:|---:|---:|---|---:|"])
        for left, right in comparisons:
            paired = _paired_metric(df, left, right, metric)
            if paired.empty:
                continue
            delta = paired["delta"].to_numpy(dtype=float)
            wins = int((delta > 0).sum())
            losses = int((delta < 0).sum())
            test_name = "none"
            pval = np.nan
            if wilcoxon is not None and delta.size >= 5 and np.any(np.abs(delta) > 1e-12):
                alt = "greater" if metric in {"macro_f1", "acc"} else "less"
                signed = delta if metric in {"macro_f1", "acc"} else -delta
                stat = wilcoxon(signed, alternative="greater", zero_method="wilcox")
                pval = float(stat.pvalue)
                test_name = "wilcoxon"
            rows.append(
                {
                    "metric": metric,
                    "left": left,
                    "right": right,
                    "n": int(delta.size),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "wins": wins,
                    "losses": losses,
                    "test": test_name,
                    "p_value": pval,
                }
            )
            lines.append(f"| {left} | {right} | {int(delta.size)} | {float(delta.mean()):.4f} | {float(np.median(delta)):.4f} | {wins} | {losses} | {test_name} | {pval:.4g} |")
        lines.append("")
    pd.DataFrame(rows).to_csv(out_dir / "significance_summary.csv", index=False)
    (out_dir / "MULTI_SEED_SIGNIFICANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cost_report(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cur = df.copy()
    for col in ["dataset", "seed", "split_seed"]:
        if col not in cur.columns:
            cur[col] = np.nan
    cur["train_wall_sec"] = pd.to_numeric(cur.get("train_wall_sec"), errors="coerce")
    cur["n_rules"] = pd.to_numeric(cur.get("n_rules"), errors="coerce")
    cur["avg_literals"] = pd.to_numeric(cur.get("avg_literals"), errors="coerce")
    summary = (
        cur.groupby("method", observed=False)[["train_wall_sec", "n_rules", "avg_literals"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = ["method"] + [f"{a}_{b}" for a, b in summary.columns.tolist()[1:]]
    summary.to_csv(out_dir / "cost_summary.csv", index=False)
    slowest = cur.sort_values("train_wall_sec", ascending=False)[["dataset", "method", "seed", "split_seed", "train_wall_sec", "n_rules", "avg_literals"]].head(20)
    slowest.to_csv(out_dir / "cost_slowest_runs.csv", index=False)
    lines = [
        "# Computational Cost Summary",
        "",
        f"- Host context: `{platform.system()} {platform.release()}` on `{platform.machine()}`.",
        "- Scope: wall-clock training time, learned rule count, and average literals per rule from the benchmark CSVs.",
        "- Not measured here: peak memory, inference latency, or cross-machine comparability.",
        "",
        "| Method | mean train sec | std | mean rules | mean avg_literals |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values("train_wall_sec_mean", ascending=False).iterrows():
        lines.append(
            f"| {row['method']} | {float(row['train_wall_sec_mean']):.2f} | {float(row['train_wall_sec_std']) if pd.notna(row['train_wall_sec_std']) else 0.0:.2f} | "
            f"{float(row['n_rules_mean']) if pd.notna(row['n_rules_mean']) else float('nan'):.2f} | {float(row['avg_literals_mean']) if pd.notna(row['avg_literals_mean']) else float('nan'):.2f} |"
        )
    lines.extend(["", "Top slowest runs are stored in `cost_slowest_runs.csv`. Treat these numbers as same-machine descriptive cost only."])
    (out_dir / "COMPUTATIONAL_COST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pool_shaping_ablation(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cur = df[df["method"].isin(["FOIL:dsgd_dempster", "RIPPER:dsgd_dempster"])].copy()
    cur = cur[cur["pool_shaping"].isin([True, False])].copy()
    coverage_rows: list[dict[str, object]] = []
    coverage_lines = [
        "## Coverage",
        "",
        "| Method | shaping=False rows | shaping=True rows | paired dataset/seed/split combos |",
        "|---|---:|---:|---:|",
    ]
    for method in ["FOIL:dsgd_dempster", "RIPPER:dsgd_dempster"]:
        sub = cur[cur["method"] == method].copy()
        off = int((sub["pool_shaping"] == False).sum())
        on = int((sub["pool_shaping"] == True).sum())
        pairable = 0
        missing_examples: list[str] = []
        if not sub.empty:
            combo = (
                sub.groupby(["dataset", "seed", "split_seed", "pool_shaping"], observed=False)
                .size()
                .reset_index(name="n")
            )
            pvt = combo.pivot_table(
                index=["dataset", "seed", "split_seed"],
                columns="pool_shaping",
                values="n",
                aggfunc="sum",
                fill_value=0,
            )
            if pvt.empty:
                has_on = pd.Series(dtype=bool)
                has_off = pd.Series(dtype=bool)
                paired_mask = pd.Series(dtype=bool)
            else:
                has_on = (pvt[True] > 0) if True in pvt.columns else pd.Series(False, index=pvt.index)
                has_off = (pvt[False] > 0) if False in pvt.columns else pd.Series(False, index=pvt.index)
                paired_mask = has_on & has_off
            pairable = int(paired_mask.sum()) if not pvt.empty else 0
            missing_idx = pvt.index[~paired_mask] if not pvt.empty else []
            for dataset, seed, split_seed in list(missing_idx[:3]):
                flags = sub[
                    (sub["dataset"] == dataset)
                    & (sub["seed"] == seed)
                    & (sub["split_seed"] == split_seed)
                ]["pool_shaping"].dropna().astype(bool).unique().tolist()
                missing_examples.append(f"{dataset}/seed{int(seed)}/split{int(split_seed)} only={sorted(flags)}")
        coverage_rows.append(
            {
                "method": method,
                "shaping_false_rows": off,
                "shaping_true_rows": on,
                "paired_combos": pairable,
                "missing_examples": "; ".join(missing_examples),
            }
        )
        coverage_lines.append(f"| {method} | {off} | {on} | {pairable} |")
        if missing_examples:
            coverage_lines.append(f"- {method} missing pair examples: " + "; ".join(missing_examples))
    coverage_lines.append("")
    pd.DataFrame(coverage_rows).to_csv(out_dir / "pool_shaping_coverage.csv", index=False)
    if cur.empty:
        (out_dir / "POOL_SHAPING_ABLATION.md").write_text(
            "# Pool Shaping Ablation\n\nDeferred research debt: no paired runs with both `pool_shaping=True` and `pool_shaping=False` were found in the current benchmark exports, so this ablation is intentionally suppressed rather than shown as an empty table.\n\n"
            + "\n".join(coverage_lines)
            + "\n",
            encoding="utf-8",
        )
        return
    rows = []
    lines = ["# Pool Shaping Ablation", ""]
    for metric in ["macro_f1", "acc", "nll", "ece"]:
        lines.extend([f"## {metric}", "", "| Method | n | mean(on-off) | median(on-off) | shaping_on_wins | shaping_off_wins |", "|---|---:|---:|---:|---:|---:|"])
        for method in ["FOIL:dsgd_dempster", "RIPPER:dsgd_dempster"]:
            sub = cur[cur["method"] == method].copy()
            pvt = sub.pivot_table(index=["dataset", "seed", "split_seed"], columns="pool_shaping", values=metric, aggfunc="mean")
            if True not in pvt.columns or False not in pvt.columns:
                continue
            delta = (pvt[True] - pvt[False]).dropna().to_numpy(dtype=float)
            if delta.size == 0:
                continue
            wins = int((delta > 0).sum()) if metric in {"macro_f1", "acc"} else int((delta < 0).sum())
            losses = int((delta < 0).sum()) if metric in {"macro_f1", "acc"} else int((delta > 0).sum())
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "n": int(delta.size),
                    "mean_on_minus_off": float(delta.mean()),
                    "median_on_minus_off": float(np.median(delta)),
                    "shaping_on_wins": wins,
                    "shaping_off_wins": losses,
                }
            )
            lines.append(f"| {method} | {int(delta.size)} | {float(delta.mean()):.4f} | {float(np.median(delta)):.4f} | {wins} | {losses} |")
        lines.append("")
    pd.DataFrame(rows).to_csv(out_dir / "pool_shaping_ablation.csv", index=False)
    if rows:
        (out_dir / "POOL_SHAPING_ABLATION.md").write_text("\n".join(lines + coverage_lines) + "\n", encoding="utf-8")
    else:
        (out_dir / "POOL_SHAPING_ABLATION.md").write_text(
            "# Pool Shaping Ablation\n\nDeferred research debt: benchmark rows mention pool-shaping metadata, but no method currently has paired on/off measurements on matching dataset/seed/split combinations.\n\n"
            + "\n".join(coverage_lines)
            + "\n",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize multi-seed, cost, and pool-shaping experiment outputs.")
    ap.add_argument("--bench-dirs", type=str, required=True, help="Comma-separated benchmark directories.")
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args(argv)

    bench_dirs = [Path(x).expanduser().resolve() for x in args.bench_dirs.split(",") if x.strip()]
    out_dir = Path(args.out_dir).expanduser().resolve()
    df = load_runs(bench_dirs)
    if df.empty:
        print("[error] no benchmark rows found")
        return 2
    write_significance_report(df, out_dir)
    write_cost_report(df, out_dir)
    write_pool_shaping_ablation(df, out_dir)
    print(f"wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
