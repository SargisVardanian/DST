from __future__ import annotations

import sys
import tempfile
import unittest
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from DSClassifierMultiQ import DSClassifierMultiQ
from DSModelMultiQ import DSModelMultiQ
from Datasets_loader import load_dataset
from build_report import build_summary, render_report
import build_report
from benchmark_protocol import protocol_from_cli, protocol_from_passthrough
from experiment_analysis import write_cost_report, write_pool_shaping_ablation
from experiment_analysis import load_runs
from rule_generator import RuleGenerator


class RuntimeAndReportTests(unittest.TestCase):
    def test_native_ordered_rule_raw_predictions_are_supported(self) -> None:
        X = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        y = np.asarray([0, 0, 1, 1], dtype=int)
        clf = DSClassifierMultiQ(k=2, rule_algo="STATIC", max_iter=1, batch_size=2, early_stop_patience=1)
        clf.fit(X, y, feature_names=["x0", "x1"])

        pred = clf.raw_predict(X, method="native_ordered_rule")
        proba = clf.raw_predict_proba(X, method="native_ordered_rule")

        self.assertEqual(pred.shape, (4,))
        self.assertEqual(proba.shape, (4, 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))
        self.assertTrue(np.all((proba == 0.0) | (proba == 1.0)))

    def test_unknown_combination_rule_raises(self) -> None:
        with self.assertRaises(ValueError):
            DSModelMultiQ._normalize_combination_rule("banana")
        with self.assertRaises(ValueError):
            DSModelMultiQ._normalize_combination_rule("dsgd")
        with self.assertRaises(ValueError):
            DSModelMultiQ._normalize_combination_rule("cdsgd")

    def test_cache_failures_are_reported_explicitly(self) -> None:
        X = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.2],
                [0.2, 1.0],
                [0.8, 0.0],
                [1.0, 0.8],
            ],
            dtype=float,
        )
        y = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
        clf = DSClassifierMultiQ(k=2, rule_algo="STATIC", max_iter=2, batch_size=4, early_stop_patience=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "toy_dst.pkl"
            meta = clf.fit(
                X,
                y,
                feature_names=["x0", "x1"],
                use_cached_rules=True,
                rules_path=str(Path(tmpdir) / "missing.pkl"),
                save_rules_path=str(save_path),
            )
        self.assertEqual(meta["rule_source"], "generated")
        self.assertFalse(meta["cache_failures"])
        self.assertTrue(meta["cache_misses"])
        self.assertIn("missing.pkl", meta["cache_misses"][0])
        self.assertEqual(clf._last_fit_meta["rule_source"], "generated")

    def test_cache_failures_can_fail_fast(self) -> None:
        X = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        y = np.asarray([0, 0, 1, 1], dtype=int)
        clf = DSClassifierMultiQ(k=2, rule_algo="STATIC", max_iter=1, batch_size=2, early_stop_patience=1)
        with self.assertRaises(RuntimeError):
            with patch.object(clf, "load_model", side_effect=ValueError("bad artifact version")):
                clf.fit(
                    X,
                    y,
                    feature_names=["x0", "x1"],
                    use_cached_rules=True,
                    fail_on_cache_mismatch=True,
                    rules_path="/tmp/incompatible.pkl",
                )

    def test_cache_path_list_is_supported(self) -> None:
        X = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        y = np.asarray([0, 0, 1, 1], dtype=int)
        clf = DSClassifierMultiQ(k=2, rule_algo="STATIC", max_iter=1, batch_size=2, early_stop_patience=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "toy_dst.pkl"
            meta = clf.fit(
                X,
                y,
                feature_names=["x0", "x1"],
                use_cached_rules=True,
                rules_path=[str(Path(tmpdir) / "missing_a.pkl"), str(Path(tmpdir) / "missing_b.pkl")],
                save_rules_path=str(save_path),
            )
        self.assertEqual(meta["rule_source"], "generated")
        self.assertEqual(len(meta["cache_misses"]), 2)
        self.assertFalse(meta["cache_failures"])

    def test_cache_fallback_uses_split_seed_without_incompatibility_warning(self) -> None:
        X = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        y = np.asarray([0, 0, 1, 1], dtype=int)
        producer = DSClassifierMultiQ(k=2, rule_algo="STATIC", max_iter=1, batch_size=2, early_stop_patience=1, seed=42)
        consumer = DSClassifierMultiQ(k=2, rule_algo="STATIC", max_iter=1, batch_size=2, early_stop_patience=1, seed=43)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "seed42.pkl"
            producer.fit(X, y, feature_names=["x0", "x1"], save_rules_path=str(cache_path))
            meta = consumer.fit(
                X,
                y,
                feature_names=["x0", "x1"],
                use_cached_rules=True,
                rules_path=[str(Path(tmpdir) / "seed43.pkl"), str(cache_path)],
            )
        self.assertEqual(meta["rule_source"], "cache")
        self.assertEqual(meta["loaded_from"], str(cache_path))
        self.assertEqual(len(meta["cache_misses"]), 1)
        self.assertFalse(meta["cache_failures"])

    def test_brief_report_handles_uppercase_schema(self) -> None:
        rows = [
            {
                "Dataset": "adult",
                "Algo": "FOIL",
                "Method": "Dempster",
                "Acc": "0.90",
                "F1": "0.80",
                "NLL": "0.40",
                "ECE": "0.05",
            },
            {
                "Dataset": "adult",
                "Algo": "RF",
                "Method": "rf",
                "Acc": "0.91",
                "F1": "0.82",
                "NLL": "0.30",
                "ECE": "0.04",
            },
        ]
        summary = build_summary(rows)
        self.assertIn("adult", summary["datasets"])
        self.assertIn("FOIL:dsgd_dempster", summary["mean_metrics"])
        self.assertIn("RF", summary["mean_metrics"])

    def test_brief_report_uses_measured_wording_and_disclosure(self) -> None:
        rows = [
            {
                "dataset": "adult",
                "method": "RF",
                "acc": "0.91",
                "macro_f1": "0.82",
                "nll": "0.30",
                "ece": "0.04",
                "n_runs": "1",
                "n_seeds": "1",
            },
            {
                "dataset": "adult",
                "method": "FOIL:dsgd_dempster",
                "acc": "0.90",
                "macro_f1": "0.80",
                "nll": "0.40",
                "ece": "0.05",
                "n_runs": "1",
                "n_seeds": "1",
            },
        ]
        summary = build_summary(rows)
        text = render_report(summary, Path("/tmp/metrics.csv"), Path("/tmp/hard_cases.md"))
        self.assertIn("repository-specific induced-and-shaped rule pipelines", text)
        self.assertIn("learned Dempster rows usually rank near the top of the rule-based methods", text)
        self.assertIn("not as evidence for a broader mechanism", text)
        self.assertIn("not a broader comparative interpretability result", text)
        self.assertNotIn("strongest rule-based average configuration", text)

    def test_brief_report_renders_mean_plus_std_when_available(self) -> None:
        rows = [
            {
                "dataset": "adult",
                "method": "RF",
                "acc": "0.91",
                "std_acc": "0.02",
                "macro_f1": "0.82",
                "std_macro_f1": "0.03",
                "nll": "0.30",
                "std_nll": "0.01",
                "ece": "0.04",
                "std_ece": "0.01",
                "n_runs": "3",
                "n_seeds": "3",
            },
            {
                "dataset": "adult",
                "method": "FOIL:dsgd_dempster",
                "acc": "0.90",
                "std_acc": "0.01",
                "macro_f1": "0.80",
                "std_macro_f1": "0.02",
                "nll": "0.40",
                "std_nll": "0.02",
                "ece": "0.05",
                "std_ece": "0.01",
                "n_runs": "3",
                "n_seeds": "3",
            },
        ]
        summary = build_summary(rows)
        text = render_report(summary, Path("/tmp/metrics.csv"), Path("/tmp/hard_cases.md"))
        self.assertIn("0.9100 ± 0.0200", text)
        self.assertIn("| RF |", text)
        self.assertIn("3 / 3", text)

    def test_rule_generator_uses_configured_seed(self) -> None:
        first = RuleGenerator(seed=7)
        second = RuleGenerator(seed=7)
        third = RuleGenerator(seed=8)
        self.assertEqual(int(first._rng.integers(0, 10_000)), int(second._rng.integers(0, 10_000)))
        self.assertNotEqual(int(first._rng.integers(0, 10_000)), int(third._rng.integers(0, 10_000)))

    def test_rule_generator_pool_shaping_changes_learned_pool(self) -> None:
        X = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.1],
                [0.8, 0.9, 1.0],
                [0.9, 0.8, 1.0],
                [1.0, 0.9, 0.8],
                [0.9, 1.0, 0.9],
                [0.3, 0.2, 0.1],
                [0.4, 0.2, 0.1],
                [0.6, 0.8, 0.9],
                [0.7, 0.9, 0.8],
            ],
            dtype=float,
        )
        y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], dtype=int)
        feature_names = ["f0", "f1", "f2"]

        plain = RuleGenerator(algo="ripper", seed=42, verbose=False)
        plain.fit(X, y, feature_names=feature_names)

        shaped = RuleGenerator(
            algo="ripper",
            seed=42,
            verbose=False,
            enable_pool_shaping=True,
            pool_target_ratio=0.5,
            pool_min_keep_per_class=1,
        )
        shaped.fit(X, y, feature_names=feature_names)

        self.assertGreater(len(plain.ordered_rules), 0)
        self.assertLessEqual(len(shaped.ordered_rules), len(plain.ordered_rules))
        self.assertTrue(any("pool_score" in stats for _, _, stats in shaped.ordered_rules))

    def test_loader_keeps_unique_continuous_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "toy.csv"
            pd.DataFrame(
                {
                    "fixed acidity": [0.11, 0.22, 0.33, 0.44],
                    "Solidity": [10.0, 20.0, 30.0, 40.0],
                    "labels": [0, 1, 0, 1],
                }
            ).to_csv(csv_path, index=False)
            X, y, feature_names, value_decoders, stats = load_dataset(csv_path, return_stats=True)

        self.assertEqual(stats["shape"], (4, 2))
        self.assertEqual(feature_names, ["fixed acidity", "Solidity"])
        self.assertEqual(value_decoders, {})

    def test_loader_drops_target_adjacent_good_from_df_wine(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "df_wine.csv"
            pd.DataFrame(
                {
                    "fixed acidity": [7.4, 7.8, 7.8, 11.2],
                    "quality": [5, 5, 5, 6],
                    "good": [0, 0, 0, 1],
                    "labels": [0, 0, 0, 1],
                }
            ).to_csv(csv_path, index=False)
            X, y, feature_names, value_decoders, stats = load_dataset(csv_path, return_stats=True)

        self.assertEqual(stats["shape"], (4, 2))
        self.assertEqual(feature_names, ["fixed acidity", "quality"])
        self.assertEqual(value_decoders, {})

    def test_canonical_numeric_candidates_include_supervised_midpoint(self) -> None:
        X = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [100.0], [101.0]], dtype=float)
        pm = np.asarray([False, False, False, False, False, False, True, True], dtype=bool)

        generator = RuleGenerator(seed=42)
        generator._X = X
        generator._feat = ["f0"]
        generator._is_cat = [False]

        thresholds = {round(float(thr), 6) for _, thr in generator._candidates_for_feature(0, pm)}

        self.assertIn(52.5, thresholds)

    def test_canonical_pruning_prefers_better_prefix_accuracy(self) -> None:
        X = np.asarray(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=float,
        )
        prune_pos = np.asarray([True, True, True, True, True, False, False, False, False, False, False], dtype=bool)
        prune_neg = ~prune_pos
        cond = {"x0": (">", 0.5), "x1": (">", 0.5)}

        generator = RuleGenerator(seed=42)
        generator._X = X
        generator._feat = ["x0", "x1"]
        generator._is_cat = [False, False]

        pruned = generator._reduced_error_prune(cond, prune_pos, prune_neg)

        self.assertEqual(list(pruned.keys()), ["x0"])

    def test_build_report_preserves_raw_run_models_and_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "results"
            raw_runs = results_dir / "raw_runs"
            bench_dir = raw_runs / "benchmarks"
            model_dir = raw_runs / "models"
            rule_dir = raw_runs / "rules"
            hard_case_dir = results_dir / "hard_cases"

            bench_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            rule_dir.mkdir(parents=True)
            hard_case_dir.mkdir(parents=True)

            (model_dir / "keep.pkl").write_text("model", encoding="utf-8")
            (rule_dir / "keep.dsb").write_text("rule", encoding="utf-8")

            df = pd.DataFrame(
                [
                    {
                        "dataset": "adult",
                        "system": "rf",
                        "inducer": "RF",
                        "acc": 0.91,
                        "macro_f1": 0.82,
                        "precision": 0.83,
                        "recall": 0.81,
                        "nll": 0.30,
                        "ece": 0.04,
                        "unc_mean": 0.20,
                        "unc_comb": 0.10,
                    },
                    {
                        "dataset": "adult",
                        "system": "dsgd_dempster",
                        "inducer": "FOIL",
                        "acc": 0.90,
                        "macro_f1": 0.80,
                        "precision": 0.81,
                        "recall": 0.79,
                        "nll": 0.40,
                        "ece": 0.05,
                        "unc_mean": 0.30,
                        "unc_comb": 0.15,
                    },
                ]
            )
            df.to_csv(bench_dir / "bench__adult__FOIL.csv", index=False)

            argv = [
                "build_report.py",
                "--out-root",
                str(raw_runs),
                "--results-dir",
                str(results_dir),
                "--hard-case-dir",
                str(hard_case_dir),
                "--datasets",
                "adult",
            ]
            with patch.object(sys, "argv", argv):
                rc = build_report.report_outputs_main()
            model_kept = (model_dir / "keep.pkl").exists()
            rule_kept = (rule_dir / "keep.dsb").exists()

        self.assertEqual(rc, 0)
        self.assertTrue(model_kept)
        self.assertTrue(rule_kept)

    def test_build_report_main_report_mode_routes_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "results"
            raw_runs = results_dir / "raw_runs"
            bench_dir = raw_runs / "benchmarks"
            hard_case_dir = results_dir / "hard_cases"
            bench_dir.mkdir(parents=True)
            hard_case_dir.mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "dataset": "adult",
                        "system": "rf",
                        "inducer": "RF",
                        "acc": 0.91,
                        "macro_f1": 0.82,
                        "precision": 0.83,
                        "recall": 0.81,
                        "nll": 0.30,
                        "ece": 0.04,
                        "unc_mean": np.nan,
                        "unc_comb": np.nan,
                    },
                    {
                        "dataset": "adult",
                        "system": "dsgd_dempster",
                        "inducer": "FOIL",
                        "acc": 0.90,
                        "macro_f1": 0.80,
                        "precision": 0.81,
                        "recall": 0.79,
                        "nll": 0.40,
                        "ece": 0.05,
                        "unc_mean": 0.30,
                        "unc_comb": 0.15,
                    },
                ]
            ).to_csv(bench_dir / "bench__adult__FOIL.csv", index=False)

            rc = build_report.main(
                [
                    "report",
                    "--out-root",
                    str(raw_runs),
                    "--results-dir",
                    str(results_dir),
                    "--hard-case-dir",
                    str(hard_case_dir),
                    "--datasets",
                    "adult",
                ]
            )
            metrics_exists = (results_dir / "ALL_DATASETS_metrics.csv").exists()

        self.assertEqual(rc, 0)
        self.assertTrue(metrics_exists)

    def test_protocol_defaults_to_standard_mode(self) -> None:
        protocol = protocol_from_cli(raw_seeds="7,8", raw_test_size=0.25, paper_mode=False)
        self.assertEqual(protocol.mode_label, "standard")
        self.assertEqual(protocol.seeds, [7, 8])
        self.assertAlmostEqual(protocol.test_size, 0.25)
        self.assertFalse(protocol.override_messages)

    def test_protocol_paper_mode_records_overrides(self) -> None:
        protocol = protocol_from_cli(raw_seeds="7,8", raw_test_size=0.25, paper_mode=True)
        self.assertEqual(protocol.mode_label, "paper")
        self.assertEqual(protocol.seeds, [42])
        self.assertAlmostEqual(protocol.test_size, 0.2)
        self.assertEqual(len(protocol.override_messages), 2)

    def test_build_report_passthrough_protocol_matches_cli_semantics(self) -> None:
        standard = protocol_from_passthrough(["--seeds", "9", "--test-size", "0.3"])
        paper = protocol_from_passthrough(["--seeds", "9", "--test-size", "0.3", "--paper-mode"])
        self.assertEqual(standard.mode_label, "standard")
        self.assertEqual(standard.seeds, [9])
        self.assertAlmostEqual(standard.test_size, 0.3)
        self.assertEqual(paper.mode_label, "paper")
        self.assertEqual(paper.seeds, [42])
        self.assertAlmostEqual(paper.test_size, 0.2)

    def test_build_report_aggregates_multi_seed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results_dir = root / "results"
            raw_runs = results_dir / "raw_runs"
            bench_dir = raw_runs / "benchmarks"
            hard_case_dir = results_dir / "hard_cases"
            bench_dir.mkdir(parents=True)
            hard_case_dir.mkdir(parents=True)

            df = pd.DataFrame(
                [
                    {
                        "dataset": "adult",
                        "inducer": "FOIL",
                        "seed": 1,
                        "system": "dsgd_dempster",
                        "acc": 0.80,
                        "macro_f1": 0.70,
                        "precision": 0.70,
                        "recall": 0.70,
                        "nll": 0.40,
                        "ece": 0.08,
                        "unc_mean": 0.20,
                        "unc_comb": 0.10,
                        "fusion_depth": 2.0,
                    },
                    {
                        "dataset": "adult",
                        "inducer": "FOIL",
                        "seed": 2,
                        "system": "dsgd_dempster",
                        "acc": 0.90,
                        "macro_f1": 0.80,
                        "precision": 0.80,
                        "recall": 0.80,
                        "nll": 0.30,
                        "ece": 0.06,
                        "unc_mean": 0.30,
                        "unc_comb": 0.12,
                        "fusion_depth": 4.0,
                    },
                    {
                        "dataset": "adult",
                        "inducer": "RF",
                        "seed": 1,
                        "system": "rf",
                        "acc": 0.95,
                        "macro_f1": 0.85,
                        "precision": 0.85,
                        "recall": 0.85,
                        "nll": 0.20,
                        "ece": 0.03,
                        "unc_mean": np.nan,
                        "unc_comb": np.nan,
                        "fusion_depth": np.nan,
                    },
                ]
            )
            df.to_csv(bench_dir / "bench__adult__FOIL.csv", index=False)

            argv = [
                "build_report.py",
                "--out-root",
                str(raw_runs),
                "--results-dir",
                str(results_dir),
                "--hard-case-dir",
                str(hard_case_dir),
                "--datasets",
                "adult",
            ]
            with patch.object(sys, "argv", argv):
                rc = build_report.report_outputs_main()

            agg = pd.read_csv(results_dir / "ALL_DATASETS_metrics.csv")
            raw = pd.read_csv(results_dir / "method_suite" / "method_suite_long_raw.csv")

        self.assertEqual(rc, 0)
        self.assertEqual(len(raw), 3)
        foil = agg[agg["method"] == "FOIL:dsgd_dempster"].iloc[0]
        self.assertEqual(int(foil["n_runs"]), 2)
        self.assertEqual(int(foil["n_seeds"]), 2)
        self.assertAlmostEqual(float(foil["acc"]), 0.85)
        self.assertAlmostEqual(float(foil["macro_f1"]), 0.75)
        self.assertAlmostEqual(float(foil["std_acc"]), float(np.std([0.80, 0.90], ddof=1)))
        self.assertAlmostEqual(float(foil["min_acc"]), 0.80)
        self.assertAlmostEqual(float(foil["max_acc"]), 0.90)

    def test_init_fallback_is_recorded_in_fit_meta(self) -> None:
        X = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        y = np.asarray([0, 0, 1, 1], dtype=int)
        clf = DSClassifierMultiQ(k=2, rule_algo="STATIC", max_iter=1, batch_size=2, early_stop_patience=1)
        with patch.object(clf.model, "init_masses_dsgdpp", side_effect=RuntimeError("boom")):
            meta = clf.fit(X, y, feature_names=["x0", "x1"])
        self.assertTrue(meta["warnings"])
        self.assertIn("init failed", meta["warnings"][0])
        self.assertIn("reset_masses()", meta["warnings"][0])

    def test_binary_artifact_version_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "bad.pkl"
            payload = {
                "artifact_version": 1,
                "k": 2,
                "algo": "STATIC",
                "feature_names": ["x0", "x1"],
                "value_decoders": {},
                "rules": [],
            }
            with bad_path.open("wb") as fh:
                pickle.dump(payload, fh)
            model = DSModelMultiQ(k=2, algo="STATIC")
            with self.assertRaises(ValueError):
                model.load_rules_bin(bad_path)

    def test_experiment_analysis_loads_seed_and_pool_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_dir = Path(tmpdir) / "benchmarks"
            bench_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "dataset": "adult",
                        "inducer": "FOIL",
                        "seed": 42,
                        "split_seed": 42,
                        "system": "dsgd_dempster",
                        "pool_shaping": True,
                        "acc": 0.9,
                        "macro_f1": 0.8,
                    }
                ]
            ).to_csv(bench_dir / "bench__adult__FOIL__split42__seed42.csv", index=False)
            df = load_runs([bench_dir])

        self.assertEqual(df.loc[0, "method"], "FOIL:dsgd_dempster")
        self.assertEqual(int(df.loc[0, "split_seed"]), 42)
        self.assertTrue(bool(df.loc[0, "pool_shaping"]))

    def test_pool_shaping_ablation_suppresses_empty_placeholder_tables(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "dataset": "adult",
                    "method": "FOIL:dsgd_dempster",
                    "pool_shaping": True,
                    "seed": 42,
                    "split_seed": 42,
                    "macro_f1": 0.8,
                    "acc": 0.9,
                    "nll": 0.3,
                    "ece": 0.05,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            write_pool_shaping_ablation(df, out_dir)
            text = (out_dir / "POOL_SHAPING_ABLATION.md").read_text(encoding="utf-8")
        self.assertIn("Deferred research debt", text)
        self.assertNotIn("| Method | n | mean(on-off)", text)

    def test_cost_report_includes_scope_and_limitations(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "method": "FOIL:dsgd_dempster",
                    "train_wall_sec": 1.2,
                    "n_rules": 10,
                    "avg_literals": 2.5,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            write_cost_report(df, out_dir)
            text = (out_dir / "COMPUTATIONAL_COST.md").read_text(encoding="utf-8")
        self.assertIn("Host context", text)
        self.assertIn("Not measured here", text)


if __name__ == "__main__":
    unittest.main()
