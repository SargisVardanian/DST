from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import streamlit as st
except ModuleNotFoundError:  # CLI mode does not require Streamlit.
    st = None

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

try:
    from .app_support import (  # noqa: E402
        APP_STORE,
        EXTRACTED_DIR,
        ROOT,
        RUNS_DIR,
        create_run_manifest,
        decode_row_for_manual_input,
        encode_manual_values,
        ensure_app_dirs,
        extract_bundle,
        get_run_by_id,
        infer_schema_from_csv,
        infer_schema_from_model,
        json_default,
        list_saved_runs,
        load_processed_dataset_rows,
        load_run_manifest,
        locate_model_and_rules_from_bundle,
        locate_run_artifacts,
        make_model_bundle,
        persist_uploaded_file,
        read_csv_preview,
        read_json,
        read_run_schema,
        read_text_preview,
        resolve_latest_run_manifest,
        resolve_source_dataset_path,
        run_id_for,
        schema_default_values,
        slugify,
        store_latest_run,
        utc_stamp,
        write_json,
        zip_dir_bytes,
    )
    from .benchmark_protocol import protocol_from_cli  # noqa: E402
    from .Datasets_loader import load_dataset  # noqa: E402
    from .DSClassifierMultiQ import DSClassifierMultiQ  # noqa: E402
    from .sample_rule_inspector import build_web_inspection_payload  # noqa: E402
    from .train_test_runner import benchmark_main  # noqa: E402
except ImportError:  # pragma: no cover - direct script/import fallback
    from app_support import (  # noqa: E402
        APP_STORE,
        EXTRACTED_DIR,
        ROOT,
        RUNS_DIR,
        create_run_manifest,
        decode_row_for_manual_input,
        encode_manual_values,
        ensure_app_dirs,
        extract_bundle,
        get_run_by_id,
        infer_schema_from_csv,
        infer_schema_from_model,
        json_default,
        list_saved_runs,
        load_processed_dataset_rows,
        load_run_manifest,
        locate_model_and_rules_from_bundle,
        locate_run_artifacts,
        make_model_bundle,
        persist_uploaded_file,
        read_csv_preview,
        read_json,
        read_run_schema,
        read_text_preview,
        resolve_latest_run_manifest,
        resolve_source_dataset_path,
        run_id_for,
        schema_default_values,
        slugify,
        store_latest_run,
        utc_stamp,
        write_json,
        zip_dir_bytes,
    )
    from benchmark_protocol import protocol_from_cli  # noqa: E402
    from Datasets_loader import load_dataset  # noqa: E402
    from DSClassifierMultiQ import DSClassifierMultiQ  # noqa: E402
    from sample_rule_inspector import build_web_inspection_payload  # noqa: E402
    from train_test_runner import benchmark_main  # noqa: E402

APP_ENTRY = Path(__file__).resolve()


def run_cmd_stream(args: list[str], output_placeholder, status_placeholder) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        args,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    lines: list[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            lines.append(line)
            output_placeholder.code("".join(lines[-250:]) or "(no output yet)", language="text")
    ret = proc.wait()
    status_placeholder.empty()
    return ret, "".join(lines)


def build_raw_row_from_schema(schema: dict[str, Any], *, prefix: str = "manual", prefill: dict[str, Any] | None = None) -> list[Any]:
    values: list[Any] = []
    features = list(schema.get("features", []))
    if not features:
        return values
    total = len(features)
    for idx, feature in enumerate(features, start=1):
        name = str(feature.get("name", "feature"))
        kind = str(feature.get("kind", "numeric"))
        key = f"{prefix}__{slugify(name)}"
        default_value = (prefill or {}).get(name, feature.get("default", 0.0))
        st.markdown(f"**{idx}. {name}**")
        if kind == "categorical":
            choices = list(feature.get("choices") or [])
            if default_value is None:
                default = choices[0] if choices else ""
            else:
                try:
                    dv = float(default_value)
                    default = str(int(round(dv))) if abs(dv - round(dv)) <= 1e-9 else str(default_value)
                except Exception:
                    default = str(default_value)
            if default not in choices and choices:
                default = choices[0]
            if choices:
                st.caption(f"Choice {idx} of {total} • categorical • available values: {', '.join(choices[:8])}{' ...' if len(choices) > 8 else ''}")
            else:
                st.caption(f"Choice {idx} of {total} • categorical")
            st.session_state.setdefault(key, default)
            selected = st.selectbox(
                "value",
                options=choices,
                index=choices.index(st.session_state[key]) if st.session_state.get(key) in choices else choices.index(default) if default in choices else 0,
                key=key,
                label_visibility="collapsed",
            )
            values.append(selected)
            continue

        min_value = feature.get("min")
        max_value = feature.get("max")
        default = float(default_value if default_value is not None else 0.0)
        step = float(feature.get("step", 1.0) or 1.0)
        range_text = ""
        if min_value is not None and max_value is not None:
            range_text = f" • range {float(min_value):.4g} .. {float(max_value):.4g}"
        st.caption(f"Choice {idx} of {total} • numeric{range_text}")
        st.session_state.setdefault(key, default)
        current = st.session_state.get(key, default)
        if min_value is None or max_value is None:
            entered = st.number_input(
                "value",
                value=float(current),
                step=step,
                key=key,
                label_visibility="collapsed",
            )
        else:
            entered = st.number_input(
                "value",
                min_value=float(min_value),
                max_value=float(max_value),
                value=float(current),
                step=step,
                key=key,
                label_visibility="collapsed",
            )
        values.append(float(entered))
    return values


def _find_firing_example(
    *,
    dataset_path: Path,
    model_path: Path | None = None,
    clf: DSClassifierMultiQ | None = None,
    combine_rule: str = "dempster",
    limit: int = 1000,
) -> dict[str, Any]:
    def _result(
        *,
        status: str,
        message: str | None = None,
        row_index: int | None = None,
        feature_names: list[str] | None = None,
        value_decoders: dict[str, Any] | None = None,
        sample_values: list[float] | None = None,
        active_rule_count: int | None = None,
        combined: dict[str, Any] | None = None,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": status,
            "message": message or "",
        }
        if error_type:
            payload["error_type"] = error_type
        if row_index is not None:
            payload["row_index"] = int(row_index)
        if feature_names is not None:
            payload["feature_names"] = list(feature_names)
        if value_decoders is not None:
            payload["value_decoders"] = dict(value_decoders)
        if sample_values is not None:
            payload["sample_values"] = [float(v) for v in sample_values]
        if active_rule_count is not None:
            payload["active_rule_count"] = int(active_rule_count)
        if combined is not None:
            payload["combined"] = dict(combined)
        return payload

    try:
        X, y, feature_names, value_decoders, stats = load_dataset(dataset_path, return_stats=True)
        del y, stats
        if clf is None:
            if model_path is None:
                return _result(status="error", message="No classifier or model path was provided.", error_type="missing_model")
            clf = load_classifier(model_path)
        if clf.model is None:
            return _result(status="error", message="Loaded classifier does not contain a model.", error_type="missing_model")
        X = np.asarray(X, dtype=np.float32)
        if len(X) == 0:
            return _result(status="empty", message=f"Dataset {dataset_path} has no rows.")
        X_t = clf.model._prepare_numeric_tensor(X)  # type: ignore[union-attr]
        act = clf.model._activation_matrix(X_t)  # type: ignore[union-attr]
        counts = np.asarray(act.sum(dim=1).detach().cpu().numpy(), dtype=int)
        active_rows = np.flatnonzero(np.asarray(counts > 0, dtype=bool))
        if active_rows.size == 0:
            return _result(
                status="empty",
                message="No dataset rows activated any rules for the selected model and combination rule.",
                feature_names=list(feature_names),
                value_decoders=dict(value_decoders),
            )
        if active_rows.size > 0:
            ordered = active_rows[np.argsort(-counts[active_rows])]
            for idx in ordered[: max(1, min(int(limit), len(ordered)))]:
                sample = np.asarray(X[int(idx)], dtype=np.float32)
                combined = clf.model.get_combined_rule(  # type: ignore[union-attr]
                    sample,
                    return_details=True,
                    combination_rule=str(combine_rule),
                    include_merged_rule=True,
                    merged_rule_beta=1.0,
                )
                if isinstance(combined, dict) and str(combined.get("combined_condition", "")).strip():
                    return _result(
                        status="ok",
                        row_index=int(idx),
                        feature_names=list(feature_names),
                        value_decoders=dict(value_decoders),
                        sample_values=sample.tolist(),
                        active_rule_count=int(counts[int(idx)]),
                        combined=combined,
                    )
            idx = int(ordered[0])
            sample = np.asarray(X[idx], dtype=np.float32)
            combined = clf.model.get_combined_rule(  # type: ignore[union-attr]
                sample,
                return_details=True,
                combination_rule=str(combine_rule),
                include_merged_rule=True,
                merged_rule_beta=1.0,
            )
            message = "Found an active row, but no non-empty combined condition was produced."
            return _result(
                status="partial",
                message=message,
                row_index=int(idx),
                feature_names=list(feature_names),
                value_decoders=dict(value_decoders),
                sample_values=sample.tolist(),
                active_rule_count=int(counts[idx]),
                combined=combined if isinstance(combined, dict) else {},
            )
    except Exception as exc:
        return _result(
            status="error",
            message=f"Could not inspect firing example: {type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
        )
    return _result(status="empty", message="No firing example was found.")


def load_classifier(model_path: Path) -> DSClassifierMultiQ:
    clf = DSClassifierMultiQ(k=2)
    clf.load_model(str(model_path))
    return clf


def render_rule_table(activated_rules: list[dict[str, Any]]) -> None:
    if not activated_rules:
        st.info("No rules fired for this sample.")
        return
    rows: list[dict[str, Any]] = []
    for item in activated_rules:
        mass = list(item.get("mass") or [])
        rows.append(
            {
                "rule_id": item.get("id"),
                "class": item.get("class"),
                "condition": item.get("condition"),
                "omega": round(float(mass[-1]), 6) if mass else None,
                "mass_vector": json.dumps([round(float(v), 6) for v in mass]),
                "support": item.get("stats", {}).get("support"),
                "precision": item.get("stats", {}).get("precision"),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def render_literal_weights(combined: dict[str, Any]) -> None:
    literals = list(combined.get("combined_summary_literals") or [])
    if not literals:
        st.info("No merged literals were produced for this sample.")
        return
    rows: list[dict[str, Any]] = []
    for item in literals:
        rows.append(
            {
                "literal": str(item.get("expression", "")),
                "weight": round(float(item.get("confidence", 0.0)), 6),
                "contributors": " | ".join(str(x) for x in list(item.get("contributors") or [])),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _extract_prediction_sections(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if "combined_rule" in payload and "activation" in payload:
        prediction_payload = dict(payload)
        validation = {}
    else:
        prediction_payload = dict(payload.get("prediction", {}))
        validation = dict(payload.get("validation", {}))
    prediction = dict(prediction_payload.get("prediction", {}))
    combined = dict(prediction_payload.get("combined_rule", {}))
    activation = dict(prediction_payload.get("activation", {}))
    return prediction, combined, activation, validation


def render_prediction_summary(payload: dict[str, Any], validation: dict[str, Any] | None = None) -> None:
    prediction, combined, activation, payload_validation = _extract_prediction_sections(payload)
    validation = dict(validation or {})
    if payload_validation:
        validation = payload_validation

    st.markdown("### Result")
    cols = st.columns(4)
    cols[0].metric("Fired rules", str(activation.get("n_rules_fired", 0)))
    cols[1].metric("Predicted class", str(prediction.get("selected_class_original", "n/a")))
    cols[2].metric("Uncertainty", f"{float(prediction.get('unc_comb') or 0.0):.4f}" if prediction.get("unc_comb") is not None else "n/a")
    cols[3].metric("Rule pool", str(combined.get("rules_pool_count", 0)))

    warnings = list(validation.get("warnings", []))
    for warning in warnings:
        st.warning(warning)

    fired = list(prediction.get("activated_rules", []))
    if fired:
        st.markdown("#### Fired Rules")
        render_rule_table(fired)
    else:
        st.info("No rules fired for this sample.")

    st.markdown("#### Final Combined Rule")
    combined_condition = str(combined.get("combined_condition", "")).strip()
    if not combined_condition:
        literals = list(combined.get("combined_summary_literals") or [])
        combined_condition = " AND ".join(str(row.get("expression", "")).strip() for row in literals if str(row.get("expression", "")).strip())
    st.code(combined_condition or "(no combined condition)", language="text")
    if combined.get("summary_line"):
        st.caption("Combined rule summary")
        st.code(str(combined.get("summary_line")), language="text")
    st.markdown("#### Literal Weights")
    render_literal_weights(combined)
    fused_mass = combined.get("fused_mass")
    if fused_mass is None:
        fused_mass = prediction.get("masses")
    if fused_mass is not None:
        st.markdown("#### Mass Vector")
        st.code(json.dumps(fused_mass, indent=2), language="json")
    if combined.get("combined_summary"):
        st.markdown("#### Human-Readable Rule")
        st.code(str(combined.get("combined_summary")), language="text")


def _app_train_main(argv: list[str]) -> int:
    return int(benchmark_main(argv))


def _parse_sample_json(raw: str, schema: dict[str, Any]) -> list[Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--sample-json must decode to an object keyed by feature name.")
    features = list(schema.get("features", []))
    if not features:
        raise ValueError("Could not infer feature schema for the provided model/dataset.")
    values: list[Any] = []
    for feature in features:
        name = str(feature.get("name", ""))
        if name not in payload:
            raise ValueError(f"Missing feature in --sample-json: {name}")
        values.append(payload[name])
    return values


def _app_inspect_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="User-facing inspection CLI for a trained DST model.")
    parser.add_argument("--model", required=True, help="Path to trained .pkl model.")
    parser.add_argument("--dataset", default="", help="Optional dataset path for row-based inspection and schema inference.")
    parser.add_argument("--row-index", type=int, default=None, help="Inspect a row from the provided dataset.")
    parser.add_argument("--split", default="full", help="Label shown in the payload for row-based inspection.")
    parser.add_argument("--combine-rule", default="dempster", choices=["dempster", "yager", "vote"])
    parser.add_argument("--sample-json", default="", help="Manual sample as JSON object keyed by feature name.")
    args = parser.parse_args(argv)

    model_path = Path(args.model).expanduser().resolve()
    clf = load_classifier(model_path)
    dataset_path = Path(args.dataset).expanduser().resolve() if str(args.dataset).strip() else None

    schema = infer_schema_from_csv(dataset_path) if dataset_path and dataset_path.exists() else infer_schema_from_model(clf)
    feature_names = list(schema.get("feature_names") or [])
    value_decoders = dict(schema.get("value_decoders") or {})

    if args.row_index is not None:
        if dataset_path is None or not dataset_path.exists():
            raise SystemExit("--dataset is required and must exist when using --row-index.")
        X, proc_feature_names, proc_value_decoders = load_processed_dataset_rows(dataset_path)
        idx = int(args.row_index)
        if idx < 0 or idx >= len(X):
            raise SystemExit(f"--row-index {idx} is out of range for dataset with {len(X)} rows.")
        payload = build_web_inspection_payload(
            clf=clf,
            sample_values=np.asarray(X[idx], dtype=float).tolist(),
            feature_names=proc_feature_names,
            value_decoders=proc_value_decoders,
            combine_rule=args.combine_rule,
            dataset_name=dataset_path.stem,
            algo=str(getattr(clf, "rule_algo", "RIPPER")),
            model_path=str(model_path),
        )
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default))
        sys.stdout.write("\n")
        return 0

    if not str(args.sample_json).strip():
        raise SystemExit("Provide either --row-index with --dataset or --sample-json for manual inspection.")
    sample_values = _parse_sample_json(args.sample_json, schema)
    encoded = encode_manual_values(sample_values, schema)
    payload = build_web_inspection_payload(
        clf=clf,
        sample_values=encoded,
        feature_names=feature_names,
        value_decoders=value_decoders,
        combine_rule=args.combine_rule,
        dataset_name=(dataset_path.stem if dataset_path else "manual"),
        algo=str(getattr(clf, "rule_algo", "RIPPER")),
        model_path=str(model_path),
    )
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default))
    sys.stdout.write("\n")
    return 0


def _maybe_run_cli() -> int | None:
    if len(sys.argv) < 2:
        return None
    subcommand = str(sys.argv[1]).strip().lower()
    args = sys.argv[2:]
    if subcommand == "train":
        return _app_train_main(args)
    if subcommand == "inspect":
        return _app_inspect_main(args)
    return None


cli_exit = _maybe_run_cli()
if cli_exit is not None:
    raise SystemExit(cli_exit)
if st is None:
    raise ModuleNotFoundError("streamlit is required to run the web UI; CLI subcommands work without it.")


def render_bundle_summary(bundle_path: Path, model_path: Path | None, rules_path: Path | None, *, key_suffix: str) -> None:
    st.caption("Artifact bundle keeps the trained weights and readable rules together.")
    cols = st.columns(3)
    cols[0].write(f"Bundle: `{bundle_path.name}`")
    cols[1].write(f"Model: `{model_path.name if model_path else 'n/a'}`")
    cols[2].write(f"Rules: `{rules_path.name if rules_path else 'n/a'}`")
    st.download_button(
        "Download This Bundle",
        data=bundle_path.read_bytes(),
        file_name=bundle_path.name,
        mime="application/zip",
        width="stretch",
        key=f"download_{slugify(bundle_path.stem)}_{slugify(key_suffix)}",
    )


def load_latest_run_or_none() -> dict[str, Any]:
    latest = resolve_latest_run_manifest()
    if latest is None:
        return {}
    manifest = read_json(latest)
    if manifest:
        manifest["manifest_path"] = str(latest)
        manifest["run_dir"] = str(latest.parent)
        return manifest
    return {}


ensure_app_dirs()

st.set_page_config(page_title="DST Trainer", layout="wide")
st.title("DST Rule Trainer")
st.write(
    "Train FOIL/RIPPER rule models on a CSV, keep the learned rules and weights on disk, "
    "and inspect predictions either from saved data or from manually entered feature values."
)

if "last_run_manifest_path" not in st.session_state:
    st.session_state["last_run_manifest_path"] = ""
if "last_run_dir" not in st.session_state:
    st.session_state["last_run_dir"] = ""
if "last_uploaded_schema" not in st.session_state:
    st.session_state["last_uploaded_schema"] = {}

tab_train, tab_inspect = st.tabs(["Train", "Inspect"])

with tab_train:
    st.subheader("Train On A CSV")
    uploaded = st.file_uploader("Dataset CSV", type=["csv"])
    st.caption(
        "Training results are written to persistent app storage under `results/raw_runs/app_sessions/`, "
        "so a page refresh does not erase the learned rules or weights."
    )
    inducers = st.multiselect("Inducers", ["RIPPER", "FOIL", "STATIC"], default=["RIPPER", "FOIL"])
    max_epochs = st.number_input("Max epochs", min_value=1, max_value=500, value=100)
    test_size = st.number_input("Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    seed = st.number_input("Seed", min_value=0, max_value=100000, value=42)
    paper_mode = st.checkbox("Paper mode", value=False)

    if st.button("Run Training", type="primary", width="stretch"):
        if uploaded is None:
            st.error("Upload a CSV file first.")
        elif not inducers:
            st.error("Select at least one inducer.")
        else:
            dataset_name = Path(uploaded.name).stem
            run_dir = RUNS_DIR / run_id_for(dataset_name)
            run_dir.mkdir(parents=True, exist_ok=False)
            uploads_dir = run_dir / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            results_root = run_dir / "results"
            results_root.mkdir(parents=True, exist_ok=True)

            dataset_path = uploads_dir / uploaded.name
            dataset_path.write_bytes(uploaded.getbuffer())

            st.session_state["last_run_dir"] = str(run_dir)
            st.session_state["last_uploaded_schema"] = {}

            cmd = [
                sys.executable,
                str(APP_ENTRY),
                "train",
                "--dataset-path",
                str(dataset_path),
                "--inducers",
                ",".join(inducers),
                "--save-root",
                str(results_root),
                "--max-epochs",
                str(int(max_epochs)),
                "--test-size",
                str(float(test_size)),
                "--seeds",
                str(int(seed)),
            ]
            if paper_mode:
                cmd.append("--paper-mode")
            protocol = protocol_from_cli(raw_seeds=str(int(seed)), raw_test_size=float(test_size), paper_mode=bool(paper_mode))

            status_placeholder = st.empty()
            log_placeholder = st.empty()
            status_placeholder.info("Training in progress...")
            code, output = run_cmd_stream(cmd, log_placeholder, status_placeholder)

            log_path = run_dir / "training.log"
            log_path.write_text(output or "", encoding="utf-8")

            st.subheader("Training Log")
            st.code(output or "(no output)", language="text")
            if code != 0:
                st.error("Training failed.")
            else:
                st.success("Training finished.")

                schema = infer_schema_from_csv(dataset_path)
                st.session_state["last_uploaded_schema"] = schema
                schema_path = run_dir / "schema.json"
                write_json(schema_path, schema)

                benchmark_dir = results_root / "benchmarks"
                benchmark_files = sorted(benchmark_dir.glob("*.csv"))
                benchmark_preview_path = benchmark_files[0] if benchmark_files else None
                if benchmark_preview_path is not None:
                    st.subheader("Benchmark Preview")
                    preview = read_csv_preview(benchmark_preview_path)
                    if preview:
                        st.code(preview, language="csv")

                model_entries: list[dict[str, Any]] = []
                model_dir = results_root / "models"
                rules_dir = results_root / "rules"
                model_files = sorted(model_dir.glob("*.pkl"))
                if model_files:
                    st.subheader("Generated Artifacts")
                for model_file in model_files:
                    match = re.match(r"^(?P<algo>[^_]+)_(?P<dataset>.+)_dst$", model_file.stem)
                    algo = match.group("algo").upper() if match else "MODEL"
                    dataset_stem = match.group("dataset") if match else dataset_name
                    rule_path = rules_dir / f"{dataset_stem}__{algo}.dsb"
                    bundle_path = make_model_bundle(
                        run_dir=run_dir,
                        dataset_name=dataset_stem,
                        algo=algo,
                        model_path=model_file,
                        rules_path=rule_path if rule_path.exists() else None,
                        schema=schema,
                        benchmark_preview_path=benchmark_preview_path,
                        log_path=log_path,
                    )
                    entry = {
                        "algo": algo,
                        "model_path": str(model_file.relative_to(run_dir)),
                        "rules_path": str(rule_path.relative_to(run_dir)) if rule_path.exists() else None,
                        "bundle_path": str(bundle_path.relative_to(run_dir)),
                    }
                    model_entries.append(entry)
                    with st.expander(f"{algo} bundle", expanded=(len(model_files) == 1)):
                        render_bundle_summary(
                            bundle_path,
                            model_file,
                            rule_path if rule_path.exists() else None,
                            key_suffix=f"generated_{algo}_{dataset_stem}",
                        )
                        if rule_path.exists():
                            st.subheader("Rules")
                            st.code(read_text_preview(rule_path), language="text")

                manifest_path = create_run_manifest(
                    run_dir=run_dir,
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    inducers=[str(x).upper() for x in inducers],
                    protocol_mode=protocol.mode_label,
                    test_size=float(protocol.test_size),
                    seed=int(protocol.seeds[0]),
                    paper_mode=bool(paper_mode),
                    schema=schema,
                    model_entries=model_entries,
                )
                store_latest_run(manifest_path)
                st.session_state["last_run_manifest_path"] = str(manifest_path)
                st.session_state["last_run_dir"] = str(run_dir)

                st.info(
                    "The run has been stored on disk. Refreshing the page will keep these artifacts available "
                    "under the saved run list in the Inspect tab."
                )

                full_run_zip = zip_dir_bytes(run_dir)
                st.download_button(
                    "Download Full Run ZIP",
                    data=full_run_zip,
                    file_name=f"{dataset_name}_dst_run.zip",
                    mime="application/zip",
                    width="stretch",
                )

with tab_inspect:
    st.subheader("Inspect A Saved Model")

    saved_runs = list_saved_runs()
    latest_manifest = load_latest_run_or_none()
    default_run_id = str(latest_manifest.get("run_id", "")) if latest_manifest else ""
    run_ids = [run["run_id"] for run in saved_runs]
    run_label_to_run = {run["run_id"]: run for run in saved_runs}

    source_mode = st.radio(
        "Artifact source",
        ["Latest saved run", "Saved run", "Upload bundle", "Upload files", "Manual paths"],
        index=0,
        horizontal=True,
    )

    selected_run = None
    model_path: Path | None = None
    rules_path: Path | None = None
    schema: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    uploaded_bundle_root: Path | None = None

    if source_mode == "Latest saved run":
        if latest_manifest:
            selected_run = get_run_by_id(str(latest_manifest.get("run_id", "")))
            manifest = load_run_manifest(selected_run) if selected_run else latest_manifest
            schema = read_run_schema(manifest)
        else:
            st.warning("No saved run exists yet. Train a model first.")
    elif source_mode == "Saved run":
        if run_ids:
            chosen_run_id = st.selectbox("Run", run_ids, index=0 if default_run_id not in run_ids else run_ids.index(default_run_id))
            selected_run = run_label_to_run.get(chosen_run_id)
            manifest = load_run_manifest(selected_run)
            schema = read_run_schema(manifest)
        else:
            st.warning("No saved runs found.")
    elif source_mode == "Upload bundle":
        uploaded_bundle = st.file_uploader("Upload a bundle ZIP", type=["zip"])
        if uploaded_bundle is not None:
            uploaded_bundle_root = extract_bundle(uploaded_bundle)
            model_path, rules_path, schema = locate_model_and_rules_from_bundle(uploaded_bundle_root)
            if not schema:
                schema = infer_schema_from_model(load_classifier(model_path)) if model_path else {}
            st.caption(f"Bundle extracted to `{uploaded_bundle_root}`.")
    elif source_mode == "Upload files":
        uploaded_model = st.file_uploader("Upload model (.pkl)", type=["pkl"])
        uploaded_rules = st.file_uploader("Upload readable rules (.dsb)", type=["dsb"])
        if uploaded_model is not None:
            model_path = persist_uploaded_file(uploaded_model, prefix="model")
        if uploaded_rules is not None:
            rules_path = persist_uploaded_file(uploaded_rules, prefix="rules")
        if model_path is not None and not schema:
            schema = infer_schema_from_model(load_classifier(model_path))
    else:
        dataset_path_text = st.text_input("Dataset path", value="")
        model_path_text = st.text_input("Model path (.pkl)", value="")
        rules_path_text = st.text_input("Rules path (.dsb)", value="")
        st.session_state["manual_dataset_path"] = dataset_path_text.strip()
        if dataset_path_text.strip():
            dataset_guess = Path(dataset_path_text.strip()).expanduser()
            if dataset_guess.exists():
                schema = infer_schema_from_csv(dataset_guess)
        if model_path_text.strip():
            model_path = Path(model_path_text.strip()).expanduser()
        if rules_path_text.strip():
            rules_path = Path(rules_path_text.strip()).expanduser()

    if selected_run and not manifest:
        manifest = load_run_manifest(selected_run)
    if manifest and not schema:
        schema = read_run_schema(manifest)

    if not schema and model_path and model_path.exists():
        try:
            schema = infer_schema_from_model(load_classifier(model_path))
        except Exception:
            schema = {}

    st.markdown("### Model Selection")
    algo_guess = "RIPPER"
    if model_path is not None:
        lowered = model_path.name.lower()
        if "foil" in lowered:
            algo_guess = "FOIL"
        elif "static" in lowered:
            algo_guess = "STATIC"
    if manifest and manifest.get("artifacts"):
        artifact_algos = [str(art.get("algo", "MODEL")).upper() for art in manifest.get("artifacts", [])]
        model_algo = st.selectbox(
            "Model artifact",
            artifact_algos,
            index=artifact_algos.index(algo_guess) if algo_guess in artifact_algos else 0,
        )
        chosen_art = locate_run_artifacts(manifest, algo=model_algo)
        if chosen_art.get("model_path"):
            candidate = Path(str(manifest.get("run_dir", ""))) / str(chosen_art["model_path"])
            if candidate.exists():
                model_path = candidate
        if chosen_art.get("rules_path"):
            candidate = Path(str(manifest.get("run_dir", ""))) / str(chosen_art["rules_path"])
            if candidate.exists():
                rules_path = candidate
    else:
        model_algo = st.selectbox("Algorithm", ["STATIC", "RIPPER", "FOIL"], index=["STATIC", "RIPPER", "FOIL"].index(algo_guess))

    if model_path is not None:
        st.code(str(model_path), language="text")
    if rules_path is not None:
        st.code(str(rules_path), language="text")
    combine_rule = st.selectbox("Combine rule", ["dempster", "yager", "vote"], index=0)

    clf: DSClassifierMultiQ | None = None
    if model_path is not None and model_path.exists():
        try:
            clf = load_classifier(model_path)
        except Exception as exc:
            st.warning(f"Could not load model for inspection: {exc}")

    st.markdown("### Inspect Input")
    if schema.get("features"):
        default_values = schema_default_values(schema)
        feature_names = list(schema.get("feature_names") or [])
        value_decoders = dict(schema.get("value_decoders") or {})
        source_dataset = resolve_source_dataset_path(
            manifest=manifest,
            manual_dataset_path=str(st.session_state.get("manual_dataset_path", "")) if source_mode == "Manual paths" else "",
        )
        firing_example: dict[str, Any] = {}
        dataset_rows: np.ndarray | None = None
        dataset_feature_names: list[str] = feature_names
        dataset_value_decoders: dict[str, dict[int, str]] = value_decoders

        if clf is not None and source_dataset is not None and source_dataset.exists():
            try:
                dataset_rows, dataset_feature_names, dataset_value_decoders = load_processed_dataset_rows(source_dataset)
            except Exception as exc:
                st.warning(f"Could not load dataset rows for index-based inspection: {exc}")
                dataset_rows = None
            firing_example = _find_firing_example(
                dataset_path=source_dataset,
                clf=clf,
                combine_rule=combine_rule,
                limit=1200,
            )
            firing_message = str(firing_example.get("message", "")).strip()
            firing_status = str(firing_example.get("status", "")).strip().lower()
            if firing_status in {"error", "partial"} and firing_message:
                st.warning(firing_message)
            if firing_example.get("sample_values"):
                default_values = decode_row_for_manual_input(
                    np.asarray(list(firing_example["sample_values"]), dtype=np.float32),
                    schema,
                )
                st.caption(
                    f"Prefilled with row {firing_example.get('row_index', 'n/a')} "
                    f"that activates {firing_example.get('active_rule_count', 0)} rules."
                )

        inspect_mode = st.radio(
            "Inspection mode",
            ["Manual feature input", "Dataset sample index"],
            index=0,
            horizontal=True,
        )

        if inspect_mode == "Dataset sample index":
            if dataset_rows is None or source_dataset is None:
                st.info("This mode needs the source dataset path from a saved run or manual path.")
            else:
                st.caption(
                    "This uses the processed dataset representation used by training. "
                    "For Dempster, the result is the DSGD-Auto evidential prediction."
                )
                row_limit = max(0, int(dataset_rows.shape[0]) - 1)
                default_row = int(firing_example.get("row_index", 0)) if firing_example else 0
                default_row = min(max(0, default_row), row_limit)
                selected_row = st.number_input(
                    "Sample index",
                    min_value=0,
                    max_value=row_limit,
                    value=default_row,
                    step=1,
                )
                selected_sample = np.asarray(dataset_rows[int(selected_row)], dtype=np.float32)
                preview_rows = []
                decoded_defaults = decode_row_for_manual_input(selected_sample, schema)
                for name in feature_names:
                    preview_rows.append({"feature": name, "value": decoded_defaults.get(name)})
                st.dataframe(pd.DataFrame(preview_rows), width="stretch", hide_index=True)
                sample_cols = st.columns(2)
                if sample_cols[0].button("Load Row Into Manual Form", width="stretch"):
                    for feature in list(schema.get("features", [])):
                        name = str(feature.get("name", "feature"))
                        st.session_state[f"manual__{slugify(name)}"] = decoded_defaults.get(name, feature.get("default", 0.0))
                    st.rerun()
                predict_row_clicked = sample_cols[1].button("Predict This Dataset Row", width="stretch")
                if predict_row_clicked:
                    try:
                        result = build_web_inspection_payload(
                            clf=clf,
                            sample_values=[float(v) for v in selected_sample.tolist()],
                            feature_names=dataset_feature_names,
                            value_decoders=dataset_value_decoders,
                            combine_rule=combine_rule,
                            merged_rule=True,
                            merged_rule_beta=1.0,
                            dataset_name=str(manifest.get("dataset_name") or source_dataset.stem),
                            algo=str(model_algo),
                            model_path=str(model_path),
                            top_k=12,
                        )
                        prediction_payload = dict(result.get("prediction", {}))
                        prediction_inner = dict(prediction_payload.get("prediction", {}))
                        classes = list(schema.get("classes") or [])
                        pred_idx = int(prediction_inner.get("selected_class_idx", -1))
                        pred_label = classes[pred_idx] if 0 <= pred_idx < len(classes) else str(pred_idx)
                        st.success(f"DSGD-Auto prediction for sample {int(selected_row)}: {pred_label}")
                        probs = list(prediction_inner.get("proba", []))
                        if probs:
                            prob_rows = []
                            for idx, p in enumerate(probs):
                                label = classes[idx] if idx < len(classes) else f"class_{idx}"
                                prob_rows.append({"class": label, "probability": round(float(p), 6)})
                            st.dataframe(pd.DataFrame(prob_rows), width="stretch", hide_index=True)
                        render_prediction_summary(result)
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
        else:
            if st.button("Load example that fires rules", width="stretch"):
                for feature in list(schema.get("features", [])):
                    name = str(feature.get("name", "feature"))
                    key = f"manual__{slugify(name)}"
                    if firing_example.get("sample_values"):
                        decoded_defaults = decode_row_for_manual_input(
                            np.asarray(list(firing_example["sample_values"]), dtype=np.float32),
                            schema,
                        )
                        st.session_state[key] = decoded_defaults.get(name, feature.get("default", 0.0))
                    else:
                        st.session_state[key] = feature.get("default", 0.0)
                st.rerun()

            with st.form("manual_predict_form"):
                sample_values = build_raw_row_from_schema(schema, prefill=default_values)
                predict_clicked = st.form_submit_button("Predict from Manual Input", width="stretch")
            if predict_clicked:
                if clf is None or model_path is None or not model_path.exists():
                    st.error("Provide a valid model `.pkl` or select a saved run.")
                else:
                    try:
                        encoded_values = encode_manual_values(sample_values, schema)
                        result = build_web_inspection_payload(
                            clf=clf,
                            sample_values=encoded_values,
                            feature_names=feature_names,
                            value_decoders=value_decoders,
                            combine_rule=combine_rule,
                            merged_rule=True,
                            merged_rule_beta=1.0,
                            dataset_name=str(manifest.get("dataset_name") or "manual"),
                            algo=str(model_algo),
                            model_path=str(model_path),
                            top_k=12,
                        )
                        prediction_payload = dict(result.get("prediction", {}))
                        prediction_inner = dict(prediction_payload.get("prediction", {}))
                        classes = list(schema.get("classes") or [])
                        pred_idx = int(prediction_inner.get("selected_class_idx", -1))
                        pred_label = classes[pred_idx] if 0 <= pred_idx < len(classes) else str(pred_idx)

                        st.success(f"Predicted class: {pred_label}")
                        probs = list(prediction_inner.get("proba", []))
                        if probs:
                            prob_rows = []
                            for idx, p in enumerate(probs):
                                label = classes[idx] if idx < len(classes) else f"class_{idx}"
                                prob_rows.append({"class": label, "probability": round(float(p), 6)})
                            st.dataframe(pd.DataFrame(prob_rows), width="stretch", hide_index=True)
                        render_prediction_summary(result)
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
    else:
        st.info("Load a saved run, upload a bundle, or provide a dataset and model path to enable manual input.")
    if manifest:
        st.markdown("### Saved Run")
        cols = st.columns(4)
        cols[0].write(f"Run: `{manifest.get('run_id', 'n/a')}`")
        cols[1].write(f"Dataset: `{manifest.get('dataset_name', 'n/a')}`")
        cols[2].write(f"Inducers: `{', '.join(manifest.get('inducers', []))}`")
        cols[3].write(f"Created: `{manifest.get('created_at', 'n/a')}`")
        artifacts = list(manifest.get("artifacts", []))
        if artifacts:
            for art in artifacts:
                bundle = Path(str(manifest.get("run_dir", ""))) / str(art.get("bundle_path", ""))
                model = Path(str(manifest.get("run_dir", ""))) / str(art.get("model_path", ""))
                rules = Path(str(manifest.get("run_dir", ""))) / str(art.get("rules_path", "")) if art.get("rules_path") else None
                if bundle.exists():
                    with st.expander(f"{art.get('algo', 'MODEL')} bundle", expanded=False):
                        render_bundle_summary(
                            bundle,
                            model if model.exists() else None,
                            rules if rules and rules.exists() else None,
                            key_suffix=f"stored_{manifest.get('run_id', 'run')}_{art.get('algo', 'model')}",
                        )
