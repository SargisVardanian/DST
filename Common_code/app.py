from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
COMMON = ROOT / "Common_code"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from Datasets_loader import _drop_id_like, _pick_label_column, _read_csv_any, load_dataset  # noqa: E402
from DSClassifierMultiQ import DSClassifierMultiQ  # noqa: E402
from sample_rule_inspector import build_web_inspection_payload, summarize_rule_generation  # noqa: E402

RUNNER = COMMON / "test_Ripper_DST.py"

APP_STORE = ROOT / "artifacts" / "app_sessions"
RUNS_DIR = APP_STORE / "runs"
UPLOADS_DIR = APP_STORE / "uploads"
EXTRACTED_DIR = APP_STORE / "extracted"
LATEST_POINTER = APP_STORE / "latest_run.json"


def ensure_app_dirs() -> None:
    for path in (APP_STORE, RUNS_DIR, UPLOADS_DIR, EXTRACTED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def slugify(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "run"


def run_id_for(dataset_name: str) -> str:
    return f"{utc_stamp()}__{slugify(dataset_name)}"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_cmd(args: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout


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


def zip_dir_bytes(root: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(root))
    buf.seek(0)
    return buf.read()


def zip_files_bytes(entries: list[tuple[Path, str]], *, metadata: dict[str, Any] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, arcname in entries:
            if file_path.exists():
                zf.write(file_path, arcname=arcname)
        if metadata is not None:
            zf.writestr("manifest.json", json.dumps(metadata, indent=2, ensure_ascii=False))
    buf.seek(0)
    return buf.read()


def read_text_preview(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines: list[str] = []
        for idx, line in enumerate(f):
            if idx >= max_lines:
                lines.append("...")
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines)


def read_csv_preview(path: Path, max_rows: int = 10) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    rows = rows[:max_rows]
    return "\n".join(",".join(row) for row in rows)


def latest_run_pointer() -> Path:
    return LATEST_POINTER


def store_latest_run(manifest_path: Path) -> None:
    write_json(latest_run_pointer(), {"manifest_path": str(manifest_path.resolve())})


def resolve_latest_run_manifest() -> Path | None:
    data = read_json(latest_run_pointer())
    raw = str(data.get("manifest_path", "")).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_file() else None


def list_saved_runs() -> list[dict[str, Any]]:
    ensure_app_dirs()
    runs: list[dict[str, Any]] = []
    for manifest_path in sorted(RUNS_DIR.glob("*/manifest.json"), reverse=True):
        manifest = read_json(manifest_path)
        if not manifest:
            continue
        runs.append(
            {
                "run_id": manifest.get("run_id") or manifest_path.parent.name,
                "manifest_path": str(manifest_path),
                "run_dir": str(manifest_path.parent),
                "created_at": manifest.get("created_at", ""),
                "dataset_name": manifest.get("dataset_name", manifest_path.parent.name),
                "model_count": len(manifest.get("artifacts", [])),
            }
        )
    return runs


def load_run_manifest(run: dict[str, Any] | None) -> dict[str, Any]:
    if not run:
        return {}
    manifest_path = Path(str(run.get("manifest_path", "")))
    manifest = read_json(manifest_path)
    if manifest:
        manifest["manifest_path"] = str(manifest_path)
        manifest["run_dir"] = str(manifest_path.parent)
    return manifest


def get_run_by_id(run_id: str) -> dict[str, Any] | None:
    for run in list_saved_runs():
        if str(run.get("run_id")) == str(run_id):
            return run
    return None


def read_run_schema(run_manifest: dict[str, Any]) -> dict[str, Any]:
    schema_path = Path(str(run_manifest.get("schema_path", "")))
    if schema_path.exists():
        return read_json(schema_path)
    return {}


def infer_schema_from_csv(csv_path: Path) -> dict[str, Any]:
    df = _read_csv_any(csv_path)
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    df = _drop_id_like(df)
    label_col = _pick_label_column(df)
    X_df = df.drop(columns=[label_col]).copy()
    X, y, feature_names, value_decoders, stats = load_dataset(csv_path, return_stats=True)
    del X, y

    features: list[dict[str, Any]] = []
    for name in feature_names:
        col = X_df[name]
        series = col.dropna()
        unique = list(pd.unique(series))
        is_object = pd.api.types.is_object_dtype(col) or pd.api.types.is_categorical_dtype(col) or pd.api.types.is_bool_dtype(col)
        is_numeric = pd.api.types.is_numeric_dtype(col) and not is_object
        numeric_values = None
        if is_numeric and not series.empty:
            try:
                numeric_values = series.astype(float).to_numpy(dtype=float, copy=False)
            except Exception:
                numeric_values = None

        integerish = False
        if numeric_values is not None and numeric_values.size:
            integerish = bool(np.all(np.isclose(numeric_values, np.round(numeric_values))))

        if is_object or (is_numeric and integerish and len(unique) <= 12):
            choices = [str(v) for v in unique]
            default = choices[0] if choices else ""
            features.append(
                {
                    "name": name,
                    "kind": "categorical",
                    "choices": choices,
                    "default": default,
                }
            )
            continue

        arr = series.astype(float).to_numpy(dtype=float, copy=False) if not series.empty else np.asarray([0.0], dtype=float)
        mn = float(np.nanmin(arr)) if arr.size else 0.0
        mx = float(np.nanmax(arr)) if arr.size else 0.0
        med = float(np.nanmedian(arr)) if arr.size else 0.0
        step = 1.0 if integerish else max((mx - mn) / 100.0, 0.01)
        features.append(
            {
                "name": name,
                "kind": "numeric",
                "min": mn,
                "max": mx,
                "default": med if np.isfinite(med) else 0.0,
                "step": float(step),
            }
        )

    return {
        "dataset_name": csv_path.stem,
        "label_column": label_col,
        "feature_names": list(feature_names),
        "classes": list(stats.get("classes", [])),
        "distribution": list(stats.get("distribution", [])),
        "shape": list(stats.get("shape", [])) if stats.get("shape") else None,
        "value_decoders": value_decoders,
        "features": features,
    }


def infer_schema_from_model(clf: DSClassifierMultiQ) -> dict[str, Any]:
    feature_names = list(clf.model.feature_names or []) if clf.model is not None else []
    value_names = dict(clf.model.value_names or {}) if clf.model is not None else {}
    features: list[dict[str, Any]] = []
    for name in feature_names:
        dec = value_names.get(name) or {}
        if dec:
            ordered = [dec[k] for k in sorted(dec.keys(), key=lambda x: (str(type(x)), str(x)))]
            features.append(
                {
                    "name": name,
                    "kind": "categorical",
                    "choices": [str(v) for v in ordered],
                    "default": str(ordered[0]) if ordered else "",
                }
            )
        else:
            features.append(
                {
                    "name": name,
                    "kind": "numeric",
                    "min": None,
                    "max": None,
                    "default": 0.0,
                    "step": 1.0,
                }
            )
    return {
        "dataset_name": "unknown",
        "label_column": None,
        "feature_names": feature_names,
        "classes": [],
        "distribution": [],
        "shape": None,
        "value_decoders": value_names,
        "features": features,
    }


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


def encode_manual_values(sample_values: list[Any], schema: dict[str, Any]) -> list[float]:
    encoded: list[float] = []
    features = list(schema.get("features", []))
    value_decoders = dict(schema.get("value_decoders") or {})
    for feature, raw_value in zip(features, sample_values):
        kind = str(feature.get("kind", "numeric"))
        if kind == "categorical":
            feature_name = str(feature.get("name", ""))
            decoder = value_decoders.get(feature_name) or {}
            reverse = {str(v): float(k) for k, v in decoder.items()}
            key = str(raw_value)
            if key in reverse:
                encoded.append(float(reverse[key]))
                continue
            try:
                encoded.append(float(raw_value))
                continue
            except Exception:
                encoded.append(0.0)
                continue
        try:
            encoded.append(float(raw_value))
        except Exception:
            encoded.append(0.0)
    return encoded


def _schema_default_values(schema: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for feature in list(schema.get("features", [])):
        name = str(feature.get("name", "feature"))
        defaults[name] = feature.get("default", 0.0)
    return defaults


def _find_firing_example(
    *,
    dataset_path: Path,
    model_path: Path | None = None,
    clf: DSClassifierMultiQ | None = None,
    combine_rule: str = "dempster",
    limit: int = 1000,
) -> dict[str, Any]:
    try:
        X, y, feature_names, value_decoders, stats = load_dataset(dataset_path, return_stats=True)
        del y, stats
        if clf is None:
            if model_path is None:
                return {}
            clf = load_classifier(model_path)
        if clf.model is None:
            return {}
        X = np.asarray(X, dtype=np.float32)
        if len(X) == 0:
            return {}
        X_t = clf.model._prepare_numeric_tensor(X)  # type: ignore[union-attr]
        act = clf.model._activation_matrix(X_t)  # type: ignore[union-attr]
        counts = np.asarray(act.sum(dim=1).detach().cpu().numpy(), dtype=int)
        active_rows = np.flatnonzero(np.asarray(counts > 0, dtype=bool))
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
                    return {
                        "row_index": int(idx),
                        "feature_names": list(feature_names),
                        "value_decoders": dict(value_decoders),
                        "sample_values": [float(v) for v in sample.tolist()],
                        "active_rule_count": int(counts[int(idx)]),
                        "combined": combined,
                    }
            idx = int(ordered[0])
            sample = np.asarray(X[idx], dtype=np.float32)
            combined = clf.model.get_combined_rule(  # type: ignore[union-attr]
                sample,
                return_details=True,
                combination_rule=str(combine_rule),
                include_merged_rule=True,
                merged_rule_beta=1.0,
            )
            return {
                "row_index": int(idx),
                "feature_names": list(feature_names),
                "value_decoders": dict(value_decoders),
                "sample_values": [float(v) for v in sample.tolist()],
                "active_rule_count": int(counts[idx]),
                "combined": combined if isinstance(combined, dict) else {},
            }
    except Exception:
        return {}
    return {}


def load_classifier(model_path: Path) -> DSClassifierMultiQ:
    clf = DSClassifierMultiQ(k=2)
    clf.load_model(str(model_path))
    return clf


def locate_run_artifacts(run_manifest: dict[str, Any], algo: str | None = None) -> dict[str, Any]:
    artifacts = list(run_manifest.get("artifacts", []))
    if not artifacts:
        return {}
    if algo:
        algo_u = str(algo).upper()
        for art in artifacts:
            if str(art.get("algo", "")).upper() == algo_u:
                return dict(art)
    return dict(artifacts[0])


def extract_bundle(uploaded_file) -> Path:
    ensure_app_dirs()
    stamp = utc_stamp()
    name = slugify(Path(uploaded_file.name).stem)
    dest = EXTRACTED_DIR / f"{name}__{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    bundle_path = dest / uploaded_file.name
    bundle_path.write_bytes(uploaded_file.getbuffer())
    if bundle_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(bundle_path, "r") as zf:
            zf.extractall(dest / "extracted")
        return dest / "extracted"
    return dest


def persist_uploaded_file(uploaded_file, *, prefix: str) -> Path:
    ensure_app_dirs()
    stamp = utc_stamp()
    dest_dir = EXTRACTED_DIR / f"{slugify(prefix)}__{slugify(Path(uploaded_file.name).stem)}__{stamp}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / uploaded_file.name
    dest_path.write_bytes(uploaded_file.getbuffer())
    return dest_path


def locate_model_and_rules_from_bundle(bundle_root: Path) -> tuple[Path | None, Path | None, dict[str, Any]]:
    manifest_path = bundle_root / "manifest.json"
    schema_path = bundle_root / "schema.json"
    manifest = read_json(manifest_path)
    model_path = None
    rules_path = None
    if manifest.get("model_path"):
        p = bundle_root / str(manifest["model_path"])
        if p.exists():
            model_path = p
    if manifest.get("rules_path"):
        p = bundle_root / str(manifest["rules_path"])
        if p.exists():
            rules_path = p
    if model_path is None:
        candidates = sorted(bundle_root.rglob("*.pkl"))
        if candidates:
            model_path = candidates[0]
    if rules_path is None:
        candidates = sorted(bundle_root.rglob("*.dsb"))
        if candidates:
            rules_path = candidates[0]
    schema = read_json(schema_path) if schema_path.exists() else {}
    return model_path, rules_path, schema


def parse_rule_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        if "|| masses:" in line:
            left, masses = line.split("|| masses:", 1)
        else:
            left, masses = line, ""
        rows.append(
            {
                "line": line,
                "rule": left.strip(),
                "masses": masses.strip(),
            }
        )
    return rows


def predict_row_from_dataset(
    *,
    dataset_path: Path,
    model_path: Path,
    row_index: int,
    split: str,
    combine_rule: str,
) -> tuple[int, str]:
    cmd = [
        sys.executable,
        str(COMMON / "sample_rule_inspector.py"),
        "--dataset",
        str(dataset_path),
        "--model",
        str(model_path),
        "--row-index",
        str(int(row_index)),
        "--split",
        str(split),
        "--combine-rule",
        str(combine_rule),
    ]
    return run_cmd(cmd)


def make_model_bundle(
    *,
    run_dir: Path,
    dataset_name: str,
    algo: str,
    model_path: Path,
    rules_path: Path | None,
    schema: dict[str, Any],
    benchmark_preview_path: Path | None,
    log_path: Path | None,
) -> Path:
    bundle_dir = run_dir / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / f"{slugify(dataset_name)}__{slugify(algo)}.zip"
    manifest = {
        "dataset_name": dataset_name,
        "algo": algo,
        "model_path": "model.pkl",
        "rules_path": "rules.dsb" if rules_path and rules_path.exists() else None,
        "schema_path": "schema.json",
    }
    entries = [(model_path, "model.pkl")]
    if rules_path and rules_path.exists():
        entries.append((rules_path, "rules.dsb"))
    if benchmark_preview_path and benchmark_preview_path.exists():
        entries.append((benchmark_preview_path, "benchmark_preview.csv"))
    if log_path and log_path.exists():
        entries.append((log_path, "training.log"))
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, arcname in entries:
            zf.write(file_path, arcname=arcname)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        zf.writestr("schema.json", json.dumps(schema, indent=2, ensure_ascii=False))
    return bundle_path


def create_run_manifest(
    *,
    run_dir: Path,
    dataset_name: str,
    dataset_path: Path,
    inducers: list[str],
    test_size: float,
    seed: int,
    paper_mode: bool,
    schema: dict[str, Any],
    model_entries: list[dict[str, Any]],
) -> Path:
    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "inducers": inducers,
        "test_size": float(test_size),
        "seed": int(seed),
        "paper_mode": bool(paper_mode),
        "schema_path": str((run_dir / "schema.json").resolve()),
        "artifacts": model_entries,
        "schema": schema,
    }
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


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
    st.dataframe(df, use_container_width=True, hide_index=True)


def _mapping_to_df(mapping: dict[str, Any], key_name: str, value_name: str) -> pd.DataFrame:
    rows = [{"bucket": key, value_name: value} for key, value in mapping.items()]
    return pd.DataFrame(rows, columns=["bucket", value_name]).rename(columns={"bucket": key_name})


def render_generation_summary(generation: dict[str, Any]) -> None:
    if not generation:
        st.info("Rule generation summary is not available for this model.")
        return

    st.markdown("### How Rules Were Generated")
    cols = st.columns(5)
    cols[0].metric("Rules", str(generation.get("n_rules", "n/a")))
    cols[1].metric("Classes", str(generation.get("n_classes", "n/a")))
    cols[2].metric("Features", str(generation.get("n_features", "n/a")))
    cols[3].metric("Pool shaping", "yes" if generation.get("pool_shaping_used") else "no")
    cols[4].metric("Combination", str(generation.get("combination_rule", "n/a")).upper())

    algo = str(generation.get("algo", "MODEL")).upper()
    if algo == "STATIC":
        st.caption(
            "Static induction enumerates candidate literals and combines them into short pair/triple rules. "
            "The diversity filter then removes near-duplicates."
        )
    elif algo == "FOIL":
        st.caption(
            "FOIL grows class-specific rules by adding literals that improve the positive/negative balance. "
            "After growth, the pool keeps a mix of useful but not overly redundant rules."
        )
    else:
        st.caption(
            "RIPPER grows rules in class-wise rounds, then optionally reshapes the pool to keep the final set "
            "balanced across depth, class coverage, novelty, and overlap."
        )
    st.info(
        "Diversity is not random. The pool-scoring step prefers high-quality proposals, then rewards underrepresented "
        "depths and classes, adds novelty, and penalizes overlap. That gives the model enough freedom to keep useful "
        "rule variants without collapsing into duplicates."
    )

    if generation.get("stage_counts"):
        st.markdown("#### Stage Mix")
        st.dataframe(
            _mapping_to_df(dict(generation.get("stage_counts") or {}), "stage", "count"),
            use_container_width=True,
            hide_index=True,
        )
    if generation.get("depth_counts"):
        st.markdown("#### Depth Mix")
        st.dataframe(
            _mapping_to_df(dict(generation.get("depth_counts") or {}), "depth_bin", "count"),
            use_container_width=True,
            hide_index=True,
        )
    if generation.get("class_counts"):
        st.markdown("#### Class Mix")
        st.dataframe(
            _mapping_to_df(dict(generation.get("class_counts") or {}), "class", "count"),
            use_container_width=True,
            hide_index=True,
        )

    top_support = pd.DataFrame(generation.get("top_rules_by_support") or [])
    if not top_support.empty:
        st.markdown("#### Strongest Rules By Support")
        keep_cols = [c for c in ["rule_id", "class", "stage", "depth_bin", "support", "precision", "recall", "f1", "literals", "pool_score", "proposal_quality", "caption"] if c in top_support.columns]
        st.dataframe(top_support[keep_cols], use_container_width=True, hide_index=True)

    top_precision = pd.DataFrame(generation.get("top_rules_by_precision") or [])
    if not top_precision.empty:
        st.markdown("#### Strongest Rules By Precision")
        keep_cols = [c for c in ["rule_id", "class", "stage", "depth_bin", "support", "precision", "recall", "f1", "literals", "pool_score", "proposal_quality", "caption"] if c in top_precision.columns]
        st.dataframe(top_precision[keep_cols], use_container_width=True, hide_index=True)


def render_prediction_summary(payload: dict[str, Any], validation: dict[str, Any]) -> None:
    prediction = dict(payload.get("prediction", {}))
    combined = dict(payload.get("combined_rule", {}))
    activation = dict(payload.get("activation", {}))

    st.markdown("### Activation And Combination")
    cols = st.columns(4)
    cols[0].metric("Fired rules", str(activation.get("n_rules_fired", 0)))
    cols[1].metric("Predicted class", str(prediction.get("selected_class_original", "n/a")))
    cols[2].metric("Uncertainty", f"{float(prediction.get('unc_comb') or 0.0):.4f}" if prediction.get("unc_comb") is not None else "n/a")
    cols[3].metric("Checks", "passed" if validation.get("checks", {}).get("all_passed") else "review")

    checks = dict(validation.get("checks", {}))
    if checks:
        checks_rows = [{"check": key, "value": value} for key, value in checks.items()]
        st.dataframe(pd.DataFrame(checks_rows), use_container_width=True, hide_index=True)

    warnings = list(validation.get("warnings", []))
    for warning in warnings:
        st.warning(warning)

    fired = list(prediction.get("activated_rules", []))
    if fired:
        st.markdown("#### Fired Rules")
        render_rule_table(fired)
    else:
        st.info("No rules fired for this sample.")

    st.markdown("#### Combined Rule")
    st.code(str(combined.get("combined_condition", "")) or "(no combined condition)", language="text")
    if combined.get("fused_mass") is not None:
        st.caption("Fused mass")
        st.code(json.dumps(combined.get("fused_mass"), indent=2), language="json")
    if combined.get("combined_summary"):
        st.caption("Human-readable summary")
        st.code(str(combined.get("combined_summary")), language="text")


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
        use_container_width=True,
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
        "Training results are written to persistent app storage under `artifacts/app_sessions/`, "
        "so a page refresh does not erase the learned rules or weights."
    )
    inducers = st.multiselect("Inducers", ["RIPPER", "FOIL", "STATIC"], default=["RIPPER", "FOIL"])
    max_epochs = st.number_input("Max epochs", min_value=1, max_value=500, value=100)
    test_size = st.number_input("Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    seed = st.number_input("Seed", min_value=0, max_value=100000, value=42)
    paper_mode = st.checkbox("Paper mode", value=False)

    if st.button("Run Training", type="primary", use_container_width=True):
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
            artifacts_root = run_dir / "artifacts"
            artifacts_root.mkdir(parents=True, exist_ok=True)

            dataset_path = uploads_dir / uploaded.name
            dataset_path.write_bytes(uploaded.getbuffer())

            st.session_state["last_run_dir"] = str(run_dir)
            st.session_state["last_uploaded_schema"] = {}

            cmd = [
                sys.executable,
                str(RUNNER),
                "--dataset-path",
                str(dataset_path),
                "--inducers",
                ",".join(inducers),
                "--save-root",
                str(artifacts_root),
                "--max-epochs",
                str(int(max_epochs)),
                "--test-size",
                str(float(test_size)),
                "--seeds",
                str(int(seed)),
            ]
            if not paper_mode:
                cmd.append("--no-paper-mode")

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

                benchmark_dir = artifacts_root / "benchmarks"
                benchmark_files = sorted(benchmark_dir.glob("*.csv"))
                benchmark_preview_path = benchmark_files[0] if benchmark_files else None
                if benchmark_preview_path is not None:
                    st.subheader("Benchmark Preview")
                    preview = read_csv_preview(benchmark_preview_path)
                    if preview:
                        st.code(preview, language="csv")

                model_entries: list[dict[str, Any]] = []
                model_dir = artifacts_root / "models"
                rules_dir = artifacts_root / "rules"
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
                    test_size=float(test_size),
                    seed=int(seed),
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
                    use_container_width=True,
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

    if clf is not None:
        try:
            generation_summary = summarize_rule_generation(clf, top_k=12)
        except Exception as exc:
            generation_summary = {}
            st.warning(f"Could not summarize rule generation: {exc}")
        render_generation_summary(generation_summary)
    else:
        st.info("Load a model to see how the rules were generated and shaped.")

    st.markdown("### Manual Feature Input")
    if schema.get("features"):
        default_values = _schema_default_values(schema)
        firing_example: dict[str, Any] = {}
        if clf is not None:
            source_dataset = None
            if manifest.get("dataset_path"):
                source_dataset = Path(str(manifest.get("dataset_path", ""))).expanduser()
            elif source_mode == "Manual paths":
                dataset_guess = st.session_state.get("manual_dataset_path", "")
                if dataset_guess:
                    source_dataset = Path(str(dataset_guess)).expanduser()
            if source_dataset is not None and source_dataset.exists():
                firing_example = _find_firing_example(
                    dataset_path=source_dataset,
                    clf=clf,
                    combine_rule=combine_rule,
                    limit=1200,
                )
                if firing_example.get("sample_values"):
                    default_values = {
                        name: value
                        for name, value in zip(
                            list(schema.get("feature_names") or []),
                            list(firing_example["sample_values"]),
                        )
                    }
                    st.caption(
                        f"Prefilled with row {firing_example.get('row_index', 'n/a')} "
                        f"that activates {firing_example.get('active_rule_count', 0)} rules."
                    )

        if st.button("Load example that fires rules", use_container_width=True):
            for feature in list(schema.get("features", [])):
                name = str(feature.get("name", "feature"))
                key = f"manual__{slugify(name)}"
                if firing_example.get("sample_values"):
                    idx = list(schema.get("feature_names") or []).index(name)
                    value = firing_example["sample_values"][idx]
                    if str(feature.get("kind", "numeric")) == "categorical":
                        try:
                            value = str(int(round(float(value))))
                        except Exception:
                            value = str(value)
                    st.session_state[key] = value
                else:
                    st.session_state[key] = feature.get("default", 0.0)
            st.rerun()

        with st.form("manual_predict_form"):
            sample_values = build_raw_row_from_schema(schema, prefill=default_values)
            predict_clicked = st.form_submit_button("Predict from Manual Input", use_container_width=True)
        if predict_clicked:
            if clf is None or model_path is None or not model_path.exists():
                st.error("Provide a valid model `.pkl` or select a saved run.")
            else:
                try:
                    encoded_values = encode_manual_values(sample_values, schema)
                    result = build_web_inspection_payload(
                        clf=clf,
                        sample_values=encoded_values,
                        feature_names=list(schema.get("feature_names") or []),
                        value_decoders=dict(schema.get("value_decoders") or {}),
                        combine_rule=combine_rule,
                        merged_rule=True,
                        merged_rule_beta=1.0,
                        dataset_name=str(manifest.get("dataset_name") or "manual"),
                        algo=str(model_algo),
                        model_path=str(model_path),
                        top_k=12,
                    )
                    classes = list(schema.get("classes") or [])
                    pred_idx = int(result.get("prediction", {}).get("selected_class_idx", -1))
                    pred_label = classes[pred_idx] if 0 <= pred_idx < len(classes) else str(pred_idx)

                    st.success(f"Predicted class: {pred_label}")
                    probs = list(result.get("prediction", {}).get("proba", []))
                    if probs:
                        prob_rows = []
                        for idx, p in enumerate(probs):
                            label = classes[idx] if idx < len(classes) else f"class_{idx}"
                            prob_rows.append({"class": label, "probability": round(float(p), 6)})
                        st.dataframe(pd.DataFrame(prob_rows), use_container_width=True, hide_index=True)
                    render_prediction_summary(result, dict(result.get("validation", {})))
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
    else:
        st.info("Load a saved run, upload a bundle, or provide a dataset and model path to enable manual input.")

    st.markdown("### Saved Run Preview")
    if manifest:
        cols = st.columns(4)
        cols[0].write(f"Run: `{manifest.get('run_id', 'n/a')}`")
        cols[1].write(f"Dataset: `{manifest.get('dataset_name', 'n/a')}`")
        cols[2].write(f"Inducers: `{', '.join(manifest.get('inducers', []))}`")
        cols[3].write(f"Created: `{manifest.get('created_at', 'n/a')}`")

        if manifest.get("schema"):
            st.caption("Schema loaded from the saved run manifest.")
            st.dataframe(pd.DataFrame(manifest["schema"].get("features", [])), use_container_width=True, hide_index=True)

        artifacts = list(manifest.get("artifacts", []))
        if artifacts:
            st.markdown("### Stored Bundles")
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
                        if rules and rules.exists():
                            st.code(read_text_preview(rules), language="text")

    st.markdown("### Debug Access")
    st.caption(
        "The old row-index inspector is intentionally hidden from the main workflow. "
        "Use manual feature input for normal inspection, or switch to a saved bundle if you need reproducible inputs."
    )
    with st.expander("Legacy row-index workflow", expanded=False):
        dataset_path = st.text_input("Dataset path for row inspection", value="")
        row_index = st.number_input("Row index", min_value=0, max_value=10_000_000, value=0)
        split = st.selectbox("Split", ["test", "train", "full"], index=0)
        if st.button("Inspect Sample", use_container_width=True):
            if not dataset_path.strip() or model_path is None:
                st.error("Provide a dataset path and a model path.")
            else:
                code, output = predict_row_from_dataset(
                    dataset_path=Path(dataset_path.strip()),
                    model_path=model_path,
                    row_index=int(row_index),
                    split=split,
                    combine_rule=combine_rule,
                )
                st.code(output or "(no output)", language="text")
                if code != 0:
                    st.error("Inspection failed.")
