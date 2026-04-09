from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .Datasets_loader import _drop_id_like, _pick_label_column, _read_csv_any, load_dataset
    from .DSClassifierMultiQ import DSClassifierMultiQ
except ImportError:  # pragma: no cover - direct script/import fallback
    from Datasets_loader import _drop_id_like, _pick_label_column, _read_csv_any, load_dataset
    from DSClassifierMultiQ import DSClassifierMultiQ


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
APP_STORE = SRC / "results" / "raw_runs" / "app_sessions"
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


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def zip_dir_bytes(root: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(root))
    buf.seek(0)
    return buf.read()


def read_text_preview(path: Path, max_lines: int = 120) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines: list[str] = []
        for idx, line in enumerate(handle):
            if idx >= max_lines:
                lines.append("...")
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines)


def read_csv_preview(path: Path, max_rows: int = 10) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.reader(handle))
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
        is_object = (
            pd.api.types.is_object_dtype(col)
            or isinstance(col.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(col)
        )
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
                {"name": name, "kind": "categorical", "choices": choices, "default": default}
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
                {"name": name, "kind": "numeric", "min": None, "max": None, "default": 0.0, "step": 1.0}
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


def resolve_decoder_display(decoder: dict[Any, Any], raw_value: Any) -> Any:
    if not decoder:
        return raw_value
    candidates: list[Any] = [raw_value, str(raw_value)]
    try:
        f_value = float(raw_value)
        candidates.extend([f_value])
        if abs(f_value - round(f_value)) <= 1e-9:
            i_value = int(round(f_value))
            candidates.extend([i_value, str(i_value)])
    except Exception:
        pass
    for candidate in candidates:
        if candidate in decoder:
            return decoder[candidate]
    return raw_value


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


def schema_default_values(schema: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for feature in list(schema.get("features", [])):
        name = str(feature.get("name", "feature"))
        defaults[name] = feature.get("default", 0.0)
    return defaults


def resolve_source_dataset_path(*, manifest: dict[str, Any], manual_dataset_path: str) -> Path | None:
    if manifest.get("dataset_path"):
        path = Path(str(manifest.get("dataset_path", ""))).expanduser()
        return path if path.exists() else None
    dataset_guess = str(manual_dataset_path).strip()
    if dataset_guess:
        path = Path(dataset_guess).expanduser()
        return path if path.exists() else None
    return None


def load_processed_dataset_rows(dataset_path: Path) -> tuple[np.ndarray, list[str], dict[str, dict[int, str]]]:
    X, _y, feature_names, value_decoders, _stats = load_dataset(dataset_path, return_stats=True)
    return np.asarray(X, dtype=np.float32), list(feature_names), dict(value_decoders)


def decode_row_for_manual_input(sample: np.ndarray, schema: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    feature_names = list(schema.get("feature_names") or [])
    value_decoders = dict(schema.get("value_decoders") or {})
    for idx, name in enumerate(feature_names):
        value = float(sample[idx])
        decoder = value_decoders.get(name) or {}
        if decoder:
            defaults[name] = resolve_decoder_display(decoder, value)
        else:
            defaults[name] = value
    return defaults


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
        candidate = bundle_root / str(manifest["model_path"])
        if candidate.exists():
            model_path = candidate
    if manifest.get("rules_path"):
        candidate = bundle_root / str(manifest["rules_path"])
        if candidate.exists():
            rules_path = candidate
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
    protocol_mode: str,
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
        "protocol_mode": str(protocol_mode),
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
