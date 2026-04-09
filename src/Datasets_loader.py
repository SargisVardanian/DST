"""
Universal dataset loader and simple resampling utilities.

Goals:
  • Auto-detect the label column
  • Drop ID-like columns
  • Integer-encode categorical features while preserving a decoder to strings
  • (Optionally) normalize numeric columns (off by default for rule learners)

Public API
----------
load_dataset(csv_path: str | Path, normalize: bool = False)
    -> tuple[np.ndarray, np.ndarray, list[str], dict]
    Returns (X, y, feature_names, value_decoders)

upsample_minority(X, y, target_pos_ratio=0.35, random_state=42)
    Upsamples the minority class to the desired ratio.
"""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

_DATASET_COLUMN_EXCLUDES: dict[str, set[str]] = {
    # `good` is a target-adjacent helper column derived from wine quality. Keep the
    # benchmark on the raw chemistry features instead of carrying this proxy signal.
    "df_wine": {"good"},
}


def _read_csv_any(csv_path: Path) -> pd.DataFrame:
    """Try common separators (',' and ';') and return a pandas DataFrame."""
    last_err = None
    for sep in (",", ";"):
        try:
            df = pd.read_csv(csv_path, sep=sep, na_values=["?", "NA", "nan"], engine="python")
            if df.shape[1] > 1:
                return df
        except Exception as e:
            last_err = e
    raise ValueError(f"Cannot read {csv_path!s} with ',' or ';' separators: {last_err}")


def _pick_label_column(df: pd.DataFrame) -> str:
    """Heuristically choose the label/target column name."""
    preferred = {"label", "labels", "class", "target", "outcome", "diagnosis", "y", "result", "income", "chd"}
    for c in df.columns:
        if c.strip().lower() in preferred:
            return c

    # Try a single binary non-float column
    candidates = [c for c in df.columns if df[c].nunique(dropna=True) == 2 and df[c].dtype != float]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # pick the most balanced
        def imbalance(col):
            vc = df[col].value_counts(normalize=True, dropna=True)
            p = vc.iloc[0] if not vc.empty else 1.0
            return abs(float(p) - 0.5)
        return min(candidates, key=imbalance)

    # Otherwise pick the last low-cardinality column (<=10 uniques)
    small = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 10]
    if small:
        return small[-1]
    # Fallback: last column
    return df.columns[-1]


def _drop_id_like(df: pd.DataFrame) -> pd.DataFrame:
    """Remove only columns that are very likely to be row identifiers.

    A previous version also dropped any column with fully unique values. That is too
    aggressive for tabular benchmarks because continuous measurements can easily be
    unique without being IDs, which silently corrupts the feature set.
    """
    cols = []
    for c in df.columns:
        name = c.lower().strip()
        series = df[c]
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", name) if tok]
        is_named_id = (
            name in {"id", "uid", "identifier"}
            or name.startswith("id_")
            or name.endswith("_id")
            or "identifier" in tokens
            or "uid" in tokens
            or "id" in tokens
        )
        if is_named_id:
            cols.append(c)
            continue

        if not series.is_unique:
            continue

        # Allow dropping bare row-index style columns such as 0..n-1 or 1..n.
        if pd.api.types.is_integer_dtype(series):
            arr = series.to_numpy()
            expected_zero = np.arange(len(arr), dtype=arr.dtype)
            expected_one = expected_zero + 1
            if np.array_equal(arr, expected_zero) or np.array_equal(arr, expected_one):
                cols.append(c)
    return df.drop(columns=cols) if cols else df


def _drop_dataset_specific_columns(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    excluded = _DATASET_COLUMN_EXCLUDES.get(csv_path.stem.lower())
    if not excluded:
        return df
    present = [col for col in df.columns if col in excluded]
    return df.drop(columns=present) if present else df


def load_dataset(
    csv_path: str | Path,
    *,
    normalize: bool = False,
    return_stats: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, list[str], dict[str, dict[int, str]]]
    | tuple[np.ndarray, np.ndarray, list[str], dict[str, dict[int, str]], dict[str, object]]
):
    """Load a CSV into (X, y, feature_names, value_decoders).

    value_decoders maps a *feature name* to {int_code -> original_string} for categorical
    columns. For numeric comparisons inside rules (>, >=, <, <=), we keep raw floats.
    """
    csv_path = Path(csv_path)
    df = _read_csv_any(csv_path)

    # Drop rows with any missing values to keep rule learning simple
    df = df.dropna(axis=0, how="any").reset_index(drop=True)

    # Remove ID-like columns
    df = _drop_id_like(df)
    df = _drop_dataset_specific_columns(df, csv_path)

    # Detect label column
    label_col = _pick_label_column(df)

    # Factorize labels to y (int), but print original classes for user feedback
    y, classes_ = pd.factorize(df[label_col], sort=True)
    if len(classes_) < 2:
        raise ValueError(f"Too few classes in '{label_col}'")

    # Build X dataframe (drop label)
    X_df = df.drop(columns=[label_col]).copy()

    value_decoders: dict[str, dict[int, str]] = {}
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        cat = pd.Categorical(X_df[c])
        codes = cat.codes.astype(np.float32)
        X_df[c] = codes

        ordered_values = list(cat.categories)
        dec: dict[int, str] = {}
        for code, raw_label in enumerate(ordered_values):
            dec[int(code)] = str(raw_label)
        value_decoders[c] = dec

    # Normalize numeric columns only (optional, off by default)
    if normalize:
        num_cols = X_df.columns[X_df.dtypes.apply(lambda t: np.issubdtype(t, np.number))].tolist()
        if num_cols:
            scaler = StandardScaler()
            X_df[num_cols] = scaler.fit_transform(X_df[num_cols])

    X = X_df.to_numpy(dtype=np.float32, copy=True)
    y = np.asarray(y, dtype=int)
    feature_names = X_df.columns.tolist()

    uniq, cnt = np.unique(y, return_counts=True)
    dist = np.zeros(int(uniq.max()) + 1, dtype=int)
    dist[uniq] = cnt

    stats = {
        "label_column": label_col,
        "classes": classes_.tolist(),
        "shape": (int(X.shape[0]), int(X.shape[1])),
        "distribution": dist.tolist(),
    }

    if return_stats:
        return X, y, feature_names, value_decoders, stats
    return X, y, feature_names, value_decoders


def upsample_minority(
    X: np.ndarray,
    y: np.ndarray,
    *,
    target_pos_ratio: float = 0.35,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Upsample the minority (class==1) to a target ratio; returns (X', y')."""
    X = np.asarray(X); y = np.asarray(y).astype(int)
    pos_mask = (y == 1)
    neg_mask = ~pos_mask
    X_pos, X_neg = X[pos_mask], X[neg_mask]
    y_pos, y_neg = y[pos_mask], y[neg_mask]

    cur = float(y_pos.size) / max(1, y.size)
    if cur >= target_pos_ratio:
        return X, y

    need_pos = int(np.round(target_pos_ratio * y.size / (1 - target_pos_ratio))) - y_pos.size
    if need_pos <= 0:
        return X, y

    X_add, y_add = resample(
        X_pos, y_pos, replace=True, n_samples=need_pos, random_state=random_state
    )
    X_new = np.concatenate([X_neg, X_pos, X_add], axis=0)
    y_new = np.concatenate([y_neg, y_pos, y_add], axis=0)
    return X_new, y_new
