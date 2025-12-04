"""
Utility helpers used across the DST codebase.

This module keeps tiny, dependency-light helpers together in one place:
- basic array transforms (normalize, one_hot)
- smooth indicator functions (h_center / h_right / h_left)
- breaks/thresholds generation (natural_breaks via KMeans, statistic_breaks via mean±σ)
- simple structure converters (transform_X_to_cases, build_classes_dict)
- tiny type heuristics (is_categorical)
- optional rule-based filtering with a clustering-backed confidence score (filter_by_rule)

Notes
-----
* The public function names are kept stable to avoid breaking existing users.
* Imports are deliberately lean; plotting is optional and executed only if
  Plotly is available and `only_plot=True`.
"""
import logging
from collections.abc import Callable, Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    calinski_harabasz_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

# Keep compatibility with configs that define defaults for this module.
try:
    from config import *  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass

try:
    # Optional plotting dependency, used only when requested.
    import plotly.express as px  # type: ignore
except Exception:  # pragma: no cover
    px = None

# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def is_categorical(arr: np.ndarray, max_cat: int = 20) -> bool:
    """Heuristic: a column is categorical if it has a small number of unique values.

    Parameters
    ----------
    arr : np.ndarray
        1-D array-like column.
    max_cat : int, default=20
        Maximum distinct values to still treat a column as categorical.

    Returns
    -------
    bool
        True if `arr` has <= `max_cat` unique non-NaN values.
    """
    arr = np.asarray(arr)
    return len(np.unique(arr[~np.isnan(arr)])) <= max_cat


def normalize(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Z-normalize a 1-D array, guarding against zero variance."""
    a = np.asarray(a, dtype=float)
    mu = np.nanmean(a)
    sd = np.nanstd(a)
    sd = sd if sd > eps else 1.0
    return (a - mu) / sd


def one_hot(n: Sequence[int] | np.ndarray, k: int) -> np.ndarray:
    """Return a one-hot matrix for integer labels in range [0, k)."""
    n = np.asarray(n, dtype=int).ravel()
    out = np.zeros((len(n), k), dtype=np.float32)
    for i, lab in enumerate(n):
        if 0 <= lab < k:
            out[i, lab] = 1.0
    return out


# smooth indicator bumps based on tanh
def h_center(z: np.ndarray) -> np.ndarray:
    """Symmetric bump around 0: ~1 in the center, ~0 outside."""
    z = np.asarray(z, dtype=float)
    return (1.0 - np.tanh(z) ** 2)


def h_right(z: np.ndarray) -> np.ndarray:
    """Right-sided smooth step."""
    z = np.asarray(z, dtype=float)
    return (1.0 + np.tanh(z - 1.0)) / 2.0


def h_left(z: np.ndarray) -> np.ndarray:
    """Left-sided smooth step."""
    z = np.asarray(z, dtype=float)
    return (1.0 - np.tanh(z + 1.0)) / 2.0


# ---------------------------------------------------------------------------
# Threshold / break generators
# ---------------------------------------------------------------------------

def natural_breaks(data: Iterable[float], k: int = 5, append_infinity: bool = False) -> list[float]:
    """Compute `k` natural breaks using 1-D KMeans on `data`.

    Returns the sorted cluster boundaries. If `append_infinity=True`, the last
    boundary is extended to +inf to simplify "binning" logic elsewhere.
    """
    data = np.asarray(list(data), dtype=float)
    km = KMeans(n_clusters=k, max_iter=150, n_init=5)
    km.fit(data.reshape(-1, 1))

    # The rightmost boundary per cluster
    breaks: list[float] = []
    for i in range(k):
        pts = data[km.labels_ == i]
        if pts.size:
            breaks.append(float(np.max(pts)))
    breaks = sorted(set(breaks))

    if append_infinity and breaks:
        breaks[-1] = float("inf")
    return breaks


def statistic_breaks(arr: Iterable[float], k: int = 5, sigma_tol: float = 2.0) -> np.ndarray:
    """Evenly spaced thresholds over [μ-σ·tol, μ+σ·tol]."""
    arr = np.asarray(list(arr), dtype=float)
    mu, sd = float(np.nanmean(arr)), float(np.nanstd(arr))
    grid = np.linspace(mu - sd * sigma_tol, mu + sd * sigma_tol, k)
    return grid


# ---------------------------------------------------------------------------
# Rule-based filtering with confidence via clustering
# ---------------------------------------------------------------------------

def filter_by_rule(
    df,
    rule_lambda: Callable,
    lower_confidence_by_proportion: bool = True,
    only_plot: bool = False,
    print_results: bool = False,
    label_column: str = "cluster_id",
):
    """Filter rows that satisfy a rule and score confidence via cluster compactness.

    Pipeline
    --------
    1) Apply `rule_lambda(row) -> bool` to mark rows where the rule holds.
    2) On the filtered subset, compute the dominant cluster (by `label_column`).
    3) Confidence = mean normalized distance to the dominant cluster centroid.
       Lower is better → we report `confidence` as the *closeness*:  `1 - mean_dist`.
    4) Optionally scale by the proportion of rows in the dominant cluster.

    Returns
    -------
    tuple
        For a 3-way setting, returns (p0, p1, p2) as a simple split:
        the dominant cluster gets `confidence`, the others share the remainder.
    """
    import pandas as pd  # local import to keep base deps light

    assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
    assert label_column in df.columns, f"'{label_column}' column is required"
    assert "distance_norm" in df.columns, "'distance_norm' column is required"
    assert not df["distance_norm"].isna().any(), "distance_norm contains NaNs"
    assert not np.isinf(df["distance_norm"]).any(), "distance_norm contains inf"

    df = df.copy()
    df["rule_applies"] = df.apply(rule_lambda, axis=1)
    df_rule = df[df["rule_applies"]]

    if only_plot and px is not None:
        fig = px.scatter(df, x="x", y="y", color="rule_applies")
        fig.update_layout(title="Rule filter preview")
        fig.show()

    if df_rule.empty:
        raise ValueError("No rows satisfy the rule.")

    num_clusters = int(df_rule[label_column].nunique())
    if print_results:
        logging.debug("rows after filter=%d, clusters=%d", len(df_rule), num_clusters)

    dominant = int(df_rule[label_column].mode().values[0])

    # Base confidence: closeness to the dominant centroid (lower distance → higher confidence)
    mean_dist = float(df_rule[df_rule[label_column] == dominant]["distance_norm"].mean())
    confidence = 1.0 - mean_dist

    if lower_confidence_by_proportion:
        prop = float((df_rule[label_column] == dominant).mean())
        confidence *= prop

    # Produce a simple 3-prob split for downstream compatibility.
    confidence = float(np.clip(confidence, 0.0, 1.0))
    rest = (1.0 - confidence) / 2.0
    if dominant == 0:
        return confidence, rest, rest
    elif dominant == 1:
        return rest, confidence, rest
    else:
        return rest, rest, confidence


# ---------------------------------------------------------------------------
# Structure converters
# ---------------------------------------------------------------------------

def transform_X_to_cases(X: np.ndarray, column_names: Sequence[str]) -> list[dict[str, list[float]]]:
    """Convert a design matrix X into a list of dict "cases" keyed by feature name.

    Example output (for 3 columns)::
        [{'f1': [x11], 'f2': [x12], 'f3': [x13]}, {'f1': [x21], ...}, ...]
    """
    X = np.asarray(X)
    cases: list[dict[str, list[float]]] = []
    for row in X:
        case = {col: [row[i]] for i, col in enumerate(column_names)}
        cases.append(case)
    return cases


def build_classes_dict(cases: Sequence[dict[str, list[float]]], y: Sequence[int]) -> dict[int, list[dict[str, list[float]]]]:
    """Group cases by integer label array `y`."""
    classes: dict[int, list[dict[str, list[float]]]] = {int(l): [] for l in np.unique(y)}
    for i, case in enumerate(cases):
        classes[int(y[i])].append(case)
    return classes
