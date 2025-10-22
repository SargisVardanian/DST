# -*- coding: utf-8 -*-
"""
Tunable FOIL/RIPPER rule generator + static rule bundle helpers.

Compatible with DSModelMultiQ.generate_raw():
    from rule_generator import RuleGenerator, generate_static_rule_bundle

Key goals:
- Generate MANY useful rules (recall↑) without wrecking precision;
- Make search space tunable (quantiles, min_precision, max_literals, etc.);
- Track per-rule stats for inspection;
- Provide static (label-free) rule bundles with dense numeric cuts + pairs.

Author: Sargis-ready edition
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np


# ----------------------------------------------------------------------
# Core Rule Generator
# ----------------------------------------------------------------------
class RuleGenerator:
    """
    Simple, tunable FOIL/RIPPER sequential-covering rule inducer.

    Parameters
    ----------
    algo : {'ripper','foil'}
        RIPPER applies reduced-error pruning after growth; FOIL does not.
    verbose : bool
        Print rule growth/pruning events.
    max_literals : int
        Max number of literals allowed in a grown rule.
    min_precision : float
        Global lower bound on precision for accepting a rule.
        (Dynamically tightened relative to class prior during growth.)
    min_gain : float
        Minimal FOIL-gain to accept a literal (rarely binding with dense grids).
    max_failures : int
        Max consecutive failures (no acceptable candidate) before stopping for a class.
    quantiles : Sequence[float] | None
        Quantile grid for numeric thresholds (values in (0,1)). If None → 0.05..0.95 step 0.05.
    jaccard_max : float
        Max Jaccard overlap (on positive coverage) to treat two rules as duplicates.
    minority_first : bool
        If True, induce rules for minority label first.
    max_univariate_rules : int | None
        Upper bound for extra 1-literal rules per class (after sequential covering).
        If None → max(50, 5*#features).
    """

    def __init__(
        self,
        *,
        algo: str = "ripper",
        verbose: bool = False,
        max_literals: int = 5,
        min_precision: float = 0.25,
        min_gain: float = 0.0,
        max_failures: int = 256,
        quantiles: Optional[Sequence[float]] = None,
        jaccard_max: float = 0.90,
        minority_first: bool = True,
        max_univariate_rules: Optional[int] = None,
        **_ignore,
    ) -> None:
        algo = str(algo or "ripper").lower()
        if algo not in {"ripper", "foil"}:
            raise ValueError("algo must be 'ripper' or 'foil'")
        self.algo = algo
        self.verbose = bool(verbose)

        # Knobs
        self._MAX_LITERALS = max(1, int(max_literals))
        self._MAX_FAILURES = max(8, int(max_failures))
        self._MIN_PREC = max(0.05, float(min_precision))
        self._MIN_GAIN = float(min_gain)
        self._JACCARD_MAX = float(jaccard_max)

        if quantiles is None:
            # Dense 5% grid by default
            self._QUANTILES = tuple(np.linspace(0.05, 0.95, 19))
        else:
            q = [float(v) for v in quantiles if 0.0 < float(v) < 1.0]
            if not q:
                raise ValueError("quantiles must contain values strictly between 0 and 1")
            self._QUANTILES = tuple(sorted(set(q)))

        self._MINORITY_FIRST = bool(minority_first)
        self._max_univariate_rules = max_univariate_rules

        # Runtime
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._feat: List[str] = []
        self._is_cat: List[bool] = []
        self.ruleset: Dict[int, List[Dict[str, Tuple[str, float]]]] = defaultdict(list)
        self._ordered_rules: List[Tuple[int, Dict[str, Tuple[str, float]]]] = []
        self._default_label: Optional[int] = None
        self._seen: set[str] = set()
        self._rule_metrics: Dict[str, Dict[str, float]] = {}  # signature -> metrics

    # ------------------------- Utilities -------------------------
    @staticmethod
    def _is_categorical(col: np.ndarray) -> bool:
        # Strings/objects → categorical; numeric with small cardinality → categorical
        if col.dtype.kind in ("U", "S", "O"):
            return True
        if col.dtype.kind != "f":
            return np.unique(col).size <= 20
        vals = col[np.isfinite(col)]
        return np.unique(vals).size <= 20

    def _literal_mask(self, j: int, op: str, thr: float) -> np.ndarray:
        c = self._X[:, j]
        if self._is_cat[j]:
            return c == thr
        if op == ">":
            return c > thr
        if op == ">=":
            return c >= thr
        if op == "<":
            return c < thr
        if op == "<=":
            return c <= thr
        if op == "==":
            return c == thr
        if op == "!=":
            return c != thr
        # Fallback to <= for unknown ops to avoid silent crashes
        return c <= thr

    @staticmethod
    def _precision(pos: np.ndarray, neg: np.ndarray, mk: np.ndarray) -> float:
        p = int((mk & pos).sum())
        n = int((mk & neg).sum())
        return p / (p + n + 1e-12)

    @staticmethod
    def _recall(pos: np.ndarray, mk: np.ndarray) -> float:
        p = int((mk & pos).sum())
        tot = int(pos.sum())
        return p / (tot + 1e-12)

    @staticmethod
    def _jaccard_pos(covered_pos: np.ndarray, mk_pos: np.ndarray) -> float:
        inter = int((covered_pos & mk_pos).sum())
        uni = int((covered_pos | mk_pos).sum())
        return inter / max(1, uni)

    def _candidates_for_feature(self, j: int, pos_idx: np.ndarray):
        xj = self._X[:, j]
        if self._is_cat[j]:
            # equality tests for every seen category (already numeric-encoded in upstream)
            return [("==", float(v)) for v in np.unique(xj)]
        # numeric: grid + class-conditional cuts
        qs = list(self._QUANTILES)
        global_q = np.quantile(xj, qs)
        class_q = np.quantile(xj[pos_idx], qs) if pos_idx.any() else np.array([])
        thr = np.unique(np.concatenate([global_q, class_q]).astype(float))
        out = []
        for t in thr:
            out.append((">", float(t)))
            out.append(("<=", float(t)))
        return out

    def _foil_gain(
        self,
        pos: np.ndarray,
        neg: np.ndarray,
        cm: np.ndarray,
        lm: np.ndarray,
        rule_len: int,
        w_pos: float,
    ) -> float:
        # FOIL gain with mild length penalty and class-imbalance weight
        p0 = int((cm & pos).sum())
        n0 = int((cm & neg).sum())
        new = cm & lm
        p1 = int((new & pos).sum())
        n1 = int((new & neg).sum())
        if p1 == 0:
            return -math.inf

        def log_prec(p, n):
            return math.log((p + 1e-9) / (p + n + 1e-9))

        gain = p1 * (log_prec(p1, n1) - log_prec(p0, n0))
        gain *= max(1.0, w_pos)
        gain /= (rule_len + 1.5)
        return gain

    def _print_rule(
        self,
        cls: int,
        cond: Dict[str, Tuple[str, float]],
        stage: str,
        support: int,
        precision: float,
        recall: float,
        neg_covered: int,
    ) -> None:
        if not self.verbose:
            return
        parts = [f"{a} {op} {v:.6g}" for a, (op, v) in cond.items()]
        body = " & ".join(parts)
        print(
            f"[RuleGenerator][{stage}] class={cls} :: {body}  "
            f"| support={support}  prec={precision:.3f}  rec={recall:.3f}  neg_covered={neg_covered}"
        )

    # ---------------------- Grow / Prune ------------------------
    def _grow_rule(self, pos: np.ndarray, neg: np.ndarray, w_pos: float):
        cond: Dict[str, Tuple[str, float]] = {}
        curr = np.ones_like(pos, dtype=bool)
        while len(cond) < self._MAX_LITERALS:
            best_gain = -math.inf
            best = None
            for j, name in enumerate(self._feat):
                if name in cond:
                    continue
                for op, thr in self._candidates_for_feature(j, pos):
                    lm = self._literal_mask(j, op, thr)
                    g = self._foil_gain(pos, neg, curr, lm, len(cond), w_pos)
                    if g > best_gain and g >= self._MIN_GAIN:
                        best_gain, best = g, (name, op, float(thr), lm)
            if best is None:
                break
            a, op, v, lm = best
            cond[a] = (op, v)
            curr &= lm
            # early stop if we have removed all negatives
            if not (curr & neg).any():
                break

        if not cond:
            return None, None
        # guard: minimal precision
        precision = self._precision(pos, neg, curr)
        if int((curr & pos).sum()) == 0 or precision < self._MIN_PREC:
            return None, None
        return cond, curr

    def _prune_rule(self, pos: np.ndarray, neg: np.ndarray, cond: Dict[str, Tuple[str, float]]):
        if self.algo != "ripper" or len(cond) <= 1:
            return cond
        best = cond.copy()
        # compute mask for a given set of literals
        def mask_of(cnd):
            mk = np.ones_like(pos, dtype=bool)
            for a, (op, v) in cnd.items():
                j = self._feat.index(a)
                mk &= self._literal_mask(j, op, v)
            return mk

        best_mk = mask_of(best)
        best_prec = self._precision(pos, neg, best_mk)
        improved = True
        while improved and len(best) > 1:
            improved = False
            for rm_attr in list(best.keys()):
                trial = {k: v for k, v in best.items() if k != rm_attr}
                mk = mask_of(trial)
                prec = self._precision(pos, neg, mk)
                if prec >= best_prec - 1e-12:
                    best, best_prec, improved = trial, prec, True
                    break
        return best

    # -------------------- Induce per class ----------------------
    def _rules_for_class(self, cls: int, pos_all: np.ndarray, neg_all: np.ndarray):
        res: List[Dict[str, Tuple[str, float]]] = []
        fails = 0
        n_pos = int(pos_all.sum())
        n_neg = int(neg_all.sum())
        w_pos = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0

        covered_pos = np.zeros_like(pos_all, dtype=bool)
        pos, neg = pos_all.copy(), neg_all.copy()
        prior = n_pos / float(max(1, n_pos + n_neg))

        # tighten the acceptance precision wrt prior
        prev_prec = self._MIN_PREC
        self._MIN_PREC = max(self._MIN_PREC, min(0.95, prior + 0.10))

        while pos.any() and fails < self._MAX_FAILURES:
            cand, mk = self._grow_rule(pos, neg, w_pos)
            if cand is None:
                fails += 1
                continue
            if self.algo == "ripper":
                cand = self._prune_rule(pos_all, neg_all, cand)

            # recompute mk after pruning
            mk = np.ones_like(pos, dtype=bool)
            for a, (op, v) in cand.items():
                mk &= self._literal_mask(self._feat.index(a), op, v)

            # filter near-duplicates on positive coverage
            if self._jaccard_pos(covered_pos, mk & pos_all) > self._JACCARD_MAX:
                fails += 1
                continue

            # stats
            p_cov = int((mk & pos_all).sum())
            n_cov = int((mk & neg_all).sum())
            prec = p_cov / max(1, p_cov + n_cov)
            rec = p_cov / max(1, n_pos)
            sig = "|".join(f"{k}{v[0]}{v[1]}" for k, v in sorted(cand.items()))
            self._rule_metrics[sig] = {
                "class": float(cls),
                "support_pos": float(p_cov),
                "support_neg": float(n_cov),
                "precision": float(prec),
                "recall": float(rec),
            }

            res.append(cand)
            covered_pos |= (mk & pos_all)
            self._print_rule(cls, cand, "cover", p_cov, prec, rec, n_cov)
            # remove covered positives for subsequent rules
            pos = pos & ~mk

        self._MIN_PREC = prev_prec
        return res

    # -------------------- Extra univariate rules ----------------
    def _augment_univariate_rules(self):
        X, y = self._X, self._y
        classes = np.unique(y)
        if self._MINORITY_FIRST:
            classes = sorted(classes, key=lambda c: (y == c).sum())
        for cls in classes:
            pos = (y == cls)
            neg = ~pos
            n_pos = int(pos.sum())
            if n_pos == 0:
                continue
            prior = n_pos / float(max(1, len(y)))
            min_precision = max(prior + 0.05, 0.30)  # permissive
            min_support = max(2, int(round(0.003 * n_pos)))
            candidates: List[Tuple[float, float, int, int, Dict[str, Tuple[str, float]]]] = []
            for j, name in enumerate(self._feat):
                feats = self._candidates_for_feature(j, pos)
                if not feats:
                    continue
                for op, thr in feats:
                    mask = self._literal_mask(j, op, thr)
                    pos_cov = int((mask & pos).sum())
                    if pos_cov < min_support:
                        continue
                    neg_cov = int((mask & neg).sum())
                    total = pos_cov + neg_cov
                    if total == 0:
                        continue
                    precision = pos_cov / total
                    if precision < min_precision:
                        continue
                    cond = {name: (op, float(thr))}
                    signature = "|".join(f"{k}{v[0]}{v[1]}" for k, v in sorted(cond.items()))
                    if signature in self._seen:
                        continue
                    recall = pos_cov / float(max(1, n_pos))
                    candidates.append((precision, recall, pos_cov, neg_cov, cond))
            if not candidates:
                continue
            candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            max_uni = self._max_univariate_rules
            if max_uni is None:
                max_uni = max(50, len(self._feat) * 5)
            added = 0
            for precision, recall, support, neg_cov, cond in candidates:
                if added >= max_uni:
                    break
                sig = "|".join(f"{k}{v[0]}{v[1]}" for k, v in sorted(cond.items()))
                if sig in self._seen:
                    continue
                self._seen.add(sig)
                self.ruleset[int(cls)].append(cond)
                self._ordered_rules.append((int(cls), cond))
                # stats for univariate rules
                self._rule_metrics[sig] = {
                    "class": float(cls),
                    "support_pos": float(support),
                    "support_neg": float(neg_cov),
                    "precision": float(precision),
                    "recall": float(recall),
                }
                self._print_rule(int(cls), cond, "univariate", support, precision, recall, neg_cov)
                added += 1

    # -------------------------- API -----------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, *, feature_names: Sequence[str]):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D with same #rows as X")

        self._feat = list(feature_names)
        self._is_cat = [self._is_categorical(X[:, j]) for j in range(X.shape[1])]
        self._X = X
        self._y = y
        self.ruleset.clear()
        self._ordered_rules.clear()
        self._seen.clear()
        self._rule_metrics.clear()

        classes = np.unique(self._y)
        if self._MINORITY_FIRST:
            classes = sorted(classes, key=lambda c: (self._y == c).sum())

        # sequential covering per class
        for cls in classes:
            pos = (self._y == cls)
            neg = ~pos
            for cond in self._rules_for_class(int(cls), pos, neg):
                sig = "|".join(f"{k}{v[0]}{v[1]}" for k, v in sorted(cond.items()))
                if sig in self._seen:
                    continue
                self._seen.add(sig)
                self.ruleset[int(cls)].append(cond)
                self._ordered_rules.append((int(cls), cond))

        # add extra univariate rules
        self._augment_univariate_rules()

        # default rule label for uncovered examples
        covered = np.zeros(len(self._y), dtype=bool)
        for _, cond in self._ordered_rules:
            mk = np.ones(len(self._y), dtype=bool)
            for name, (op, val) in cond.items():
                mk &= self._literal_mask(self._feat.index(name), op, val)
            covered |= mk
        rem = self._y[~covered]
        if rem.size:
            self._default_label = int(np.bincount(rem.astype(int)).argmax())
        else:
            self._default_label = int(np.bincount(self._y.astype(int)).argmax())
        # explicit default "empty" rule at the end for DSModelMultiQ compatibility
        self._ordered_rules.append((self._default_label, {}))
        return self


# ----------------------------------------------------------------------
# Static (label-free) bundles for algo='STATIC'
# ----------------------------------------------------------------------
def _is_categorical_for_static(col: np.ndarray) -> bool:
    if col.dtype.kind in ("U", "S", "O"):
        return True
    if col.dtype.kind != "f":
        return np.unique(col).size <= 20
    finite = col[np.isfinite(col)]
    return np.unique(finite).size <= 20


def generate_statistic_single_rules(
    X: np.ndarray,
    *,
    breaks: int = 3,
    column_names: Optional[Sequence[str]] = None,
    generated_columns: Optional[Iterable[bool]] = None,
    value_decoders: Optional[Dict[str, Dict[int, Any]]] = None,
    top_k_cats: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Produce simple single-literal rules for each feature:
      - Numeric: thresholds at interior quantiles
      - Categorical: equality tests for (up to) top_k_cats categories

    Returns list of dicts with keys: 'specs', 'caption', 'label' (None)
    """
    X = np.asarray(X)
    names = list(column_names) if column_names is not None else [f"X[{i}]" for i in range(X.shape[1])]
    mask = None if generated_columns is None else np.asarray(list(generated_columns), dtype=bool)

    rules: List[Dict[str, Any]] = []
    # interior quantiles only
    q = np.linspace(0.0, 1.0, max(3, breaks + 2))[1:-1]

    for j, name in enumerate(names):
        if mask is not None and not mask[j]:
            continue
        col = X[:, j]
        if _is_categorical_for_static(col):
            # choose most frequent categories
            vals, counts = np.unique(col, return_counts=True)
            order = np.argsort(-counts)
            top = len(vals) if top_k_cats is None else min(int(top_k_cats), len(vals))
            for v in vals[order][:top]:
                caption = f"{name} == {v}"
                rules.append({"specs": ((name, "==", float(v)),), "caption": caption, "label": None})
            continue

        # numeric
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        edges = np.quantile(finite, q)
        edges = np.unique(edges.astype(float))
        for thr in edges:
            rules.append({"specs": ((name, "<=", float(thr)),), "caption": f"{name} <= {thr:.6g}", "label": None})
        # terminal right cut
        rules.append({"specs": ((name, ">", float(edges[-1])),), "caption": f"{name} > {edges[-1]:.6g}", "label": None})

    return rules


def generate_mult_pair_rules(
    X: np.ndarray,
    *,
    include_square: bool = False,
    column_names: Optional[Sequence[str]] = None,
    value_decoders: Optional[Dict[str, Dict[int, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Very cheap pairwise interaction rules at feature means:
        (fi >= mean_i) & (fj >= mean_j), (fi < mean_i) & (fj < mean_j), etc.
    """
    X = np.asarray(X)
    names = list(column_names) if column_names is not None else [f"X[{i}]" for i in range(X.shape[1])]
    mean = np.nanmean(X, axis=0)
    offset = 0 if include_square else 1

    out: List[Dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + offset, len(names)):
            fi, fj = names[i], names[j]
            mi, mj = float(mean[i]), float(mean[j])
            out.append({"specs": ((fi, ">=", mi), (fj, ">=", mj)),
                        "caption": f"{fi} >= {mi:.6g} & {fj} >= {mj:.6g}", "label": None})
            out.append({"specs": ((fi, "<", mi), (fj, "<", mj)),
                        "caption": f"{fi} < {mi:.6g} & {fj} < {mj:.6g}", "label": None})
            out.append({"specs": ((fi, ">=", mi), (fj, "<", mj)),
                        "caption": f"{fi} >= {mi:.6g} & {fj} < {mj:.6g}", "label": None})
            out.append({"specs": ((fi, "<", mi), (fj, ">=", mj)),
                        "caption": f"{fi} < {mi:.6g} & {fj} >= {mj:.6g}", "label": None})
    return out


def generate_static_rule_bundle(
    X: np.ndarray,
    *,
    breaks: int = 5,
    column_names: Optional[Sequence[str]] = None,
    generated_columns: Optional[Iterable[bool]] = None,
    value_decoders: Optional[Dict[str, Dict[int, Any]]] = None,
    top_k_cats: int = 20,
    include_pairs: bool = True,
    pair_top: int = 6,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build an unlabelled rule bundle:
      - Single-literal rules per feature using `breaks` interior quantiles
        (and top-k categorical values),
      - Optional pairwise interactions for the top-variance numeric features.

    Arguments match DSModelMultiQ.generate_raw(...) expectations.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    names = list(column_names) if column_names is not None else [f"X[{i}]" for i in range(X.shape[1])]

    # 1) Singles
    rules = generate_statistic_single_rules(
        X,
        breaks=max(3, int(breaks)),
        column_names=names,
        generated_columns=generated_columns,
        value_decoders=value_decoders,
        top_k_cats=top_k_cats,
    )

    # 2) Pairs on top-variance numeric features
    if include_pairs:
        # choose numeric columns by variance
        variances: List[Tuple[int, float]] = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if _is_categorical_for_static(col):
                continue
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                continue
            variances.append((j, float(np.var(finite))))
        variances.sort(key=lambda p: p[1], reverse=True)
        top_ids = [idx for idx, _ in variances[: max(2, int(pair_top))]]
        if len(top_ids) >= 2:
            rules.extend(
                generate_mult_pair_rules(
                    X[:, top_ids],
                    include_square=False,
                    column_names=[names[i] for i in top_ids],
                    value_decoders=value_decoders,
                )
            )

    if verbose:
        for item in rules:
            cap = str(item.get("caption", "")).strip()
            if cap:
                print(f"[RuleGenerator] emit: {cap}")
    return rules


__all__ = [
    "RuleGenerator",
    "generate_statistic_single_rules",
    "generate_mult_pair_rules",
    "generate_static_rule_bundle",
]
