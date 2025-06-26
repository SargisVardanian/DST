# DSRipper.py
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np

Mask = np.ndarray
RuleCond = Dict[str, Tuple[str, float]]  # {feature_name: (op, threshold)}


class DSRipper:
    """
    Minimal FOIL/RIPPER-style rule inducer.

    Core ideas
    ----------
    * Greedy rule growth using FOIL gain, with a mild bias towards rare (minority) classes.
    * Optional pruning by Reduced Error Pruning (REP) when `algo="ripper"`.
    * Numeric features use quantile-based thresholds; categorical features use equality tests.

    After `fit(...)`
    -----------------
    * ruleset: Dict[int, List[RuleCond]]      # per-class list of induced rules
    * _ordered_rules: List[Tuple[int, RuleCond]]
      A flat ordered list of (class, rule) pairs in discovery order;
      ends with a default rule: (default_label, {}).
    * _default_label: int                     # default predicted class

    Notes
    -----
    * This is intentionally simple: few (fixed) internal knobs, no external hyperparams
      besides `algo`. Good as a lightweight baseline or seed generator for downstream DST.
    """

    def __init__(self, *, algo: str = "ripper", **_ignored):
        """
        Parameters
        ----------
        algo : {"ripper", "foil"}
            * "foil": grow-only (no pruning).
            * "ripper": grow + REP prune.
        """
        self.algo = algo.lower()
        assert self.algo in {"ripper", "foil"}

        # Internal settings (kept simple and fixed on purpose)
        self._MAX_LITERALS = 4              # max literals per rule
        self._MAX_FAILURES = 128            # limit futile grow attempts per class
        self._MIN_PREC = 0.25               # allow risky rules, but not too noisy
        self._MIN_GAIN = 1e-6               # minimal FOIL-gain to accept a literal
        self._JACCARD_MAX = 0.95            # drop near-duplicates by positive Jaccard
        self._QUANTILES = tuple({           # candidate numeric thresholds (global + class-conditional)
            0.05, 0.10, 0.20, 0.30, 0.40,
            0.50, 0.60, 0.70, 0.80, 0.90,
            0.95
        })
        self._MINORITY_FIRST = True         # learn rules for minority classes first

        # State (set in fit)
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._feat: List[str] = []          # feature names
        self._is_cat: List[bool] = []       # per-feature categorical flag
        self.ruleset: Dict[int, List[RuleCond]] = defaultdict(list)
        self._ordered_rules: List[Tuple[int, RuleCond]] = []
        self._default_label: Optional[int] = None
        self._seen = set()                  # dedup signature cache

    # ------------------------- utilities -------------------------

    def _is_categorical(self, col: np.ndarray) -> bool:
        """
        Heuristic: treat as categorical if dtype is object/str OR
        the number of unique values is small (<= 20).
        """
        if col.dtype == object or col.dtype.kind in {"U", "S"}:
            return True
        vals = np.unique(col[~np.isnan(col)]) if col.dtype.kind == "f" else np.unique(col)
        return vals.size <= 20

    def _literal_mask(self, j: int, op: str, thr: float) -> Mask:
        """
        Build a boolean mask for a single literal on feature j.
        For categorical features, the operator is treated as equality.
        For numeric features, only '>' and '<' are used.
        """
        c = self._X[:, j]
        if self._is_cat[j]:
            return (c == thr)
        return (c > thr) if op == ">" else (c < thr)

    @staticmethod
    def _precision(pos: Mask, neg: Mask, mk: Mask) -> float:
        """Precision among covered examples."""
        p = int((mk & pos).sum()); n = int((mk & neg).sum())
        return p / (p + n + 1e-12)

    @staticmethod
    def _jaccard_pos(covered_pos: Mask, mk_pos: Mask) -> float:
        """Jaccard similarity over positives (for dedup/near-duplicate filtering)."""
        inter = int((covered_pos & mk_pos).sum())
        uni = int((covered_pos | mk_pos).sum())
        return inter / max(1, uni)

    def _candidates_for_feature(self, j: int, pos_idx: Mask) -> List[Tuple[str, float]]:
        """
        Generate candidate literals for feature j:
        * Categorical → equality to each observed category value.
        * Numeric → a small set of thresholds from global and class-conditional quantiles,
                    each producing both ">" and "<" literals.
        """
        xj = self._X[:, j]
        if self._is_cat[j]:
            return [("==", float(v)) for v in np.unique(xj)]

        qs = np.array(sorted(self._QUANTILES), dtype=float)
        # global and class-conditional thresholds
        gq = np.quantile(xj, qs)
        pq = np.quantile(xj[pos_idx], qs) if pos_idx.any() else np.array([])
        thr = np.unique(np.concatenate([gq, pq]).astype(float))

        out: List[Tuple[str, float]] = []
        for t in thr:
            out.append((">", float(t)))
            out.append(("<", float(t)))
        return out

    # ------------------------- scoring (FOIL gain) -------------------------

    def _foil_gain(
        self,
        pos: Mask,
        neg: Mask,
        cm: Mask,
        lm: Mask,
        rule_len: int,
        w_pos: float
    ) -> float:
        """
        FOIL gain for adding literal mask `lm` on top of current mask `cm`.

        We use log-precision and:
          * upweight rare positives via w_pos,
          * mild complexity regularization by 1/(rule_len + 1.25).

        Returns -inf if no positives remain after adding the literal.
        """
        p0 = int((cm & pos).sum()); n0 = int((cm & neg).sum())
        new = cm & lm
        p1 = int((new & pos).sum()); n1 = int((new & neg).sum())
        if p1 == 0:
            return -math.inf

        def _log_prec(p, n) -> float:
            return math.log((p + 1e-9) / (p + n + 1e-9))

        gain = p1 * (_log_prec(p1, n1) - _log_prec(p0, n0))
        gain *= max(1.0, w_pos)         # emphasize rare positives
        gain /= (rule_len + 1.25)       # very light length regularization
        return gain

    # ------------------------- grow / prune -------------------------

    def _grow_rule(self, pos: Mask, neg: Mask, w_pos: float) -> Tuple[Optional[RuleCond], Optional[Mask]]:
        """
        Greedily add literals that maximize FOIL gain until:
          * max literal count reached,
          * no literal improves the gain sufficiently,
          * negatives are (almost) eliminated.
        Returns (rule_condition, covered_mask) or (None, None) if growth failed.
        """
        cond: RuleCond = {}
        curr = np.ones_like(pos, dtype=bool)

        while len(cond) < self._MAX_LITERALS:
            best_gain = -math.inf
            best = None
            for j, a in enumerate(self._feat):
                if a in cond:  # only one literal per feature
                    continue
                for op, thr in self._candidates_for_feature(j, pos):
                    lm = self._literal_mask(j, op, thr)
                    g = self._foil_gain(pos, neg, curr, lm, len(cond), w_pos)
                    if g > self._MIN_GAIN and g > best_gain:
                        best_gain, best = g, (a, op, float(thr), lm)

            if best is None:
                break

            a, op, v, lm = best
            cond[a] = (op, v)
            curr &= lm

            # stop once we nearly eliminated negatives
            if not (curr & neg).any():
                break

        if not cond:
            return None, None

        # sanity filter on train: require at least one positive and a bit of precision
        prec = self._precision(pos, neg, curr)
        if int((curr & pos).sum()) == 0 or prec < self._MIN_PREC:
            return None, None

        return cond, curr

    def _prune_rule(self, pos: Mask, neg: Mask, cond: RuleCond) -> RuleCond:
        """
        Reduced Error Pruning (REP): iteratively remove literals
        if precision does not degrade. Only used for algo="ripper".
        """
        if self.algo != "ripper" or cond is None or len(cond) <= 1:
            return cond

        best = cond.copy()
        mk_full = np.ones_like(pos, dtype=bool)
        for a, (op, v) in best.items():
            mk_full &= self._literal_mask(self._feat.index(a), op, v)
        best_prec = self._precision(pos, neg, mk_full)

        improved = True
        while improved and len(best) > 1:
            improved = False
            for rm_attr in list(best.keys()):
                test = {a: c for a, c in best.items() if a != rm_attr}
                mk = np.ones_like(pos, dtype=bool)
                for a, (op, v) in test.items():
                    mk &= self._literal_mask(self._feat.index(a), op, v)
                prec = self._precision(pos, neg, mk)
                if prec >= best_prec - 1e-12:
                    best, best_prec, improved = test, prec, True
                    break
        return best

    def _rules_for_class(self, cls: int, pos_all: Mask, neg_all: Mask) -> List[RuleCond]:
        """
        Induce multiple rules for a given class by repeatedly growing a rule
        and removing the covered positives (negatives remain to encourage diversity).
        """
        res: List[RuleCond] = []
        fails = 0

        # weight for rare classes: more encouragement to cover positives
        n_pos = int(pos_all.sum()); n_neg = int(neg_all.sum())
        w_pos = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0

        covered_pos = np.zeros_like(pos_all, dtype=bool)  # track coverage to avoid near-duplicates

        pos, neg = pos_all.copy(), neg_all.copy()
        while pos.any() and fails < self._MAX_FAILURES:
            cand, mk_tr = self._grow_rule(pos, neg, w_pos)
            if cand is None:
                fails += 1
                continue

            if self.algo == "ripper":
                cand = self._prune_rule(pos_all, neg_all, cand)

            # recompute mask after pruning
            mk = np.ones_like(pos, dtype=bool)
            for a, (op, v) in cand.items():
                mk &= self._literal_mask(self._feat.index(a), op, v)

            # drop near-duplicates by positive Jaccard similarity
            if self._jaccard_pos(covered_pos, mk & pos_all) > self._JACCARD_MAX:
                fails += 1
                continue

            res.append(cand)
            covered_pos |= (mk & pos_all)

            # remove covered positives; keep negatives to allow more rules
            pos = pos & ~mk

        return res

    # --------------------------- public API ---------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, *, feature_names: List[str]):
        """
        Learn a rule set for each class.

        Parameters
        ----------
        X : (N, M) ndarray
            Feature matrix (numeric-encoded; categorical values can be codes).
        y : (N,) ndarray
            Integer class labels.
        feature_names : list of str
            Names for the features (used in conditions and signatures).

        Returns
        -------
        self
        """
        self._X, self._y = X, y
        self._feat = feature_names.copy()
        self._is_cat = [self._is_categorical(X[:, j]) for j in range(X.shape[1])]
        self.ruleset.clear()
        self._ordered_rules.clear()
        self._seen.clear()

        classes = np.unique(y)
        if self._MINORITY_FIRST:
            classes = sorted(classes, key=lambda c: (y == c).sum())

        for cls in classes:
            pos_all, neg_all = (y == cls), (y != cls)
            for cond in self._rules_for_class(int(cls), pos_all, neg_all):
                # simple signature for deduplication across classes
                sig = "|".join(f"{k}{v[0]}{v[1]}" for k, v in sorted(cond.items()))
                if sig in self._seen:
                    continue
                self._seen.add(sig)
                self.ruleset[int(cls)].append(cond)
                self._ordered_rules.append((int(cls), cond))

        # Default rule: majority among the remaining examples (if any), otherwise global majority
        covered = np.zeros(len(y), dtype=bool)
        for _, cond in self._ordered_rules:
            mk = np.ones(len(y), dtype=bool)
            for a, (op, v) in cond.items():
                mk &= self._literal_mask(self._feat.index(a), op, v)
            covered |= mk

        rem = y[~covered]
        self._default_label = int(np.bincount(rem.astype(int)).argmax()) if rem.size else int(np.bincount(y.astype(int)).argmax())
        self._ordered_rules.append((self._default_label, {}))
        return self
