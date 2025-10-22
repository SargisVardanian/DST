import math
from collections import defaultdict
import numpy as np


class RuleGenerator:
    """Simple FOIL/RIPPER rule inducer with a few tuning knobs.

    This implementation follows the classic sequential covering approach used
    in FOIL and RIPPER: rules are grown greedily using FOIL gain, optionally
    pruned with reduced error pruning (RIPPER), and collected until all
    positive examples are covered or no further useful rules can be found.

    Numeric features are discretised using a configurable set of quantiles,
    categorical features generate equality tests for each observed category,
    and duplicate or near-duplicate rules are filtered via a positive-class
    Jaccard threshold.  Several constructor arguments allow tweaking the
    inductive bias without rewriting the core algorithm.

    Parameters
    ----------
    algo : str, optional
        One of ``'ripper'`` or ``'foil'``.  In RIPPER mode, learned rules
        undergo reduced error pruning.  In FOIL mode, rules are grown only.
    verbose : bool, optional
        If ``True``, emit diagnostic information to stdout during rule
        generation.
    max_literals : int, optional
        Maximum number of literals that a single grown rule may contain.
    min_precision : float, optional
        Minimum precision threshold a candidate rule must satisfy.  The
        runtime code further tightens this threshold based on the class prior.
    min_gain : float, optional
        Minimum FOIL-gain improvement required to add a literal.
    max_failures : int, optional
        Number of consecutive failed attempts to grow a rule before giving up
        for a class.
    quantiles : sequence of float, optional
        Custom quantiles used for numeric candidate thresholds (values in
        ``(0, 1)``).  When omitted, a default dense grid is used.
    minority_first : bool, optional
        Whether to induce minority-class rules first.
    """

    def __init__(
        self,
        *,
        algo="ripper",
        verbose=False,
        max_literals: int = 4,
        min_precision: float = 0.33,
        min_gain: float = 1e-4,
        max_failures: int = 128,
        quantiles=None,
        minority_first: bool = True,
    ):
        algo = str(algo or "ripper").lower()
        if algo not in {"ripper", "foil"}:
            raise ValueError("algo must be 'ripper' or 'foil'")
        self.algo = algo
        # store verbose flag in both public and private attributes for backward compatibility
        self.verbose = bool(verbose)
        self._verbose = bool(verbose)
        # tunable knobs
        self._MAX_LITERALS = max(1, int(max_literals))
        self._MAX_FAILURES = max(8, int(max_failures))
        self._MIN_PREC = max(0.05, float(min_precision))
        self._MIN_GAIN = float(min_gain)
        self._JACCARD_MAX = 0.95
        self._QUANTILES = tuple(sorted(quantiles)) if quantiles is not None else (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
        self._MINORITY_FIRST = bool(minority_first)
        # runtime state
        self._X = None
        self._y = None
        self._feat = []
        self._is_cat = []
        self.ruleset = defaultdict(list)
        self._ordered_rules = []
        self._default_label = None
        self._seen = set()
        self._min_prec_current = self._MIN_PREC

    # ------------------------------------------------------------------
    # helper for verbose logging
    # ------------------------------------------------------------------
    def _print_rule(self, cls, cond, stage, support, precision, recall, neg_covered):
        """Internal helper to print a rule in a human friendly format.

        Parameters
        ----------
        cls : int
            Class label for which the rule is induced.
        cond : dict
            Mapping of feature names to (operator, threshold) pairs.
        stage : str
            Identifier of the stage (e.g. 'grow', 'prune', 'univariate').
        support : int
            Number of positive examples covered by the rule.
        precision : float
            Precision of the rule on the training set.
        recall : float
            Recall of the rule on the training set.
        neg_covered : int
            Number of negative examples covered by the rule.
        """
        if not getattr(self, "_verbose", False):
            return
        parts = []
        for name, (op, val) in cond.items():
            try:
                display = f"{float(val):.4f}"
            except Exception:
                display = str(val)
            parts.append(f"{name} {op} {display}")
        text = " & ".join(parts) if parts else "<empty>"
        print(
            f"[RuleGenerator] stage={stage} class={cls} literals={len(cond)} support={support} "
            f"precision={precision:.3f} recall={recall:.3f} neg={neg_covered} :: {text}"
        )

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------
    def _is_categorical(self, col):
        if col.dtype.kind in ("U", "S", "O"):
            return True
        if col.dtype.kind != "f":
            return np.unique(col).size <= 20
        vals = col[~np.isnan(col)]
        return len(np.unique(vals)) <= 20

    def _literal_mask(self, j, op, thr):
        c = self._X[:, j]
        if self._is_cat[j]:
            return c == thr
        return c > thr if op == ">" else c < thr

    @staticmethod
    def _precision(pos, neg, mk):
        p = int((mk & pos).sum())
        n = int((mk & neg).sum())
        return p / (p + n + 1e-12)

    @staticmethod
    def _jaccard_pos(covered_pos, mk_pos):
        inter = int((covered_pos & mk_pos).sum())
        uni = int((covered_pos | mk_pos).sum())
        return inter / max(1, uni)

    def _candidates_for_feature(self, j, pos_idx):
        xj = self._X[:, j]
        if self._is_cat[j]:
            return [("==", float(v)) for v in np.unique(xj)]
        qs = sorted(self._QUANTILES)
        global_q = np.quantile(xj, qs)
        class_q = np.quantile(xj[pos_idx], qs) if pos_idx.any() else np.array([])
        thr = np.unique(np.concatenate([global_q, class_q]).astype(float))
        out = []
        for t in thr:
            out.append((">", float(t)))
            out.append(("<", float(t)))
        return out

    def _foil_gain(self, pos, neg, cm, lm, rule_len, w_pos):
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
        gain /= (rule_len + 1.25)
        return gain

    def _grow_rule(self, pos, neg, w_pos):
        cond = {}
        curr = np.ones_like(pos, dtype=bool)
        while len(cond) < self._MAX_LITERALS:
            best_gain = -math.inf
            best = None
            for j, a in enumerate(self._feat):
                if a in cond:
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
            if not (curr & neg).any():
                break
        if not cond:
            return None, None
        prec = self._precision(pos, neg, curr)
        min_prec = getattr(self, "_min_prec_current", self._MIN_PREC)
        if int((curr & pos).sum()) == 0 or prec < min_prec:
            return None, None
        return cond, curr

    def _prune_rule(self, pos, neg, cond):
        if self.algo != "ripper" or not cond or len(cond) <= 1:
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
                trial = {k: v for k, v in best.items() if k != rm_attr}
                mk = np.ones_like(pos, dtype=bool)
                for a, (op, v) in trial.items():
                    mk &= self._literal_mask(self._feat.index(a), op, v)
                prec = self._precision(pos, neg, mk)
                if prec >= best_prec - 1e-12:
                    best, best_prec, improved = trial, prec, True
                    break
        return best

    def _rules_for_class(self, cls, pos_all, neg_all):
        res = []
        fails = 0
        n_pos = int(pos_all.sum())
        n_neg = int(neg_all.sum())
        w_pos = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
        covered_pos = np.zeros_like(pos_all, dtype=bool)
        pos, neg = pos_all.copy(), neg_all.copy()
        prior = n_pos / float(max(1, n_pos + n_neg))
        prev_prec = getattr(self, "_min_prec_current", self._MIN_PREC)
        self._min_prec_current = max(self._MIN_PREC, min(0.95, prior + 0.1))
        while pos.any() and fails < self._MAX_FAILURES:
            cand, mk_tr = self._grow_rule(pos, neg, w_pos)
            if cand is None:
                fails += 1
                continue
            if self.algo == "ripper":
                cand = self._prune_rule(pos_all, neg_all, cand)
            mk = np.ones_like(pos, dtype=bool)
            for a, (op, v) in cand.items():
                mk &= self._literal_mask(self._feat.index(a), op, v)
            if self._jaccard_pos(covered_pos, mk & pos_all) > self._JACCARD_MAX:
                fails += 1
                continue
            res.append(cand)
            covered_pos |= (mk & pos_all)
            pos = pos & ~mk
        self._min_prec_current = prev_prec
        return res

    def _augment_univariate_rules(self):
        if self._X is None or self._y is None:
            return
        X, y = self._X, self._y
        classes = np.unique(y)
        if self._MINORITY_FIRST:
            classes = sorted(classes, key=lambda c: (y == c).sum())
        for cls in classes:
            pos = (y == cls)
            neg = ~pos
            n_pos = int(pos.sum())
            n_neg = int(neg.sum())
            if n_pos == 0:
                continue
            prior = n_pos / float(max(1, n_pos + n_neg))
            min_precision = max(prior + 0.05, 0.40)
            min_support = max(3, int(round(0.005 * n_pos)))
            candidates = []
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
            max_uni = max(50, len(self._feat) * 3)
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
                self._print_rule(int(cls), cond, stage="univariate", support=support, precision=precision, recall=recall, neg_covered=neg_cov)
                added += 1

    def fit(self, X, y, *, feature_names):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be a 1D array aligned with X")
        self._feat = list(feature_names)
        self._is_cat = [self._is_categorical(X[:, j]) for j in range(X.shape[1])]
        self._X = X
        self._y = y
        self.ruleset.clear()
        self._ordered_rules.clear()
        self._seen.clear()
        classes = np.unique(self._y)
        if self._MINORITY_FIRST:
            classes = sorted(classes, key=lambda c: (self._y == c).sum())
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
        covered = np.zeros(len(self._y), dtype=bool)
        for _, cond in self._ordered_rules:
            mk = np.ones(len(self._y), dtype=bool)
            for a, (op, v) in cond.items():
                mk &= self._literal_mask(self._feat.index(a), op, v)
            covered |= mk
        self._augment_univariate_rules()
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
        self._ordered_rules.append((self._default_label, {}))
        return self


def generate_statistic_single_rules(X, *, breaks=3, column_names=None, generated_columns=None, value_decoders=None, top_k_cats=None):
    X = np.asarray(X)
    cols = list(column_names) if column_names is not None else [f"X[{i}]" for i in range(X.shape[1])]
    mask = None if generated_columns is None else np.asarray(list(generated_columns), dtype=bool)

    def is_categorical(col):
        if col.dtype.kind in ("U", "S", "O"):
            return True
        if col.dtype.kind != "f":
            return np.unique(col).size <= 20
        finite = col[np.isfinite(col)]
        return np.unique(finite).size <= 20

    generated = []
    quantiles = np.linspace(0.0, 1.0, max(3, breaks + 2))[1:-1]
    for idx, name in enumerate(cols):
        if mask is not None and not mask[idx]:
            continue
        column = X[:, idx]
        clean = column[~np.isnan(column)]
        if clean.size == 0:
            continue
        if is_categorical(column):
            cats, counts = np.unique(clean, return_counts=True)
            order = np.argsort(-counts)
            if top_k_cats:
                cats = cats[order[:top_k_cats]]
            else:
                cats = cats[order]
            for cat in cats:
                pretty = None
                if value_decoders and name in value_decoders:
                    try:
                        key = int(round(float(cat)))
                        pretty = value_decoders[name].get(key, cat)
                    except Exception:
                        pretty = cat
                if pretty is None:
                    pretty = cat
                generated.append({
                    'specs': ((name, '==', float(cat)),),
                    'caption': f"{name} == {pretty}",
                    'label': None
                })
            continue
        q_vals = np.quantile(clean, quantiles) if quantiles.size else np.array([])
        edges = np.unique(np.concatenate(([float(clean.min())], q_vals, [float(clean.max())])))
        if edges.size < 2:
            continue
        generated.append({
            'specs': ((name, '<=', float(edges[1])),),
            'caption': f"{name} <= {edges[1]:.6g}",
            'label': None
        })
        for left, right in zip(edges[1:-1], edges[2:]):
            if left >= right:
                continue
            caption = f"{left:.6g} < {name} <= {right:.6g}"
            generated.append({
                'specs': ((name, '>', float(left)), (name, '<=', float(right))),
                'caption': caption,
                'label': None
            })
        generated.append({
            'specs': ((name, '>', float(edges[-2])),),
            'caption': f"{name} > {edges[-2]:.6g}",
            'label': None
        })
    return generated


def generate_mult_pair_rules(X, *, include_square=False, column_names=None, value_decoders=None):
    X = np.asarray(X)
    cols = list(column_names) if column_names is not None else [f"X[{i}]" for i in range(X.shape[1])]
    mean = np.nanmean(X, axis=0)
    offset = 0 if include_square else 1
    generated = []
    for i in range(len(cols)):
        for j in range(i + offset, len(cols)):
            fi, fj = cols[i], cols[j]
            mi, mj = float(mean[i]), float(mean[j])
            generated.append({'specs': ((fi, '>=', mi), (fj, '>=', mj)), 'caption': f"{fi} >= {mi:.3f} & {fj} >= {mj:.3f}", 'label': None})
            generated.append({'specs': ((fi, '<', mi), (fj, '<', mj)), 'caption': f"{fi} < {mi:.3f} & {fj} < {mj:.3f}", 'label': None})
            generated.append({'specs': ((fi, '>=', mi), (fj, '<', mj)), 'caption': f"{fi} >= {mi:.3f} & {fj} < {mj:.3f}", 'label': None})
            generated.append({'specs': ((fi, '<', mi), (fj, '>=', mj)), 'caption': f"{fi} < {mi:.3f} & {fj} >= {mj:.3f}", 'label': None})
    return generated


def generate_static_rule_bundle(X, *, breaks=3, column_names=None, generated_columns=None, value_decoders=None, top_k_cats=12, include_pairs=True, pair_top=4, verbose=False):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    names = list(column_names) if column_names is not None else [f"X[{i}]" for i in range(X.shape[1])]
    mask = None if generated_columns is None else np.asarray(list(generated_columns), dtype=bool)
    singles = generate_statistic_single_rules(X, breaks=breaks, column_names=names, generated_columns=mask, value_decoders=value_decoders, top_k_cats=top_k_cats)
    rules = list(singles)
    if not include_pairs:
        return rules
    numeric_idx = [idx for idx, name in enumerate(names) if (mask[idx] if mask is not None else True) and np.issubdtype(X[:, idx].dtype, np.number)]
    if len(numeric_idx) < 2:
        return rules
    variances = sorted(((i, float(np.nanvar(X[:, i])) if np.isfinite(np.nanvar(X[:, i])) else 0.0) for i in numeric_idx), key=lambda item: item[1], reverse=True)
    pair_top = max(2, int(pair_top))
    top_ids = [idx for idx, _ in variances[: min(pair_top, len(variances))]]
    if len(top_ids) < 2:
        return rules
    pairs = generate_mult_pair_rules(X[:, top_ids], include_square=False, column_names=[names[i] for i in top_ids], value_decoders=value_decoders)
    rules.extend(pairs)
    if verbose:
        for item in rules:
            caption = str(item.get("caption", ""))
            if caption:
                print(f"[RuleGenerator] emit: {caption}")
    return rules


__all__ = ["RuleGenerator", "generate_statistic_single_rules", "generate_mult_pair_rules", "generate_static_rule_bundle"]
