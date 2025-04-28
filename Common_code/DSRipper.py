# --------------------------- DSRipper.py ---------------------------
import math, copy, numbers
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Sequence

import numpy as np


class DSRipper:
    """
    RIPPER / FOIL с инкрементным обучением, дедупликацией правил
    и подробным логом по каждому батчу.
    """

    # ----------  INIT ----------
    def __init__(
        self,
        d: int = 64,
        ratio: float = 2 / 3.0,
        k: int = 2,
        min_gain: float = 0.1,
        eq_tol: float = 1e-3,
        max_unique_cat: int = 10,
        algo: str = "ripper",
        min_support: float = 0.0,
        dedup_tol: float = 1e-3,
    ):
        self.d = d
        self.ratio = ratio
        self.k = k
        self.min_gain = min_gain
        self.eq_tol = eq_tol
        self.max_unique = max_unique_cat
        self.algo = algo.lower()
        self.min_support = min_support
        self.dedup_tol = dedup_tol

        # состояние, заполняется при fit()
        self.ruleset: Dict[Any, List[Dict[str, Tuple[str, Any]]]] = defaultdict(list)
        self._seen: set[str] = set()
        self._feature_names: List[str] = []

    # ----------  НИЗКОУРОВНЕВЫЕ ПОМОЩНИКИ ----------
    def _is_close(self, a: float, b: float) -> bool:
        return abs(a - b) <= self.eq_tol

    def _satisfy(self, value: Any, op: str, thr: Any) -> bool:
        if op == "==":
            if isinstance(value, numbers.Number) and isinstance(thr, numbers.Number):
                return self._is_close(value, thr)
            return value == thr
        if op == ">":
            return float(value) > float(thr)
        if op == "<":
            return float(value) < float(thr)
        raise ValueError(f"Unknown operator {op}")

    def rule_satisfied(
        self, case: Dict[str, Sequence[Any]], rule: Dict[str, Tuple[str, Any]]
    ) -> bool:
        for attr, (op, thr) in rule.items():
            if attr not in case:
                return False
            if not any(self._satisfy(v, op, thr) for v in case[attr]):
                return False
        return True

    def _canon(self, rule: Dict[str, Tuple[str, Any]]) -> str:
        """Канонический ID правила (используется для дедупликации)."""
        parts = []
        for attr, (op, thr) in sorted(rule.items()):
            if isinstance(thr, numbers.Number):
                thr = round(thr / self.dedup_tol) * self.dedup_tol
                thr = f"{thr:.6f}".rstrip("0").rstrip(".")
            parts.append(f"{attr}{op}{thr}")
        return "|".join(parts)

    # ----------  СТАТИСТИКА ----------
    def bindings(self, cases: List[Dict], rules) -> int:
        if isinstance(rules, dict):
            rules = [rules]
        return sum(all(self.rule_satisfied(c, r) for r in rules) for c in cases)

    @staticmethod
    def _dl(rule: Dict[str, Tuple[str, Any]], total_possible: int) -> float:
        k = len(rule)
        if k == 0:
            return 0.0
        p = k / float(total_possible)
        if p <= 0 or p >= 1:
            return 0.0
        return 0.5 * (
            math.log2(k)
            + k * math.log2(1.0 / p)
            + (total_possible - k) * math.log2(1.0 / (1.0 - p))
        )

    # ----------  FOIL‑GAIN ----------
    def foil_gain(
        self,
        pos: List[Dict],
        neg: List[Dict],
        literal: Tuple[str, Tuple[str, Any]],
        current_rule: Dict[str, Tuple[str, Any]],
    ) -> float:
        p0 = self.bindings(pos, current_rule) if current_rule else len(pos)
        n0 = self.bindings(neg, current_rule) if current_rule else len(neg)

        new_rule = copy.deepcopy(current_rule)
        attr, (op, thr) = literal
        new_rule[attr] = (op, thr)

        p1 = self.bindings(pos, new_rule)
        n1 = self.bindings(neg, new_rule)

        if p1 == 0:
            return -float("inf")

        i0 = math.log2((p0 + 1e-9) / (p0 + n0 + 1e-9))
        i1 = math.log2((p1 + 1e-9) / (p1 + n1 + 1e-9))

        return p1 * (i1 - i0)

    # ----------  КАНДИДАТЫ ----------
    def _collect_candidates(
        self, pos: List[Dict], rule: Dict[str, Tuple[str, Any]]
    ) -> Tuple[Dict[str, set], Dict[str, bool]]:
        values = defaultdict(set)
        for c in pos:
            for a, vs in c.items():
                if a not in rule:
                    values[a].update(vs)
        cat_flag = {a: len(vs) <= self.max_unique for a, vs in values.items()}
        return values, cat_flag

    # ----------  GROW ----------
    def grow_rule(
        self,
        pos: List[Dict],
        neg: List[Dict],
        base_rule: Optional[Dict[str, Tuple[str, Any]]] = None,
    ) -> Dict[str, Tuple[str, Any]]:
        rule = {} if base_rule is None else copy.deepcopy(base_rule)
        while True:
            candidates, is_cat = self._collect_candidates(pos, rule)
            best_gain, best_lit = -float("inf"), None

            for attr, vals in candidates.items():
                ops = ["==", ">", "<"] if is_cat[attr] else [">", "<"]
                for thr in vals:
                    for op in ops:
                        gain = self.foil_gain(
                            pos, neg, (attr, (op, thr)), rule
                        )
                        if gain > best_gain and gain >= self.min_gain:
                            best_gain, best_lit = gain, (attr, op, thr)

            if best_lit is None:
                break

            a, op, thr = best_lit
            rule[a] = (op, thr)

            # для скорости оставляем только покрытые примеры
            pos = [c for c in pos if self.rule_satisfied(c, rule)]
            neg = [c for c in neg if self.rule_satisfied(c, rule)]

        return rule

    # ----------  PRUNE ----------
    def prune_rule(
        self,
        pos: List[Dict],
        neg: List[Dict],
        rule: Dict[str, Tuple[str, Any]],
        total_cases: int,
    ) -> Dict[str, Tuple[str, Any]]:
        if len(rule) <= 1:
            return rule
        best_rule = copy.deepcopy(rule)
        best_dl = self._dl(best_rule, total_cases)

        improved = True
        while improved and len(best_rule) > 1:
            improved = False
            for attr in list(best_rule.keys()):
                tmp = copy.deepcopy(best_rule)
                tmp.pop(attr)
                dl_tmp = self._dl(tmp, total_cases)
                if dl_tmp < best_dl:
                    best_rule, best_dl = tmp, dl_tmp
                    improved = True
                    break
        return best_rule

    # ----------  IREP ----------
    def irep(self, pos: List[Dict], neg: List[Dict]) -> List[Dict]:
        rules = []
        total = len(pos) + len(neg)
        min_dl_in_set = float("inf")

        while pos:
            rule = self.grow_rule(pos, neg)
            if self.algo == "ripper":
                rule = self.prune_rule(pos, neg, rule, total)

            # проверка поддержки
            support = self.bindings(pos + neg, rule) / total
            if support < self.min_support:
                break

            dl = self._dl(rule, total)
            if rules and self.algo == "ripper" and dl > min_dl_in_set + self.d:
                break

            rules.append(rule)
            min_dl_in_set = min(min_dl_in_set, dl)

            pos = [c for c in pos if not self.rule_satisfied(c, rule)]
            neg = [c for c in neg if not self.rule_satisfied(c, rule)]

        return rules

    # ----------  OPTIMIZE ----------
    def optimize(
        self, pos: List[Dict], neg: List[Dict], rules: List[Dict]
    ) -> List[Dict]:
        if self.algo != "ripper":
            return rules
        total = len(pos) + len(neg)
        for _ in range(self.k):
            new_rules = []
            for r in rules:
                new_r = self.grow_rule(pos, neg, r)
                new_r = self.prune_rule(pos, neg, new_r, total)
                new_rules.append(new_r if self._canon(new_r) != self._canon(r)
                                 and self._dl(new_r, total) < self._dl(r, total)
                                 else r)
            rules = new_rules
        return rules

    # ----------  FIT (инкрементное) ----------
    def _rule_accuracy(self, rule, P, N):
        tp = self.bindings(P, rule)
        fp = self.bindings(N, rule)
        tn = len(N) - fp
        fn = len(P) - tp
        return (tp + tn) / (tp + fp + tn + fn + 1e-9)

    def _process_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ):
        cases = [
            {feature_names[j]: [X[i, j]] for j in range(X.shape[1])}
            for i in range(X.shape[0])
        ]
        by_class = defaultdict(list)
        for case, label in zip(cases, y):
            by_class[label].append(case)

        ordered = sorted(by_class.items(), key=lambda kv: len(kv[1]))
        batch_new_rules = 0

        for label, P in ordered:
            N = [c for l, cs in ordered if l != label for c in cs]

            uncovered = [
                c for c in P
                if not any(self.rule_satisfied(c, r) for r in self.ruleset[label])
            ]
            if not uncovered:
                continue

            new_rules = self.optimize(
                uncovered, N, self.irep(copy.deepcopy(uncovered), copy.deepcopy(N))
            )

            for r in new_rules:
                sig = self._canon(r)
                if sig in self._seen:
                    continue  # правило уже было
                self.ruleset[label].append(r)
                self._seen.add(sig)
                batch_new_rules += 1

                acc = self._rule_accuracy(r, P, N)
                print(f"    + Rule [{self._canon(r)}]  acc={acc:.3f}")

        print(f"  >> Batch done: added {batch_new_rules} new rules\n")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        batch_size: Optional[int] = None,
    ):
        self._feature_names = feature_names
        self.ruleset = defaultdict(list)
        self._seen = set()

        n = len(y)
        if batch_size is None or batch_size >= n:
            batch_size = n

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            print(f"\n--- Batch {start}-{end - 1} ---")
            self._process_batch(X[start:end], y[start:end], feature_names)

        total_rules = sum(len(r) for r in self.ruleset.values())
        print(f"=== Incremental fitting finished: {total_rules} total rules ===")
        return self.ruleset

    # ----------  PREDICT ----------
    def _predict_one(self, x: np.ndarray):
        case = {self._feature_names[i]: [x[i]] for i in range(len(x))}
        for label, rules in self.ruleset.items():
            if any(self.rule_satisfied(case, r) for r in rules):
                return label
        return max(self.ruleset.items(), key=lambda kv: len(kv[1]))[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return (self.predict(X) == y).mean()

    # ----------  ДЛЯ DSModelMultiQ  ----------
    def _make_lambda(self, rule: Dict, cols: List[str], tol: float):
        """
        Превращает словарь‑правило в lambda‑функцию.
        Используется DSModelMultiQ.generate_*_rules().
        """
        def f(sample: np.ndarray) -> bool:
            for attr, (op, thr) in rule.items():
                try:
                    idx = cols.index(attr)
                except ValueError:
                    return False  # неизвестный атрибут
                v = sample[idx]
                if op == "==" and abs(v - thr) > tol:
                    return False
                if op == ">" and not (v > thr):
                    return False
                if op == "<" and not (v < thr):
                    return False
            return True

        return f
# -------------------------------------------------------------------
