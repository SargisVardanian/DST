# DSRipper.py

"""
Vector-accelerated incremental RIPPER / FOIL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
• Батч-обучение для экономии памяти (batch_size в fit).
• Для **каждого** класса строит свой список правил.
• Кандидаты:
      – числовые признаки → 9 квантилей (10…90 %)
      – категориальные    → ≤ max_unique_cat самых частых значений
• FOIL-gain и precision считаются векторно на булевых масках NumPy.
• Дедупликация правил между батчами (_canon + _seen).
• В конце одно TRUE-правило с меткой самого частого
  среди непокрытых (или глобального модального класса).
Совместим c DSModelMultiQ: сохранены ruleset[label],
_ordered_rules, _make_lambda, сигнатура fit.
"""

from __future__ import annotations
import math, numbers, itertools
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

Rule = Dict[str, Tuple[str, Any]]
Mask = np.ndarray  # dtype=bool


class DSRipper:
    # ────────────────────── INIT ──────────────────────
    def __init__(self,
                 *,
                 algo: str = "ripper",
                 min_prec: float = 0.6,
                 min_cov_frac: float = 0.01,
                 min_gain: float = 0.001,
                 max_rule_len: int = 6,
                 max_rules_per_class: int = 50,
                 eq_tol: float = 1e-3,
                 max_unique_cat: int = 10):
        """
        Параметры:
          algo             — алгоритм индукции (‘ripper’ или ‘foil’).
          min_prec         — минимум precision (p/(p+n)) для остановки роста rule.
          min_cov_frac     — минимальная доля покрытия (coverage) POS‐примеров rule.
          min_gain         — FOIL‐gain threshold: < min_gain → переносить литерал нельзя.
          max_rule_len     — максимальная длина (число литералов) одного rule.
          max_rules_per_class — сколько rule’ов максимум строить для каждого класса.
          eq_tol           — допускаемая погрешность при сравнении вещественных (==).
          max_unique_cat   — если у признака ≤ max_unique_cat уникальных значений → это categorical;
                              тогда бираем топ max_unique_cat самых частотных категорий.
        """
        self.algo = algo.lower()
        self.min_prec = min_prec
        self.min_cov_frac = min_cov_frac
        self.min_gain = min_gain
        self.max_rule_len = max_rule_len
        self.max_rules_per_cls = max_rules_per_class
        self.eq_tol = eq_tol
        self.max_unique_cat = max_unique_cat

        # внутреннее состояние
        self._feature_names: List[str] = []
        self._is_cat: List[bool] = []
        self._X: np.ndarray | None = None

        # кэши масок (чтобы дважды не пересчитывать одни и те же булевые векторы)
        self._num_mask_cache: Dict[Tuple[int, str, float], Mask] = {}
        self._cat_mask_cache: Dict[Tuple[int, Any], Mask] = {}

        # выход: для каждого класса (label) список правил
        self.ruleset: Dict[int, List[Rule]] = defaultdict(list)
        self._ordered_rules: List[Tuple[int, Rule]] = []
        self._default_label: Optional[int] = None
        self._seen: set[str] = set()  # для дедупликации (каноничный ключ правила)

    # ─────────────────── low-level utils ───────────────────
    @staticmethod
    def _precision(pos: Mask, neg: Mask, m: Mask) -> float:
        """
        точность (precision) правила:
           p = |m ∧ pos|,  n = |m ∧ neg|
        → precision = p/(p+n)
        """
        p = (m & pos).sum()
        n = (m & neg).sum()
        return p / (p + n + 1e-12)

    def _canon(self, rule: Rule) -> str:
        """
        Канонический ключ правила:
        e.g. { 'age':('>',32.5), 'sex':('==',1.0) } → "age>32.5|sex==1.0"
        Нужно, чтобы не дублировать похожие правила.
        """
        if not rule:
            return "TRUE"
        parts = []
        for a, (op, thr) in sorted(rule.items()):
            parts.append(f"{a}{op}{thr}")
        return "|".join(parts)

    # ──────────────── mask generators ────────────────
    def _num_mask(self, col: int, op: str, thr: float) -> Mask:
        """
        Булева маска { True, False } для условия:
          если op == '>' → X[:,col] > thr
          если op == '<' → X[:,col] < thr
        """
        key = (col, op, thr)
        if key not in self._num_mask_cache:
            x = self._X[:, col]
            if op == ">":
                self._num_mask_cache[key] = (x > thr)
            else:
                self._num_mask_cache[key] = (x < thr)
        return self._num_mask_cache[key]

    def _cat_mask(self, col: int, val) -> Mask:
        """
        Булева маска для категориального условия (X[:,col] == val).
        """
        key = (col, val)
        if key not in self._cat_mask_cache:
            self._cat_mask_cache[key] = (self._X[:, col] == val)
        return self._cat_mask_cache[key]

    def _literal_mask(self, attr: str, op: str, thr) -> Mask:
        """
        В зависимости от того, categorical ли признак,
        возвращаем либо cat_mask(col,val), либо num_mask(col,op,thr).
        """
        col = self._feature_names.index(attr)
        if self._is_cat[col]:
            return self._cat_mask(col, thr)
        else:
            return self._num_mask(col, op, float(thr))

    # ──────────────── FOIL-gain (vectorised) ────────────────
    def _foil_gain(self, pos: Mask, neg: Mask, cur: Mask, lit: Mask) -> float:
        """
        Векторный расчёт FOIL‐gain:
          p0 = |cur ∧ pos|  (или pos.sum() если cur пусто)
          n0 = |cur ∧ neg|  (или neg.sum() если cur пусто)
          new_mask = cur ∧ lit
          p1 = |new_mask ∧ pos|
          n1 = |new_mask ∧ neg|
          i0 = log2((p0+ε)/(p0+n0+ε)),  i1 = log2((p1+ε)/(p1+n1+ε))
          gain = p1*(i1 − i0)
        Если p1=0 → gain = −∞, правило бесполезно (не покрывает ни одного pos).
        """
        p0 = (cur & pos).sum() or pos.sum()
        n0 = (cur & neg).sum() or neg.sum()
        new_mask = cur & lit
        p1 = (new_mask & pos).sum()
        n1 = (new_mask & neg).sum()
        if p1 == 0:
            return -math.inf
        i0 = math.log2((p0 + 1e-9) / (p0 + n0 + 1e-9))
        i1 = math.log2((p1 + 1e-9) / (p1 + n1 + 1e-9))
        return float(p1) * (i1 - i0)

    # ──────────────── grow one rule ────────────────
    def _best_literal_by_prec(self, pos: Mask, neg: Mask, cur_mask: Mask,
                              col: int, attr: str) -> Tuple[str, str, Any, Mask] | None:
        """
        Если FOIL‐gain = −∞ (ни один литерал не улучшает gain),
        но правило пустое → ищем литерал, который даёт максимум precision p/(p+n).
        Возвращает (attr, op, thr, lit_mask) либо None, если вообще ничего не подходит.
        """
        best_prec = -1.0
        best      = None

        if self._is_cat[col]:
            # перебираем все уникальные значения категории в pos
            vals, cnts = np.unique(self._X[pos, col], return_counts=True)
            # можно сразу сортировать по частоте (но на маленьких колич. это не критично)
            for v in vals:
                lit = self._cat_mask(col, v)
                prec = self._precision(pos, neg, cur_mask & lit)
                if prec > best_prec:
                    best_prec = prec
                    best = ("==", v, lit)
        else:
            # для числовых берём 9 квантилей [10%,20%, …, 90%]
            qs = np.quantile(self._X[pos, col], np.linspace(.1, .9, 9))
            for thr in qs:
                for op in (">", "<"):
                    lit = self._num_mask(col, op, thr)
                    prec = self._precision(pos, neg, cur_mask & lit)
                    if prec > best_prec:
                        best_prec = prec
                        best = (op, thr, lit)

        if best is None:
            return None

        op, thr, lit = best
        return (attr, op, thr, lit)

    def _grow_rule(self, pos: Mask, neg: Mask) -> Rule | None:
        """
        Построение одного правила (grow):
          – начинаем с пустого rule и маски cur_mask = all True
          – пока точность ((cur_mask & pos).sum() / ((cur_mask & pos).sum()+(cur_mask&neg).sum())) < min_prec
            и длина rule < max_rule_len:
            • ищем литерал с максимальным FOIL‐gain ≥ min_gain
            • если ни один gain ≥ min_gain, но rule ещё пустое → ищем багаж (галочку) по precision
            • если ничего не нашли → выходим / возвращаем None
            • добавляем найденный литерал (attr,op,thr) в rule, updates cur_mask &= lit_mask
            • если cur_mask[pos].sum() == 0 → больше не покрываем ни один pos → выходим
        """
        rule: Rule = {}
        cur_mask = np.ones_like(pos, dtype=bool)

        while True:
            # условие остановки по precision
            if rule and self._precision(pos, neg, cur_mask) >= self.min_prec:
                break
            # условие остановки по длине правила
            if len(rule) >= self.max_rule_len:
                break

            best_gain = -math.inf
            best_lit = None

            # перебираем все столбцы/признаки, не входящие пока в rule
            for col, attr in enumerate(self._feature_names):
                if attr in rule:
                    continue

                if self._is_cat[col]:
                    # перебрать ≤ max_unique_cat самых частых категорий в pos
                    vals, cnts = np.unique(self._X[pos, col], return_counts=True)
                    # отберём топ max_unique_cat по частоте
                    if len(vals) > self.max_unique_cat:
                        idx_sorted = np.argsort(cnts)[-self.max_unique_cat:]
                        vals = vals[idx_sorted]

                    for v in vals:
                        lit_mask = self._cat_mask(col, v)
                        g = self._foil_gain(pos, neg, cur_mask, lit_mask)
                        if g > best_gain and g >= self.min_gain:
                            best_gain = g
                            best_lit = (attr, "==", v, lit_mask)
                else:
                    # для числовых признаков – 9 квантилей
                    qs = np.quantile(self._X[pos, col], np.linspace(.1, .9, 9))
                    for thr in qs:
                        for op in (">", "<"):
                            lit_mask = self._num_mask(col, op, thr)
                            g = self._foil_gain(pos, neg, cur_mask, lit_mask)
                            if g > best_gain and g >= self.min_gain:
                                best_gain = g
                                best_lit = (attr, op, thr, lit_mask)

            # если ни один FOIL‐gain не подошёл, а правило пустое → fallback по precision
            if best_lit is None and not rule:
                for col, attr in enumerate(self._feature_names):
                    if attr in rule:
                        continue
                    r = self._best_literal_by_prec(pos, neg, cur_mask, col, attr)
                    if r:
                        best_lit = r
                        break

            if best_lit is None:
                # больше нечего добавить
                break

            # добавляем найденную пару (attr, op, thr) и обновляем cur_mask
            a, op, thr, lit_mask = best_lit
            rule[a] = (op, thr)
            cur_mask &= lit_mask

            # если теперь ни один pos не покрывается – выходим
            if cur_mask[pos].sum() == 0:
                break

        return rule or None

    # ──────────────── prune (RIPPER) ────────────────
    def _prune_rule(self, pos: Mask, neg: Mask, rule: Rule) -> Rule:
        """
        Для RIPPER: после `grow`, проводим «ошущение листьев» (post‐prune):
        – если dropp’нуть один литерал из rule и при этом precision= p/(p+n) всё ещё ≥ min_prec,
          то мы считаем, что этот литерал «лишний» и удаляем его. Повторяем до тех пор, пока не улучшимrule,
          либо не останется <2 литералов.
        FOIL‐режим (self.algo != 'ripper') пропускает prune и возвращает rule как есть.
        """
        if self.algo != "ripper" or len(rule) <= 1:
            return rule

        best_rule = rule
        # для текущего `rule` строим маску cur_mask
        cur_mask = np.ones(len(pos), dtype=bool)
        for a, (op, thr) in rule.items():
            cur_mask &= self._literal_mask(a, op, thr)

        # пытаемся отбросить по одному литералу и проверяем precision
        for a in list(rule.keys()):
            tmp = dict(rule)
            tmp.pop(a)
            tmp_mask = np.ones(len(pos), dtype=bool)
            for b, (op2, thr2) in tmp.items():
                tmp_mask &= self._literal_mask(b, op2, thr2)
            if self._precision(pos, neg, tmp_mask) >= self.min_prec:
                best_rule = tmp
        return best_rule

    # ──────────────── rules for one class ────────────────
    def _rules_for_class(self, pos: Mask, neg: Mask) -> List[Rule]:
        """
        Построение списка правил для одного класса:
        1. Вычисляем минимальное покрытие min_cov = max(1, min_cov_frac * |pos|).
        2. Пока ещё остались непокрытые pos (pos.sum() > 0) и не превысили max_rules_per_class:
           a. r = grow_rule(pos, neg). Если r=None → выходим.
           b. r = prune_rule(pos, neg, r).
           c. строим mask для r и проверяем, что |mask| ≥ min_cov и precision(mask) ≥ min_prec.
              Иначе выходим (r слишком неконструктивное).
           d. Добавляем r в список, отсекаем покрытые: pos &= ~mask.
        """
        rules: List[Rule] = []
        min_cov = max(1, int(self.min_cov_frac * pos.sum()))

        while pos.sum() and len(rules) < self.max_rules_per_cls:
            r = self._grow_rule(pos, neg)
            if r is None:
                break
            r = self._prune_rule(pos, neg, r)

            # строим mask для полученного r
            mask = np.ones(len(pos), dtype=bool)
            for a, (op, thr) in r.items():
                mask &= self._literal_mask(a, op, thr)

            # проверяем покр-ие и precision
            if mask.sum() < min_cov or self._precision(pos, neg, mask) < self.min_prec:
                break

            rules.append(r)
            pos &= ~mask  # отсекаем те pos, которые покрылись только что построенным правилом

        return rules

    # ────────────────────── FIT ──────────────────────
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            *,
            feature_names: List[str],
            batch_size: Optional[int] = None):
        """
        Обучение RIPPER/FOIL по батчам:
        1. Разбиваем индексы 0…N−1 на батчи размера batch_size.
        2. Для каждого батча s:e:
           a. строим булевый mask `[s:e]` (только эти индексы считаются).
           b. Для каждого класса cls в np.unique(y):
              ‒ формируем pos = (y==cls) & batch_mask, neg = (y!=cls) & batch_mask.
              ‒ вызываем _rules_for_class(pos.copy(), neg.copy()) → набор новых правил.
              ‒ для каждого нового правила r: если _canon(r) не в self._seen →
                добавляем в self.ruleset[cls], self._ordered_rules, self._seen, печатаем “+ New rule for class cls: {r}”.
        3. После прохода по всем батчам строим «TRUE–default» правило:
           – вычисляем маску covered = ⋁_{rule∈_ordered_rules} mask(rule).
           – uncovered = набор тех y, которые не покрылись ни одним правилом.
           – default_label = majority‐vote( uncovered или глобально y ).
           – Добавляем (default_label, {}) в конец списка _ordered_rules.
        """
        self._feature_names = list(feature_names)
        self._X = X
        # определяем категориальные признаки (≤ max_unique_cat уникальных)
        self._is_cat = [len(np.unique(X[:, j])) <= self.max_unique_cat
                        for j in range(X.shape[1])]

        self.ruleset.clear()
        self._ordered_rules.clear()
        self._seen.clear()

        n = len(y)
        bs = n if batch_size in (None, 0) else max(1, batch_size)

        for s in range(0, n, bs):
            e = min(s + bs, n)
            # булевый vector размера n: True на позиции в батче, False вне
            batch_mask = np.zeros(n, dtype=bool)
            batch_mask[s:e] = True

            batch_added = 0
            print(f"\n--- batch {s}-{e-1} ---")
            for cls in np.unique(y):
                pos = (y == cls) & batch_mask
                if pos.sum() == 0:
                    continue
                neg = (~(y == cls)) & batch_mask

                for r in self._rules_for_class(pos.copy(), neg.copy()):
                    sig = self._canon(r)
                    if sig in self._seen:
                        continue
                    self._seen.add(sig)
                    self.ruleset[cls].append(r)
                    self._ordered_rules.append((cls, r))
                    batch_added += 1
                    # Печатаем информацию о новом правиле
                    print(f"    + New rule for class {cls}: {r}")

            total_rules_so_far = len(self._ordered_rules)  # не считаем default
            print(f">> batch {s}-{e-1}  added: {batch_added}  "
                  f"total rules so far (без default): {total_rules_so_far}")

        # TRUE default (для непокрытых экземпляров)
        covered = np.zeros(n, dtype=bool)
        for _, r in self._ordered_rules:
            m = np.ones(n, dtype=bool)
            for a, (op, thr) in r.items():
                m &= self._literal_mask(a, op, thr)
            covered |= m
        uncovered = y[~covered]
        # если всё покрыто, смотрим глобальное majority‐label
        self._default_label = int(np.bincount(uncovered if len(uncovered) else y).argmax())
        self._ordered_rules.append((self._default_label, {}))

        print(f"\n{self.algo}: rules per class",
              {c: len(r) for c, r in self.ruleset.items()},
              "| default =", self._default_label)
        return self

    # ───────────────── predict / score ─────────────────
    def _predict_one(self, row: np.ndarray) -> int:
        """
        Последовательно пробегаем _ordered_rules:
         – если rule пустое → сразу возвращаем default_label (TRUE).
         – иначе проверяем все литералы: (attr,op,thr) – сравниваем с eq_tol.
         – если все литералы удовлетворены → возвращаем этот класс.
        """
        for lbl, rule in self._ordered_rules:
            if not rule:  # пустое правило: TRUE, значит default
                return lbl
            ok = True
            for a, (op, thr) in rule.items():
                j = self._feature_names.index(a)
                v = row[j]
                if op in ("=", "=="):
                    if not (abs(v - thr) <= self.eq_tol):
                        ok = False
                        break
                elif op == ">":
                    if not (v > thr):
                        ok = False
                        break
                elif op == "<":
                    if not (v < thr):
                        ok = False
                        break
            if ok:
                return lbl
        raise RuntimeError("unreachable")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(r) for r in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    # ───────────── λ-wrapper (DSModelMultiQ) ─────────────
    def _make_lambda(self, rule: Rule, cols: List[str], tol: float):
        """
        Превращает dict‐правило в лямбда‐функцию для DSModelMultiQ.generate_*_rules().
        """
        def f(sample: np.ndarray) -> bool:
            for a, (op, thr) in rule.items():
                idx = cols.index(a)
                v = sample[idx]
                if op in ("=", "==") and abs(v - thr) > tol:
                    return False
                if op == ">" and not (v > thr):
                    return False
                if op == "<" and not (v < thr):
                    return False
            return True
        return f
