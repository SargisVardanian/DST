import torch
from torch import nn
from torch.nn.functional import pad
import numpy as np
from scipy.stats import norm
from itertools import count
import dill, os

# --- проектно-локальные вспомогательные методы -------------------------------
from Common_code.DSRule import DSRule
from Common_code.core import create_random_maf_k
from Common_code.utils import is_categorical
from Common_code.DSRipper import DSRipper    # <<< ИЗМЕНЕНИЕ: используем обновлённый DSRipper
from Common_code.core import filter_duplicate_rules  # optional
# -----------------------------------------------------------------------------

class DSModelMultiQ(nn.Module):
    """
    Torch-имплементация Dempster–Shafer классификатора,
    основа – human-readable rules (RIPPER/FOIL).
    """

    def __init__(self, k,
                 precompute_rules: bool = False,
                 device: str = "cpu",
                 force_precompute: bool = False,
                 use_foil: bool = False):
        """
        k : int
            Number of classes.
        use_foil : bool
            Если True – FOIL, иначе RIPPER.
        """
        super().__init__()
        self.k = k
        self.precompute_rules = precompute_rules
        self.device = torch.device(device)
        self.force_precompute = force_precompute

        # контейнеры для масс и функций правил
        self._params, self.preds = [], []
        self.n = 0
        self.rmap = {}
        self.active_rules = []
        self._all_rules = None

        # <<< ИЗМЕНЕНИЕ: инстанцируем DSRipper с новыми параметрами по умолчанию
        algo = "foil" if use_foil else "ripper"
        self.generator = DSRipper(
            algo=algo
        )

        # dummy-параметр, чтобы избежать деления на 0
        self.dummy = nn.Parameter(torch.tensor(1.0))


    # ------------------------------------------------------------------ #
    #                         ОРИГИНАЛЬНЫЕ МЕТОДЫ                        #
    # ------------------------------------------------------------------ #
    def add_rule(self, pred, m_sing=None, m_uncert=None):
        """
        Добавляем правило. Если массы не заданы, генерируем случайные.
        :param pred: DSRule или лямбда-функция (predicate).
        """
        self.preds.append(pred)
        self.n += 1
        if m_sing is None or m_uncert is None or len(m_sing) != self.k:
            masses = create_random_maf_k(self.k, 0.8)
        else:
            masses = m_sing + [m_uncert]
        m = torch.tensor(masses, requires_grad=True, dtype=torch.float)
        self._params.append(m)

    def forward(self, X):
        """
        :param X: тензор с shape=(batch, k+1),
                  в первом столбце – индекс sample, далее – признаки.
        :return: вероятностное распределение (one-hot).
        """
        if not self._params:                        # <-- NEW EARLY EXIT
            batch = X.size(0)
            out   = torch.zeros(batch, self.k, device=self.device)
            maj   = getattr(self, "default_class", 0)
            out[:, maj] = 1.0
            return out
        ms = torch.stack(self._params).to(self.device)
        if self.force_precompute:
            # комбинаторика commonalities → выбор всех правил сразу
            qs = ms[:, :-1] + \
                 ms[:, -1].view(-1, 1) * torch.ones_like(ms[:, :-1])
            qt = qs.repeat(len(X), 1, 1)
            vectors, indices = X[:, 1:], X[:, 0].long()
            sel = self._select_all_rules(vectors, indices)
            qt[sel] = 1
            temp_res = qt.prod(1)
            res = torch.where(temp_res <= 1e-16, temp_res.add(1e-16), temp_res)
            out = res / res.sum(1, keepdim=True)
        else:
            out = torch.zeros(len(X), self.k).to(self.device)
            for i in range(len(X)):
                sel = self._select_rules(X[i, 1:], int(X[i, 0].item()))
                if len(sel) == 0:
                    out[i] = torch.ones(self.k, device=self.device) / self.k
                else:
                    mt = torch.index_select(ms, 0, torch.LongTensor(sel).to(self.device))
                    qt = mt[:, :-1] + mt[:, -1].view(-1, 1) * torch.ones_like(mt[:, :-1])
                    res = qt.prod(0)
                    if res.sum().item() <= 1e-16:
                        res = res + 1e-16
                        out[i] = res / res.sum()
                    else:
                        out[i] = res / res.sum()
        return out

    def clear_rmap(self):
        self.rmap = {}
        self._all_rules = None

    def _select_all_rules(self, X, indices):
        """
        Возвращает булевый тензор shape=(len(X), n_rules),
        True там, где правило НЕ срабатывает (для precompute).
        """
        if self._all_rules is None:
            self._all_rules = torch.zeros(0, self.n, dtype=torch.bool).to(self.device)
        max_index = torch.max(indices)
        len_all_rules = len(self._all_rules)
        if max_index < len_all_rules:
            return self._all_rules[indices]
        else:
            desired_len = max_index + 1
            padding = (0, 0, 0, desired_len - len_all_rules)
            self._all_rules = pad(self._all_rules, padding)
        sel = torch.zeros(len(X), self.n, dtype=torch.bool).to(self.device)
        X_np = X.cpu().data.numpy()
        for i, sample, index in zip(count(), X_np, indices):
            for j in range(self.n):
                sel[i, j] = not bool(self.preds[j](sample))
            self._all_rules[index] = sel[i]
        return sel

    def normalize(self):
        """Нормализация масс (чтобы суммировались в 1)."""
        with torch.no_grad():
            for t in self._params:
                t.clamp_(0., 1.)
                if t.sum().item() < 1:
                    t[-1].add_(1 - t.sum())
                else:
                    t.div_(t.sum())

    def check_nan(self, tag=""):
        for p in self._params:
            if torch.isnan(p).any():
                print(p)
                raise RuntimeError("NaN mass found at %s" % tag)
            if p.grad is not None and torch.isnan(p.grad).any():
                print(p)
                print(p.grad)
                raise RuntimeError("NaN grad mass found at %s" % tag)

    def parameters(self, recurse=True):
        return self._params

    def extra_repr(self):
        """
        Строит текстовое представление: какие правила, с какими массами.
        """
        builder = f"DS Classifier using {self.n} rules\n"
        for i in range(self.n):
            ps = str(self.preds[i])
            ms = self._params[i]
            builder += f"\nRule {i+1}: {ps}\n\t"
            for j in range(len(ms) - 1):
                builder += f"C{j+1}: {ms[j]:.3f}\t"
            builder += f"Unc: {ms[self.k]:.3f}\n"
        return builder[:-1]

    def find_most_important_rules(self, classes=None, threshold=0.2):
        """
        Возвращает словарь {class: [(score, idx, ps, sqrt(score), masses), ...]}.
        """
        if classes is None:
            classes = [i for i in range(self.k)]

        with torch.no_grad():
            rules = {}
            for j in range(len(classes)):
                cls = classes[j]
                found = []
                for i in range(len(self._params)):
                    ms = self._params[i].detach().numpy()
                    score = (ms[j]) * (1 - ms[-1])
                    if score >= threshold * threshold:
                        ps = str(self.preds[i])
                        found.append((score, i, ps, np.sqrt(score), ms))
                found.sort(reverse=True)
                rules[cls] = found
        return rules

    def print_most_important_rules(self, classes=None, threshold=0.2):
        rules = self.find_most_important_rules(classes, threshold)
        if classes is None:
            classes = [i for i in range(self.k)]
        builder = ""
        for i_cls in range(len(classes)):
            rs = rules[classes[i_cls]]
            builder += f"\n\nMost important rules for class {classes[i_cls]}"
            for r in rs:
                builder += f"\n\n\t[{r[3]:.3f}] R{r[1]}: {r[2]}\n\t\t"
                masses = r[4]
                for j in range(len(masses)):
                    builder += f"\t{str(classes[j])[:3] if j < len(classes) else 'Unc'}: {masses[j]:.3f}"
        print(builder)


    # ==========================  П Р И В А Т Н Ы Е  ========================== #

    def _make_lambda(self, rule_obj, cols, tol):
        """
        Строит λ-функцию (predicate) для одного правила (или списка клауз OR).
        rule_obj: dict (AND-клауза) или list[dict] (OR из нескольких клауз).
        """
        clauses = [rule_obj] if isinstance(rule_obj, dict) else rule_obj

        def _ok(value, op, thr):
            if value is None:  # nan / пропуск
                return False
            if op in ("=", "=="):
                return abs(value - thr) <= tol
            if op == ">":
                return value >  thr + tol
            if op == ">=":
                return value >= thr - tol
            if op == "<":
                return value <  thr - tol
            if op == "<=":
                return value <= thr + tol
            return False

        def f(x):
            for cl in clauses:  # OR
                good = True
                for attr, (op, thr) in cl.items():  # AND
                    v = x[cols.index(attr)]
                    if not _ok(v, op, thr):
                        good = False
                        break
                if good:
                    return True
            return False

        return f


    # ───────── фрагмент DSModelMultiQ.py ─────────
    def _rule_text(self, cond):
        """Строит строку 'attr op val …' из словаря или списка клауз."""
        if isinstance(cond, dict):
            return " AND ".join(f"{a} {op} {v:.4f}" for a, (op, v) in cond.items())
        clause_strs = []
        for cl in cond:   # OR
            clause_strs.append(" AND ".join(f"{a} {op} {v:.4f}" for a, (op, v) in cl.items()))
        return " OR ".join(f"({s})" for s in clause_strs)

    def generate_ripper_rules(self, X, y, column_names, batch_size=2000):
        self.generator.fit(X, y, feature_names=list(column_names),
                           batch_size=batch_size)
        ds_rules = []
        cols = list(column_names)
        tol  = self.generator.eq_tol
        for cls, cond in self.generator._ordered_rules:
            if isinstance(cond, dict) and not cond:        # TRUE-default пропускаем
                continue
            lam = self._make_lambda(cond, cols, tol)
            caption = f"Class {cls}: {self._rule_text(cond)}"
            ds_rules.append(DSRule(lam, caption))
        return ds_rules

    def generate_foil_rules(self, X, y, column_names, batch_size=2000):
        self.generator.fit(X, y, feature_names=list(column_names),
                           batch_size=batch_size)
        ds_rules = []
        cols = list(column_names)
        tol  = self.generator.eq_tol
        for cls, cond in self.generator._ordered_rules:
            if isinstance(cond, dict) and not cond:
                continue
            lam = self._make_lambda(cond, cols, tol)
            caption = f"Class {cls}: {self._rule_text(cond)}"
            ds_rules.append(DSRule(lam, caption))
        return ds_rules

    def precompute_fire_matrix(self, X: np.ndarray):
        """
        Вычисляет и кэширует bool-матрицу размера (N, n_rules) на CPU.
        Вызывать один раз после импорта/генерации правил и ДО любых
        set_test_usability / fast-forward операций.
        """
        N = len(X)
        fires = np.zeros((N, self.n), dtype=np.uint8)
        for j, rule in enumerate(self.preds):
            fires[:, j] = np.fromiter((rule(row) for row in X), dtype=np.uint8, count=N)
        self._fire_cpu = fires            # кэш

    def fire_matrix(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "_fire_cpu", None) is None:
            raise RuntimeError("вызывайте precompute_fire_matrix(X_train) сначала")
        # считаем fires на новом X — точно такой же цикл
        N = len(X)
        m = np.zeros((N, self.n), dtype=np.uint8)
        for j, rule in enumerate(self.preds):
            m[:, j] = np.fromiter((rule(row) for row in X), dtype=np.uint8, count=N)
        return m





    def generate_statistic_single_rules(self, X, breaks=2, column_names=None, generated_columns=None):
        """
        Populates the model with attribute-independant rules based on statistical breaks.
        In total this method generates No. attributes * (breaks + 1) rules
        :param X: Set of inputs (can be the same as training or a sample)
        :param breaks: Number of breaks per attribute
        :param column_names: Column attribute names
        :param generated_columns: array of booleans with len() = num_features representing which
            will have rules generated. The default behaviour is to generate rules for all
        """
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        brks = norm.ppf(np.linspace(0, 1, breaks + 2))[1:-1]
        num_features = len(mean)

        if generated_columns is None:
            generated_columns = np.repeat(True, num_features)
        assert generated_columns.shape == (num_features,), "generated_columns has wrong shape {}".format(
            generated_columns.shape)

        if column_names is None:
            column_names = ["X[%d]" % i for i in range(num_features)]

        for i in range(num_features):
            if not generated_columns[i]:
                continue
            if is_categorical(X[:, i]):
                categories = np.unique(X[:, i][~np.isnan(X[:, i])])
                for cat in categories:
                    self.add_rule(DSRule(lambda x, i=i, k=cat: x[i] == k, "%s = %s" % (column_names[i], str(cat))))
            else:
                # First rule
                v = mean[i] + std[i] * brks[0]
                self.add_rule(DSRule(lambda x, i=i, v=v: x[i] <= v, "%s < %.3f" % (column_names[i], v)))
                # Mid rules
                for j in range(1, len(brks)):
                    vl = v
                    v = mean[i] + std[i] * brks[j]
                    self.add_rule(DSRule(lambda x, i=i, vl=vl, v=v: vl <=
                                                                    x[i] < v,
                                         "%.3f < %s < %.3f" % (vl, column_names[i], v)))
                # Last rule
                self.add_rule(DSRule(lambda x, i=i, v=v: x[i] > v, "%s > %.3f" % (column_names[i], v)))

        print(f"self.add_rule:  ", self.add_rule)

    def generate_categorical_rules(self, X, column_names=None, exclude=None):
        """
        Populates the model with attribute-independant rules based on categories of attributes, continous columns are
        skipped.
        :param X: Set of inputs (can be the same as training or a sample)
        :param column_names: Column attribute names
        """
        m = X.shape[1]
        if column_names is None:
            column_names = ["X[%d]" % i for i in range(m)]

        if exclude is None:
            exclude = []

        for i in range(m):
            if is_categorical(X[:, i]) and column_names[i] not in exclude:
                categories = np.unique(X[:, i][~np.isnan(X[:, i])])
                for cat in categories:
                    self.add_rule(DSRule(lambda x, i=i, k=cat: x[i] == k, "%s = %s" % (column_names[i], str(cat))))

    def generate_mult_pair_rules(self, X, column_names=None, include_square=False):
        """
        Populates the model with with rules combining 2 attributes by their multipication, adding both positive
        and negative rule. In total this method generates (No. attributes)^2 rules
        :param X: Set of inputs (can be the same as training or a sample)
        :param column_names: Column attribute names
        :param include_square: Includes rules comparing the same attribute (ie x[i] * x[i])
        """
        mean = np.nanmean(X, axis=0)

        if column_names is None:
            column_names = ["X[%d]" % i for i in range(len(mean))]

        offset = 0 if include_square else 1

        for i in range(len(mean)):
            for j in range(i + offset, len(mean)):
                # mk = mean[i] * mean[j]
                mi = mean[i]
                mj = mean[j]
                self.add_rule(DSRule(lambda x, i=i, j=j, mi=mi, mj=mj: (x[i] - mi) * (x[j] - mj) > 0,
                                     "Positive %s - %.3f, %s - %.3f" % (
                                         column_names[i], mean[i], column_names[j], mean[j])))
                self.add_rule(DSRule(lambda x, i=i, j=j, mi=mi, mj=mj: (x[i] - mi) * (x[j] - mj) <= 0,
                                     "Negative %s - %.3f, %s - %.3f" % (
                                         column_names[i], mean[i], column_names[j], mean[j])))

    def generate_custom_range_single_rules(self, column_names, name, breaks):
        """
        Populates the model with attribute-independant rules based on custom defined breaks.
        In total this method generates len(breaks) + 1 rules
        :param column_names: Column attribute names
        :param name: The target column name to generate rules
        :param breaks: Array of float indicating the values of the breaks
        """
        i = column_names.tolist().index(name)
        if i == -1:
            raise NameError("Cannot find column with name %s" % name)
        v = breaks[0]
        # First rule
        self.add_rule(DSRule(lambda x, i=i, v=v: x[i] <= v, "%s < %.3f" % (name, v)))
        # Mid rules
        for j in range(1, len(breaks)):
            vl = v
            v = breaks[j]
            self.add_rule(DSRule(lambda x, i=i, vl=vl, v=v: vl <= x[i] < v, "%.3f < %s < %.3f" % (vl, name, v)))
        # Last rule
        self.add_rule(DSRule(lambda x, i=i, v=v: x[i] > v, "%s > %.3f" % (name, v)))

    def generate_custom_range_rules_by_gender(self, column_names, name, breaks_men, breaks_women, gender_name="gender"):
        """
        Populates the model with attribute-independant rules based on custom defined breaks separated by gender.
        :param column_names: Column attribute names
        :param name: The target column name to generate rules
        :param breaks_men: Array of float indicating the values of the breaks for men
        :param breaks_women: Array of float indicating the values of the breaks for women
        :param gender_name: Name of the column containing the gender
        """
        i = column_names.tolist().index(name)
        g = column_names.tolist().index(gender_name)

        if i == -1 or g == -1:
            raise NameError("Cannot find column with name %s" % name)

        for gv, gname, breaks in [(0, "Men", breaks_men), (1, "Women", breaks_women)]:
            v = breaks[0]
            # First rule
            self.add_rule(
                DSRule(lambda x, i=i, v=v, g=g, gv=gv: x[g] == gv and x[i] <= v, "%s: %s < %.3f" % (gname, name, v)))
            # Mid rules
            for j in range(1, len(breaks)):
                vl = v
                v = breaks[j]
                self.add_rule(DSRule(lambda x, i=i, g=g, gv=gv, vl=vl, v=v: x[g] == gv and vl <= x[i] < v,
                                     "%s: %.3f < %s < %.3f" % (gname, vl, name, v)))
            # Last rule
            self.add_rule(
                DSRule(lambda x, i=i, g=g, gv=gv, v=v: x[g] == gv and x[i] > v, "%s: %s > %.3f" % (gname, name, v)))

    def generate_outside_range_pair_rules(self, column_names, ranges):
        """
        Populates the model with outside-normal-range pair of attributes rules
        :param column_names: The columns names in the dataset
        :param ranges: Matrix size (k,3) indicating the lower, the upper bound and the name of the column
        """
        for index_i in range(len(ranges)):
            col_i = ranges[index_i][2]
            i = column_names.tolist().index(col_i)
            for index_j in range(index_i + 1, len(ranges)):
                col_j = ranges[index_j][2]
                j = column_names.tolist().index(col_j)
                # Extract ranges
                li = ranges[index_i][0]
                hi = ranges[index_i][1]
                lj = ranges[index_j][0]
                hj = ranges[index_j][1]
                # Add Rules
                if not np.isnan(li) and not np.isnan(lj):
                    self.add_rule(DSRule(lambda x, i=i, j=j, li=li, lj=lj: x[i] < li and x[j] < lj,
                                         "Low %s and Low %s" % (col_i, col_j)))
                if not np.isnan(hi) and not np.isnan(lj):
                    self.add_rule(DSRule(lambda x, i=i, j=j, hi=hi, lj=lj: x[i] > hi and x[j] < lj,
                                         "High %s and Low %s" % (col_i, col_j)))
                if not np.isnan(hi) and not np.isnan(hj):
                    self.add_rule(DSRule(lambda x, i=i, j=j, hi=hi, hj=hj: x[i] > hi and x[j] > hj,
                                         "High %s and High %s" % (col_i, col_j)))
                if not np.isnan(li) and not np.isnan(hj):
                    self.add_rule(DSRule(lambda x, i=i, j=j, li=li, hj=hj: x[i] < li and x[j] > hj,
                                         "Low %s and High %s" % (col_i, col_j)))

    def load_rules_bin(self, filename):
        """
        Loads rules from a file, it deletes previous rules
        :param filename: The name of the input file
        """
        if os.path.getsize(filename) == 0:
            raise RuntimeError(f"Cannot load from empty file: {filename}")
        with open(filename, "rb") as f:
            sv = dill.load(f)
            self.preds = sv["preds"]
            self._params = sv["masses"]
            self.n = len(self.preds)

        # print(self.preds)

    def save_rules_bin(self, filename):
        """
        Saves the current rules into a file
        :param filename: The name of the file
        """
        with open(filename, "wb") as f:
            sv = {"preds": self.preds, "masses": self._params}
            dill.dump(sv, f)

    def get_rules_size(self):
        return self.n

    def get_active_rules_size(self):
        return len(self.active_rules)

    def get_rules_by_instance(self, x, order_by=0):
        x = torch.Tensor(x).to(self.device)
        sel = self._select_rules(x)
        rules = np.zeros((len(sel), self.k + 1))
        preds = []
        for i in range(len(sel)):
            rules[i, :] = self._params[sel[i]].data.numpy()
            preds.append(self.preds[sel[i]])

        preds = np.array(preds)
        preds = preds[np.lexsort((rules[:, order_by],))]
        rules = rules[np.lexsort((rules[:, order_by],))]
        return rules, preds

    def keep_top_rules(self, n=5, imbalance=None):
        rd = self.find_most_important_rules()
        rs = []
        for k in rd:
            if imbalance is None:
                rs.extend(rd[k][:n])
            else:
                rs.extend(rd[k][:round(n * imbalance[k])])
        rids = [r[1] for r in rs]
        self.active_rules = rids

    # ................................................
    def import_test_rules(self, ds_rules):
        """
        Replace current rule set by `ds_rules` (list[DSRule]).
        Masses are re-initialised randomly.
        """
        self._params, self.preds = [], []
        self.n = 0
        for r in ds_rules:
            self.add_rule(r)

    # ------------------------------------------------------------------ #
    #                           RULE PRUNING                             #
    # ------------------------------------------------------------------ #
    def _rule_coverage(self, rule_lambda, X_body):
        """Number of samples covered by `rule_lambda`."""
        return sum(rule_lambda(row) for row in X_body)

    @staticmethod
    def rule_score(m_vec):
        """
        Harmonic importance score from DST_JUCS (u_adj ⊗ top-2 ratio).
        """
        masses = m_vec[:-1].detach().cpu().numpy()
        unc = m_vec[-1].item()
        top2 = np.partition(masses, -2)[-2:]
        ratio = top2[-1] / (top2[-2] + 1e-3)
        return 1.0 - unc, ratio

    def _select_rules(self, x: torch.Tensor, idx: int = None) -> list[int]:
        """
        Internal: возвращает индексы правил, сработавших на x.
        """
        # похожим образом можно хранить rmap, но обычно у нас force_precompute=True
        sel = []
        x_np = x.cpu().numpy()
        for j, rule in enumerate(self.preds):
            if rule(x_np):
                sel.append(j)
        return sel

    # ────────────────────────────────────────────────────────────────
    #  ДОБАВИТЬ в класс DSModelMultiQ (после .normalize() или где угодно выше)
    # ----------------------------------------------------------------
    def _rule_line_with_masses(self, j:int) -> str:
        """Текст правила + его массы «c1 ... ck | unc» – для .dsb-дампа"""
        mass = " ".join(f"{v:.4f}" for v in self._params[j].detach().cpu().numpy())
        return f"{self.preds[j]}   ## {mass}"

    def import_rules_with_params(self, rules, params):
        """
        Replaces current rule set by `rules`
        and **keeps** the supplied tensors `params`
        (lengths must match, no grad needed any more).
        """
        if len(rules) != len(params):
            raise ValueError("rules/params length mismatch")

        self.preds, self._params = [], []
        self.n = 0
        for r, p in zip(rules, params):
            # p is a torch.Tensor with shape (k+1,)
            m_vec = p.detach().cpu().numpy()
            self.add_rule(r,
                          m_sing=m_vec[:-1].tolist(),
                          m_uncert=float(m_vec[-1]))

    # -------------------------------------------
    #  ЗАМЕНИТЬ две функции ниже
    # ----------------------------------------------------------------
    def sort_rules_by_quality(self, alpha: float = 0.5) -> None:
        """
        quality = (best-second) − α·m_unc     (чем выше – тем правило лучше)
        default-rule (последнее) оставляем внизу.
        """
        if not self.preds:
            return

        ms      = torch.stack(self._params).detach().cpu().numpy()   # (R, k+1)
        best    = ms[:,:-1].max(1)
        second  = np.partition(ms[:,:-1], -2, 1)[:,-2]
        quality = (best - second) - alpha*ms[:,-1]
        quality[-1] = -np.inf       # default → в конец

        order = np.argsort(-quality)
        self.preds   = [self.preds[i]   for i in order]
        self._params = [self._params[i] for i in order]

    # ----------------------------------------------------------------
    def prune_rules(self,
                    max_unc  : float = 0.80,
                    min_ratio: float = 6.0):
        """
        Remove rules with  m(Θ) > max_unc  or  (top/second) < min_ratio.
        Guarantees that at least ONE rule survives (the best one).
        Returns (kept, old).
        """
        old = self.n
        keep_p, keep_m, scores = [], [], []

        for r, m in zip(self.preds, self._params):
            m_np    = m.detach().cpu().numpy()
            top2    = np.sort(m_np[:-1])[-2:]        # two largest class masses
            ratio   = top2[1] / (top2[0] + 1e-12)    # top / second
            m_unc   = m_np[-1]
            score   = ratio - 0.5 * m_unc            # quality metric (used below)
            scores.append(score)

            if m_unc <= max_unc and ratio >= min_ratio:
                keep_p.append(r)
                keep_m.append(m)

        # --- if everything got filtered out → keep the best rule anyway ----
        if not keep_p:
            best = int(np.argmax(scores))
            keep_p.append(self.preds[best])
            keep_m.append(self._params[best])

        self.preds, self._params = keep_p, keep_m
        self.n = len(self.preds)
        return self.n, old
