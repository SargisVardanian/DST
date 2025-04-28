import os
import random
import torch
import torch.nn as nn
from graphviz import Digraph
import torch.nn.functional as F


class DempsterShaferModel(nn.Module):
    """Dempster-Shafer модель с правилами и оптимизацией масс."""
    class Node:
        def __init__(self, value, left=None, right=None, indices=None):
            self.value = value
            self.left = left
            self.right = right
            self.indices = indices if indices is not None else []

    def __init__(self, input_size, num_classes, lr=2e-3, use_autograd=True):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.lr = lr
        self.rules = []
        self.rules_masses = nn.ParameterList()

        self.use_autograd = use_autograd  # если True – используем стандартный backprop
        self.optimizer = None

    def generate_rules(self, X, feature_names, rule_variant=5, discrete_threshold=8, discrete_features=None):
        """
        Для непрерывных признаков (число уникальных значений > discrete_threshold) генерируются:
          - Если rule_variant==3: 3 правила: "<", "~", ">"
          - Если rule_variant==5: дополнительно создаются 2 правила ("<~" и "~>")
        Для признаков, указанных в discrete_features (или если число уникальных значений <= discrete_threshold),
        генерируются правила вида "признак == значение" для каждого уникального значения.
        """
        if discrete_features is None:
            discrete_features = []

        rules = []
        for i, name in enumerate(feature_names):
            unique_vals = torch.unique(X[:, i])
            if (i in discrete_features) or (unique_vals.numel() <= discrete_threshold):
                # Дискретный признак: генерируем правило для каждого уникального значения
                print(f"{name}, unique_vals: ", unique_vals)
                for val in unique_vals:
                    rules.append({
                        "description": f"{name} == {val.item()}",
                        "feature_index": i,
                        "operator": "==",
                        "threshold": val.item(),
                        "tolerance": 1e-6,
                        "mass": nn.Parameter(torch.tensor([0.05, 0.05, 0.9], requires_grad=True).clone(), requires_grad=True),
                        "category": val.item(),
                        "predicate": None
                    })
            else:
                # Непрерывный признак: вычисляем среднее, std и tol
                mean_val = X[:, i].mean().item()
                std_val = X[:, i].std().item()
                tol = std_val / 2
                init_mass = torch.tensor([0.05, 0.05, 0.9], requires_grad=True)
                # Правило "<"
                rules.append({
                    "description": f"{name} < {mean_val:.4f}",
                    "feature_index": i,
                    "operator": "<",
                    "threshold": mean_val,
                    "tolerance": 0.0,
                    "mass": nn.Parameter(init_mass.clone(), requires_grad=True),
                    "predicate": None
                })
                # Правило "~"
                rules.append({
                    "description": f"{name} ~ {mean_val:.4f} ± {tol:.4f}",
                    "feature_index": i,
                    "operator": "~",
                    "threshold": mean_val,
                    "tolerance": tol,
                    "mass": nn.Parameter(init_mass.clone(), requires_grad=True),
                    "predicate": None
                })
                # Правило ">"
                rules.append({
                    "description": f"{name} > {mean_val:.4f}",
                    "feature_index": i,
                    "operator": ">",
                    "threshold": mean_val,
                    "tolerance": 0.0,
                    "mass": nn.Parameter(init_mass.clone(), requires_grad=True),
                    "predicate": None
                })
                if rule_variant == 5:
                    # Правило "<~"
                    rules.append({
                        "description": f"{name} between {mean_val - 3*tol/2:.4f} and {mean_val - tol/2:.4f}",
                        "feature_index": i,
                        "operator": "<~",
                        "threshold": mean_val - tol,
                        "tolerance": tol/2,
                        "mass": nn.Parameter(init_mass.clone(), requires_grad=True),
                        "predicate": None
                    })
                    # Правило "~>"
                    rules.append({
                        "description": f"{name} between {mean_val+tol/2:.4f} and {mean_val + 3*tol/2:.4f}",
                        "feature_index": i,
                        "operator": "~>",
                        "threshold": mean_val + tol,
                        "tolerance": tol/2,
                        "mass": nn.Parameter(init_mass.clone(), requires_grad=True),
                        "predicate": None
                    })
                elif rule_variant != 3:
                    raise ValueError("Unsupported rule_variant (для непрерывных переменных поддерживаются только варианты 3 и 5)")
        self.rules = rules
        self.rules_masses = nn.ParameterList([r["mass"] for r in rules])
        self._rebuild_all_predicates()

    def _rebuild_all_predicates(self):
        for rule in self.rules:
            i = rule["feature_index"]
            op = rule["operator"]
            thr = rule["threshold"]
            tol = rule["tolerance"]
            if op == "==":
                # Для дискретных правил сравнение выполняем с допуском
                rule["predicate"] = lambda x, i=i, c=thr, dt=tol: abs(x[i] - c) <= dt
            elif op == "<":
                rule["predicate"] = lambda x, i=i, t=thr: x[i] < t
            elif op == ">":
                rule["predicate"] = lambda x, i=i, t=thr: x[i] > t
            elif op == "~":
                rule["predicate"] = lambda x, i=i, c=thr, dt=tol: abs(x[i] - c) <= dt
            elif op == "<~":
                rule["predicate"] = lambda x, i=i, c=thr, dt=tol: abs(x[i] - c) <= dt
            elif op == "~>":
                rule["predicate"] = lambda x, i=i, c=thr, dt=tol: abs(x[i] - c) <= dt
            else:
                rule["predicate"] = None

    def export_rules(self):
        return [{
            "description": r["description"],
            "feature_index": r["feature_index"],
            "operator": r["operator"],
            "threshold": r["threshold"],
            "tolerance": r["tolerance"],
            "mass_values": r["mass"].detach().cpu().numpy().tolist()
        } for r in self.rules]

    def import_rules(self, rule_list):
        self.rules = []
        self.rules_masses = nn.ParameterList()
        for rd in rule_list:
            mass_param = nn.Parameter(torch.tensor(rd["mass_values"], dtype=torch.float32), requires_grad=True)
            new_rule = {
                "description": rd["description"],
                "feature_index": rd["feature_index"],
                "operator": rd["operator"],
                "threshold": rd["threshold"],
                "tolerance": rd["tolerance"],
                "mass": mass_param,
                "predicate": None
            }
            self.rules.append(new_rule)
            self.rules_masses.append(mass_param)
        self._rebuild_all_predicates()

    def save_model(self, path="best_dempster_shafer_model.pth"):
        checkpoint = {
            "state_dict": self.state_dict(),
            "rules_list": self.export_rules()
        }
        torch.save(checkpoint, path)
        print(f"[INFO] Model and rules saved to {path}")

    def save_rules(self, filepath="rules_saved.txt"):
        with open(filepath, "w", encoding="utf-8") as f:
            for i, (rule, param) in enumerate(zip(self.rules, self.rules_masses)):
                mass_np = param.detach().cpu().numpy()
                desc = rule.get("description", "No description")
                f.write(f"Rule {i+1}: {desc}\n")
                f.write(f"  mass = {mass_np}, sum={mass_np.sum():.4f}\n\n")
        print(f"[INFO] Rules saved to: {filepath}")

    def load_model(self, path="best_dempster_shafer_model.pth", strict=True):
        if not os.path.exists(path):
            print(f"[WARN] File {path} not found.")
            return False
        checkpoint = torch.load(path, map_location="cpu")
        if "state_dict" not in checkpoint or "rules_list" not in checkpoint:
            print("[ERROR] Checkpoint format error.")
            return False
        self.import_rules(checkpoint["rules_list"])
        try:
            self.load_state_dict(checkpoint["state_dict"], strict=strict)
        except RuntimeError as e:
            print(f"[ERROR] Error loading state_dict: {e}")
            return False
        print(f"[INFO] Model and rules loaded from {path}.")
        return True

    # -----------------------
    # forward part
    # -----------------------
    @staticmethod
    def combine_masses_dempster(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        conflict = m1[0] * m2[1] + m1[1] * m2[0]
        if conflict >= 1.0 - eps:
            return torch.ones_like(m1) / len(m1)
        out = torch.zeros_like(m1)
        out[0] = m1[-1] * m2[0] + m1[0] * m2[-1] + m1[0] * m2[0]
        out[1] = m1[-1] * m2[1] + m1[1] * m2[-1] + m1[1] * m2[1]
        out[-1] = m1[-1] * m2[-1]
        return out / (1.0 - conflict + eps)

    def build_combine_tree(self, masses, indices):
        if len(masses) == 1:
            return self.Node(masses[0], indices=[indices[0]])
        elif len(masses) == 2:
            l, r = masses
            combined = self.combine_masses_dempster(l, r)
            return self.Node(combined,
                             left=self.Node(l, indices=[indices[0]]),
                             right=self.Node(r, indices=[indices[1]]))
        else:
            mid = len(masses) // 2
            left_node = self.build_combine_tree(masses[:mid], indices[:mid])
            right_node = self.build_combine_tree(masses[mid:], indices[mid:])
            combined = self.combine_masses_dempster(left_node.value, right_node.value)
            return self.Node(combined, left=left_node, right=right_node)

    def forward(self, x, shuffle_combine=False):
        B = x.size(0)
        final_m, nodes = [], []
        for i in range(B):
            # Используем cpu().detach().numpy() для вычисления предикатов (это не участвует в графе)
            sample = x[i].cpu().detach().numpy()
            active = [j for j, r in enumerate(self.rules) if r["predicate"] and r["predicate"](sample)]
            masses = [self.rules_masses[j] for j in active]
            if shuffle_combine:
                shuf = masses[:]
                random.shuffle(shuf)
                combined = shuf[0]
                node = self.Node(combined, indices=[active[0]])
                for k in range(1, len(shuf)):
                    combined = self.combine_masses_dempster(combined, shuf[k])
                    node = self.Node(combined, left=node, right=self.Node(shuf[k], indices=[active[k]]))
            else:
                node = self.build_combine_tree(masses, active)
            nodes.append(node)
            final_m.append(node.value)
        final_m = torch.stack(final_m, dim=0)
        # Нормализуем только классовые массы (первые два элемента) через softmax
        final_m[:, :2] = F.softmax(final_m[:, :2], dim=1)
        return final_m, nodes


    # -----------------------
    # backward and optimazetion part
    # -----------------------

    @staticmethod
    def combine_masses_dempster_jacobian(lval: torch.Tensor, rval: torch.Tensor):
        # Распаковка компонентов: a, b, c = m₁; d, e, f = m₂
        a, b, c = lval
        d, e, f = rval
        eps = 1e-6
        # Вычисляем конфликт и знаменатель
        conflict = a * e + b * d
        denom = 1.0 - conflict + eps

        # Вычисляем промежуточный вектор [Nx0, Nx1, Nx2]
        Nx0 = c * d + a * (f + d)
        Nx1 = c * e + b * (f + e)
        Nx2 = c * f
        out_vec = torch.stack([Nx0, Nx1, Nx2])

        # Для m₁: аналитически получаем матрицу G1
        G1 = torch.tensor([[f + d, 0.0, d],
                           [0.0, f + e, e],
                           [0.0, 0.0, f]],
                          dtype=lval.dtype, device=lval.device)
        # Производные знаменателя по m₁: d(denom)/d(a) = -e, d(denom)/d(b) = -d, d(denom)/d(c) = 0.
        D1 = torch.tensor([-e, -d, 0.0],
                          dtype=lval.dtype, device=lval.device)
        Jleft = G1 / denom - torch.outer(out_vec, D1) / (denom ** 2)

        G2 = torch.tensor([[c + a, 0.0, a],
                           [0.0, c + b, b],
                           [0.0, 0.0, c]],
                          dtype=rval.dtype, device=rval.device)
        # Производные знаменателя по m₂: d(denom)/d(d) = -b, d(denom)/d(e) = -a, d(denom)/d(f) = 0.
        D2 = torch.tensor([-b, -a, 0.0],
                          dtype=rval.dtype, device=rval.device)
        Jright = G2 / denom - torch.outer(out_vec, D2) / (denom ** 2)

        return Jleft, Jright

    def backprop_into_tree(self, node, dValue):
        if node.left is None and node.right is None:
            if node.indices:
                j = node.indices[0]
                # if self.rules_masses[j].grad is None:
                #     self.rules_masses[j].grad = torch.zeros_like(self.rules_masses[j], device=self.rules_masses[j].device)
                self.rules_masses[j].grad += dValue
            return
        lval, rval = node.left.value, node.right.value
        Jleft, Jright = self.combine_masses_dempster_jacobian(lval, rval)
        dLeft = torch.matmul(Jleft, dValue)
        dRight = torch.matmul(Jright, dValue)
        self.backprop_into_tree(node.left, dLeft)
        self.backprop_into_tree(node.right, dRight)

    def backward_optim(self, final_m, loss, nodes):
        if self.use_autograd:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                for j, param in enumerate(self.rules_masses):
                    param.copy_(self.project_simplex(param))
            return self
        else:
            for param in self.rules_masses:
                if param.grad is not None:
                    param.grad.zero_()
                else:
                    param.grad = torch.zeros_like(param, device=param.device)
            dLoss_dfinal = torch.autograd.grad(loss, final_m, retain_graph=True)[0]
            for i in range(final_m.size(0)):
                self.backprop_into_tree(nodes[i], dLoss_dfinal[i])
            with torch.no_grad():
                for j, param in enumerate(self.rules_masses):
                    param.sub_(self.lr * param.grad)
                    param.copy_(self.project_simplex(param))
            return self

    @staticmethod
    def project_simplex(v, s=1.0):
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0) - s
        ind = torch.arange(1, len(v)+1, device=v.device, dtype=v.dtype)
        cond = u - cssv / ind > 0
        if cond.sum() == 0:
            return torch.zeros_like(v)
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
        return torch.clamp(v - theta, min=0)

    def visualize_tree_of_nodes(self, node, graph=None, parent_id=None, final_mass=None, y_true=None):
        if graph is None:
            graph = Digraph(format='png')

        if node is None:
            return graph

        # Создаём узел для текущего элемента. Добавляем массу, а если имеются индексы правил,
        # то выводим номера правил (например, Rule 3, Rule 7 и т.д.)
        label_str = f"mass={node.value.detach().cpu().numpy()}"
        if node.indices:
            # Преобразуем индексы в строку вида "Rule 3, Rule 7"
            rule_indices = ", ".join([f"Rule {i+1}" for i in node.indices])
            rule_descriptions = ", ".join([self.rules[i]["description"] for i in node.indices])
            label_str += f"\nindices: {rule_indices}\nrules: {rule_descriptions}"
        current_id = f"node_{id(node)}"
        graph.node(current_id, label=label_str)

        if parent_id is not None:
            graph.edge(parent_id, current_id)

        # Рекурсивно обходим дочерние узлы
        graph = self.visualize_tree_of_nodes(node.left, graph, current_id)
        graph = self.visualize_tree_of_nodes(node.right, graph, current_id)

        # Если это корень, добавляем итоговый узел с YPredict и YTrue
        if parent_id is None and final_mass is not None and y_true is not None:
            y_pred = int(torch.argmax(final_mass[:2]).item())  # Предсказанный класс
            final_label = f"YPredict: {y_pred}\nYTrue: {y_true}"
            final_node_id = f"final_node"
            graph.node(final_node_id, label=final_label)
            graph.edge(current_id, final_node_id)

        return graph



def generate_model_path(save_i, rule_variant):
    return f'My_code/{save_i}:/{rule_variant}:/model.pth'


def load_or_initialize_model(input_size, num_classes,
                             model_path='best_dempster_shafer_model0.pth',
                             lr=2e-3,
                             feature_names=None,
                             X_train=None):
    model = DempsterShaferModel(input_size, num_classes, lr)
    if os.path.exists(model_path):
        success = model.load_model(model_path)
        if success:
            print("Model loaded")
        else:
            # Если загрузка не удалась, сгенерировать правила заново.
            model.generate_rules(X_train, feature_names)
            print("Model didn't load")
    else:
        model.generate_rules(X_train, feature_names)
        print("Model didn't load")
    return model
