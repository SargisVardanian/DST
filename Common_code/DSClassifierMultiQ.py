import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Common_code.DSModelMultiQ import DSModelMultiQ


class DSClassifierMultiQ:
    """
    End-to-end Dempster–Shafer classifier:
      1) inducer RIPPER/FOIL → rules
      2) DST-weight training (SGD/Adam)
      3) optional pruning
      4) по окончании DST сохраняет ещё и «sorted» правила (до прунинга)
    """

    def __init__(self,
                 num_classes: int,
                 lr: float = 0.01,
                 max_iter: int = 50,
                 min_iter: int = 10,
                 min_dloss: float = 1e-4,
                 optim: str = "adam",
                 lossfn: str = "MSE",
                 batch_size: int = 4000,
                 precompute_rules: bool = False,
                 device: str = "cpu",
                 force_precompute: bool = False,
                 use_foil: bool = False,
                 batches_per_epoch: int = None,
                 rules_tag: str = "default"):
        self.k        = num_classes
        self.lr       = lr
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.min_dloss= min_dloss
        self.optim    = optim.lower()
        self.lossfn   = lossfn.upper()
        self.bs       = batch_size
        self.device   = torch.device(device)
        self.use_foil = use_foil
        self.bpe      = batches_per_epoch
        self.tag      = rules_tag

        self.model = DSModelMultiQ(
            k               = num_classes,
            precompute_rules= precompute_rules,
            device          = device,
            force_precompute= force_precompute,
            use_foil        = use_foil
        ).to(self.device)

        self.default_class_ = 0

    def fit(self, X, y, column_names=None,
            optimize_weights: bool=True,
            prune_after_fit: bool=False,
            prune_kwargs: dict=None,
            alpha_sort: float = 0.5,
            verbose: bool = True):
        """
        1) Если нет правил — индуцируем через RIPPER/FOIL.
        2) Обучаем массы (DST) → возвращаем losses, last_epoch.
        3) Сортируем правила по quality и сохраняем «sorted» дамп.
        4) Если prune_after_fit=True — пруним правила.
        Возвращаем (losses, last_epoch, sorted_rules_list).
        """
        prune_kwargs = prune_kwargs or {}
        sorted_rules = None

        # === 1) induce rules if empty ===
        if not self.model.preds:
            if column_names is None:
                raise ValueError("column_names must be provided on first fit()")
            if self.use_foil:
                rules = self.model.generate_foil_rules(X, y, column_names, batch_size=self.bs)
            else:
                rules = self.model.generate_ripper_rules(X, y, column_names, batch_size=self.bs)
            self.model.import_test_rules(rules)

        # majority-class fallback
        self.default_class_ = int(np.bincount(y).argmax())

        losses, last_epoch = None, None

        # === 2) DST-weight training ===
        if optimize_weights:
            opt_cls = torch.optim.Adam if self.optim == "adam" else torch.optim.SGD
            optimizer = opt_cls(self.model.parameters(), lr=self.lr)
            criterion = (torch.nn.CrossEntropyLoss()
                         if self.lossfn == "CE"
                         else torch.nn.MSELoss())

            # вставляем индекс образца в первый столбец
            X_idx = np.insert(X, 0, values=np.arange(len(X)), axis=1)

            if verbose:
                print("→ Starting mass optimisation …")
                losses, last_epoch = self._optimize(X_idx, y, optimizer, criterion)
                print(f"→ Finished – final loss {losses[-1]:.4f} @ epoch {last_epoch}")
        else:
            if verbose:
                print("→ Skipping mass optimisation.")

        # === 3) Sort + save sorted rules ===
        if optimize_weights:
            # сортируем по правилам качества
            self.model.sort_rules_by_quality(alpha=alpha_sort)
            # сохраняем список объектов DSRule после сортировки
            sorted_rules = list(self.model.preds)


        # === 4) Optional pruning ===
        if prune_after_fit:
            kept, old = self.model.prune_rules(**prune_kwargs)
            acc_train = (self.predict(X) == y).mean()
            print(f"Old {old}, Pruned → {kept} rules kept. Post-prune train acc: {acc_train:.3f}")

        return losses, last_epoch, sorted_rules

    def _optimize(self, X_idx, y, optimizer, criterion):
        self.model.train()
        self.model.clear_rmap()

        # готовим DataLoader по индексированному X
        Xt = torch.FloatTensor(X_idx)
        if self.lossfn == "CE":
            Yt = torch.LongTensor(y)
        else:
            Yt = torch.nn.functional.one_hot(torch.LongTensor(y), self.k).float()

        ds = TensorDataset(Xt, Yt)
        dl = DataLoader(ds, batch_size=self.bs,
                        shuffle=True, num_workers=0, pin_memory=False)

        losses = []
        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            seen = 0
            for Xi, Yi in dl:
                Xi, Yi = Xi.to(self.device), Yi.to(self.device)
                optimizer.zero_grad()
                out = self.model(Xi)
                loss = criterion(out, Yi)
                loss.backward()
                optimizer.step()
                self.model.normalize()

                batch_n = Xi.size(0)
                epoch_loss += loss.item() * batch_n
                seen += batch_n

                if self.bpe and seen >= self.bpe * self.bs:
                    break

            epoch_loss /= seen
            losses.append(epoch_loss)
            print(f"\rEpoch {epoch+1}  loss={epoch_loss:.5f} ", end="")

            if epoch >= self.min_iter and abs(losses[-2] - losses[-1]) < self.min_dloss:
                break

        return losses, epoch

    def predict(self, X):
        """
        Полный forward → argmax по классам.
        """
        self.model.eval()
        self.model.clear_rmap()
        X_idx = np.insert(X, 0, np.arange(len(X)), axis=1)
        with torch.no_grad():
            out = self.model(torch.tensor(X_idx, dtype=torch.float32, device=self.device))
        return out.argmax(dim=1).cpu().numpy()

    def predict_rules_only(self, X):
        """
        Чистый RIPPER/FOIL-стиль: первое подходящее правило отдаёт класс.
        """
        preds = np.full(len(X), self.default_class_, dtype=int)
        for i, row in enumerate(X):
            for r in self.model.preds:
                if r(row):
                    # DSRule.caption: "Class {cls}: …"
                    preds[i] = int(r.caption.split()[1].rstrip(":"))
                    break
        return preds

    # def predict_voting(self, X):
    #     """
    #     Voting-based prediction: for each sample, average the DST mass vectors of
    #     all fired rules; choose class with highest mean mass. If tie or no fired rules,
    #     return default class.
    #     """
    #     # Clear any previous rmap if used
    #     self.model.clear_rmap()
    #     # Prepare array to collect votes
    #     X_arr = np.asarray(X)
    #     n_samples = X_arr.shape[0]
    #     # For each sample, accumulate mass vectors and count
    #     votes = np.zeros((n_samples, self.k), dtype=float)
    #     counts = np.zeros(n_samples, dtype=int)
    #
    #     # Precompute masses for each rule on each sample
    #     for rule, m in zip(self.model.preds, self.model._params):
    #         # Determine which samples fire this rule
    #         fired = np.array([bool(rule(x)) for x in X_arr])
    #         if not fired.any():
    #             continue
    #         # Extract mass vector (exclude uncertainty index)
    #         m_np = m.detach().cpu().numpy()[:-1]
    #         # Add to votes
    #         votes[fired] += m_np
    #         counts[fired] += 1
    #
    #     # Compute average votes and select class per sample
    #     preds = np.full(n_samples, self.default_class_, dtype=int)
    #     for i in range(n_samples):
    #         if counts[i] > 0:
    #             avg = votes[i] / counts[i]
    #             # Check for unique max
    #             top = np.max(avg)
    #             winners = np.where(avg == top)[0]
    #             if len(winners) == 1:
    #                 preds[i] = int(winners[0])
    #             else:
    #                 preds[i] = self.default_class_
    #     return preds
