# DSClassifierMultiQ.py

import time
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn

from Common_code.DSModelMultiQ import DSModelMultiQ
from Common_code.DSRipper import DSRipper   # нужно для совместимости
from Common_code.core import rules_to_dsb
import os


class DSClassifierMultiQ:
    """
    End-to-end Dempster–Shafer classifier.
    Генерирует правила (RIPPER/FOIL), оптимизирует массы, при желании пруит.
    """

    def __init__(self,
                 num_classes: int,
                 lr: float = 0.05,
                 max_iter: int = 50,
                 min_iter: int = 5,
                 min_dloss: float = 1e-4,
                 optim: str = "adam",
                 lossfn: str = "MSE",
                 batch_size: int = 4000,
                 num_workers: int = 1,
                 precompute_rules: bool = False,
                 device: str = "cpu",
                 force_precompute: bool = False,
                 use_foil: bool = False,
                 batches_per_epoch: int = None):
        """
        Параметры:
          num_classes     — число классов (k).
          lr               — learning rate для оптимизации DST-весов.
          max_iter         — максимальное число эпох (итераций) оптимизации.
          min_iter         — минимум итераций перед досрочным стопом.
          min_dloss        — порог изменения loss (если delta < min_dloss) для остановки.
          optim            — «adam» или «sgd».
          lossfn           — «MSE» (Mean Squared Error) или «CE» (CrossEntropy).
          batch_size       — batch_size для DataLoader’а при оптимизации ваг (DST).
          num_workers      — число воркеров для DataLoader’а (рекомендуется = 0 при MPS).
          precompute_rules — если True, правила будут закешированы заранее (быстрее inference).
          device           — строка вида «cpu», «cuda», «mps».
          force_precompute  — если True, DSModelMultiQ заранее кеширует, какие правила сработают на каких образцах.
          use_foil         — если True, при генерации правил применяется FOIL; иначе RIPPER.
          batches_per_epoch — сколько мини-батчей из DataLoader обрабатывать в каждой эпохе.
                              Если None, цикл идёт до конца DataLoader (как раньше).
        """
        self.k                = num_classes
        self.lr               = lr
        self.max_iter         = max_iter
        self.min_iter         = min_iter
        self.min_dloss        = min_dloss
        self.optim            = optim.lower()
        self.lossfn           = lossfn.upper()
        self.batch_size       = batch_size
        self.num_workers      = num_workers
        self.device           = torch.device(device)
        self.use_foil         = use_foil
        self.batches_per_epoch = batches_per_epoch  # новоe поле

        # основная DS-модель (правила + массы)
        self.model = DSModelMultiQ(
            k=num_classes,
            precompute_rules=precompute_rules,
            device=self.device,
            force_precompute=force_precompute,
            use_foil=use_foil
        ).to(self.device)

        # fallback по большинству (majority-vote)
        self.default_class_ = 0

    def fit(self, X, y,
            column_names=None,
            add_single_rules=False,
            single_rules_breaks=2,
            add_mult_rules=False,
            optimize_weights=True,
            prune_after_fit=False,
            prune_kwargs=None):
        """
        1. (опционально) добавляем «handcrafted» правила
        2. генерируем RIPPER или FOIL (если списка правил нет)
        3. оптимизируем массы (DST) через градиентный спуск (опционально)
        4. пруним низкоэффективные правила (опционально)
        """
        prune_kwargs = prune_kwargs or {}

        # –– handcrafted extra rules ––
        if add_single_rules:
            self.model.generate_statistic_single_rules(
                X, breaks=single_rules_breaks, column_names=column_names)
        if add_mult_rules:
            self.model.generate_mult_pair_rules(X, column_names=column_names)

        # –– автоматическая индукция правил ––
        if not self.model.preds:
            print(f"→ Inducing {'FOIL' if self.use_foil else 'RIPPER'} rules …")
            if self.use_foil:
                rules = self.model.generate_foil_rules(
                    X, y, column_names, batch_size=self.batch_size)
            else:
                rules = self.model.generate_ripper_rules(
                    X, y, column_names, batch_size=self.batch_size)
            self.model.import_test_rules(rules)
        else:
            print("Existing rules detected – skipping induction.")

        # определяем дефолтный класс (majority)
        self.default_class_ = int(np.bincount(y).argmax())

        # accuracy по правилам (без DST) на train
        acc0 = (self.predict_rules_only(X) == y).mean()
        print(f"Rule-only train accuracy: {acc0:.3f}")

        # –– оптимизация масс DST ––
        losses, last_epoch = None, None
        if optimize_weights:
            opt_cls  = torch.optim.Adam if self.optim == "adam" else torch.optim.SGD
            optimizer = opt_cls(self.model.parameters(), lr=self.lr)
            criterion = (torch.nn.CrossEntropyLoss()
                         if self.lossfn == "CE"
                         else torch.nn.MSELoss())

            # добавляем индекс столбца слева (нужно для DSModelMultiQ.forward)
            X_idx = np.insert(X, 0, values=np.arange(len(X)), axis=1)

            print("→ Starting mass optimisation …")
            losses, last_epoch = self._optimize(X_idx, y, optimizer, criterion)
            print(f"→ Finished – final loss {losses[-1]:.4f} @ epoch {last_epoch}")
        else:
            print("→ Skipping mass optimisation.")

        # –– прунинг ––
        if prune_after_fit:
            kept, old = self.model.prune_rules(X, **prune_kwargs)
            acc1 = (self.predict(X) == y).mean()
            print(f"Old {old}, Pruned → {kept} rules kept. Post-prune train acc: {acc1:.3f}")

        return losses, last_epoch

    def _optimize(self, X, y, optimizer, criterion):
        """Train rule masses using mini-batch gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Training data with sample indices prepended.
        y : np.ndarray
            Target labels.
        optimizer : torch.optim.Optimizer
            Optimiser instance (Adam or SGD).
        criterion : torch.nn.Module
            Loss function.

        Notes
        -----
        Processes only a limited number of mini-batches per epoch if
        ``batches_per_epoch`` is set.  This loop can be slow on large datasets
        because each batch involves forward and backward passes through all
        active rules.
        """
        losses = []
        self.model.train()
        self.model.clear_rmap()

        # 1) Создаем CPU-тензоры, которые пойдут в DataLoader
        Xt_cpu = torch.FloatTensor(X)  # на CPU
        if self.lossfn == "CE":
            yt_cpu = torch.LongTensor(y)  # на CPU
        else:
            yt_cpu = torch.nn.functional.one_hot(
                torch.LongTensor(y), self.k
            ).float()  # на CPU

        dataset = torch.utils.data.TensorDataset(Xt_cpu, yt_cpu)
        loader  = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,         # желательно перемешать в начале каждой эпохи
            num_workers=0,        # обязательно 0 → MPS не умеет шарить CPU-тензоры между процессами
            pin_memory=False      # отключаем pin_memory, т.к. MPS/PyTorch не поддерживает это
        )

        total_samples = len(dataset)

        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            processed_batches = 0

            for Xi_cpu, yi_cpu in loader:
                Xi = Xi_cpu.to(self.device)
                yi = yi_cpu.to(self.device)

                optimizer.zero_grad()
                preds = self.model(Xi)
                loss = criterion(preds, yi)
                loss.backward(retain_graph=True)
                optimizer.step()
                self.model.normalize()

                epoch_loss += loss.item() * Xi.size(0) / total_samples
                processed_batches += 1

                # если задано ограничение по числу батчей в эпохе ― выходим
                if self.batches_per_epoch is not None and \
                   processed_batches >= self.batches_per_epoch:
                    break

            losses.append(epoch_loss)
            print(f"\rEpoch {epoch+1}  loss={epoch_loss:.5f}  "
                  f"(batches used: {processed_batches})", end="")

            # досрочная остановка по delta loss
            if epoch >= self.min_iter and abs(losses[-2] - epoch_loss) < self.min_dloss:
                break

        print()  # перевод строки после вывода прогресса
        return losses, epoch

    def predict(self, X, one_hot=False):
        """
        DST-предсказание: возвращает либо распределение (one_hot=True),
        либо argmax-класс (one_hot=False).
        """
        self.model.eval()
        self.model.clear_rmap()

        X_idx = np.insert(X, 0, values=np.arange(len(X)), axis=1)
        with torch.no_grad():
            out = self.model(torch.Tensor(X_idx).to(self.device))
            if one_hot:
                return out.cpu().numpy()
            else:
                return torch.argmax(out, 1).cpu().numpy()

    def predict_rules_only(self, X):
        """
        First-match rule prediction (первое сработавшее правило) с fallback → majority-class.
        """
        preds = np.full(len(X), self.default_class_, dtype=int)
        for i, row in enumerate(X):
            for rule in self.model.preds:
                if rule(row):
                    lbl = int(rule.caption.split()[1].rstrip(':'))
                    preds[i] = lbl
                    break
        return preds
