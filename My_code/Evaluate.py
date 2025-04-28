import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from My_code.Datasets_loader import load_dataset
from My_code.Metrics_cul import compute_metrics
from My_code.Model import DempsterShaferModel, generate_model_path, load_or_initialize_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Функция подготовки данных
def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# Список датасетов и вариантов правил
datasets = [1, 2, 3, 4, 5, 6]
rule_variants = [3, 5]

# Параметры модели (например, lr)
lr = 1e-1

# Создаем структуру для сбора метрик:
# metrics_results[ds][rv] = {'Accuracy': ..., 'F1_Score': ..., 'ROC_AUC': ..., 'Avg_Precision': ...}
metrics_results = {}

for ds in datasets:
    print(f"Обработка датасета {ds}...")
    # Загружаем датасет
    X, y, feature_names = load_dataset(ds)
    X_train, y_train, X_test, y_test = prepare_data(X, y)
    metrics_results[ds] = {}
    for rv in rule_variants:
        print(f"  Вариант правил: {rv}")
        # Генерируем путь модели для данного датасета и варианта правил
        model_path = generate_model_path(ds, rv)
        # Загружаем (или инициализируем) модель
        model = load_or_initialize_model(
            input_size=X_train.shape[1],
            num_classes=2,
            model_path=model_path,
            lr=lr,
            feature_names=feature_names,
            X_train=X_train
        )
        model.eval()
        with torch.no_grad():
            final_test, _ = model(X_test)
            acc, f1, roc, ap = compute_metrics(final_test[:, :2], y_test)
        metrics_results[ds][rv] = {'Accuracy': acc, 'F1_Score': f1, 'ROC_AUC': roc, 'Avg_Precision': ap}
        print(f"    Метрики: Accuracy={acc:.3f}, F1={f1:.3f}, ROC_AUC={roc:.3f}, Avg_Precision={ap:.3f}")

# Создаем папку для результатов графиков, если её нет
results_folder = "My_code/results"
os.makedirs(results_folder, exist_ok=True)

# Список метрик, для которых будем строить графики
metrics_names = ["Accuracy", "F1_Score", "ROC_AUC", "Avg_Precision"]

# Для каждой метрики создаем один график с подграфиками для каждого датасета
for metric in metrics_names:
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(4 * n_datasets, 6), sharey=True)

    # Если датасетов всего один, преобразуем axes в список
    if n_datasets == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_metrics = metrics_results[ds]
        # Для каждого варианта правил получаем значение метрики
        values = []
        for rv in rule_variants:
            if rv in ds_metrics:
                values.append(ds_metrics[rv][metric])
            else:
                values.append(np.nan)
        x = np.arange(len(rule_variants))
        ax.bar(x, values, color=["skyblue", "salmon"])
        ax.set_title(f"Dataset {ds}")
        ax.set_xlabel("Rule Variant")
        ax.set_xticks(x)
        ax.set_xticklabels([str(rv) for rv in rule_variants])
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_ylabel(metric)  # устанавливаем ось Y для каждого подграфика (можно убрать дублирование)
        # Добавляем значения над столбцами
        for j, v in enumerate(values):
            ax.text(j, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle(f"{metric} для различных вариантов правил по датасетам", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(results_folder, f"{metric.lower()}.png")
    plt.savefig(save_path)
    print(f"Сохранён график '{metric}' в файл {save_path}")
    plt.close(fig)
