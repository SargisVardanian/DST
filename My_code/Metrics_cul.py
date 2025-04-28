import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score
)

# --- МЕСТА ДЛЯ СОХРАНЕНИЯ МЕТРИК ---
train_metrics_history = {
    "accuracy": [],
    "f1": [],
    "roc_auc": [],
    "avg_precision": []
}
val_metrics_history = {
    "accuracy": [],
    "f1": [],
    "roc_auc": [],
    "avg_precision": []
}

# --- МЕСТА ДЛЯ СОХРАНЕНИЯ МАСС ПРАВИЛ ---
rules_masses_history = []  # Список списков: каждая внутренняя структура содержит массы всех правил на эпоху

def compute_metrics(final_m, y_true):
    """
    Вычисляем Accuracy, F1, ROC-AUC, Average Precision
    final_m: (N,3) => [m0,m1,m_unk]
    y_true: (N,) -> {0,1}
    """
    # Предсказываем класс по argmax(final_m[:, :2])
    with torch.no_grad():
        preds_class = final_m[:, :2].argmax(dim=1).cpu().numpy()
        # Для ROC AUC и AveragePrecision нужен "скор" класса 1
        # Берём final_m[:,1] как вероятности класса 1 (условное допущение)
        scores_class1 = final_m[:, 1].detach().cpu().numpy()

    y_true_np = y_true.cpu().numpy()

    acc = accuracy_score(y_true_np, preds_class)
    f1 = f1_score(y_true_np, preds_class)
    # для roc_auc нужен хотя бы один класс(0) и (1) в y_true
    try:
        roc_auc = roc_auc_score(y_true_np, scores_class1)
    except ValueError:
        roc_auc = 0.0
    # average precision
    avg_prec = average_precision_score(y_true_np, scores_class1)

    return acc, f1, roc_auc, avg_prec

def compute_binary_metrics(m: torch.Tensor, y: torch.Tensor):
    """
    :param m: (B,3) => m0,m1,m_unk
    :param y: (B,) => 0 or 1
    Возвращаем accuracy, f1, ...
    Для упрощения считаем, что класс = argmax(m[0], m[1])
    """
    # Предсказанный класс
    pred_cls = m[:, :2].argmax(dim=1)  # 0/1
    correct = (pred_cls == y).sum().item()
    acc = correct / y.size(0)

    # Примитивный f1
    # (можно подключить from sklearn.metrics import f1_score, ROC-AUC итд)
    # Здесь просто мини-пример
    tp = ((pred_cls==1) & (y==1)).sum().item()
    fp = ((pred_cls==1) & (y==0)).sum().item()
    fn = ((pred_cls==0) & (y==1)).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision*recall / (precision+recall+1e-9)

    return acc, f1  # Можно расширять
