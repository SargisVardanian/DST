from My_code.Datasets_loader import *
from My_code.Model import *
from My_code.Metrics_cul import *
import pickle
import matplotlib.pyplot as plt
import subprocess
import json
import os

# def generate_model_path(save_i):
#     return 'exp{tol=s_vdel2,(5)}_0/' + f'best_dempster_shafer_model{save_i}.pth'

def generate_model_path(save_i, rule_variant):
    return f'/Users/sargisvardanyan/PycharmProjects/DST/My_code/{save_i}:/{rule_variant}:/model.pth'


# num_classes = 21
# save_i = 40
# rule_variant=5
# model_path = generate_model_path(save_i, rule_variant)
# lr = 0.0000001


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

#
#
# model = load_or_initialize_model(input_size=X_train.shape[1],
#                                  num_classes=num_classes,
#                                  model_path=model_path,
#                                  lr=lr,
#                                  feature_names=data.feature_names,
#                                  X_train=X_train[:10]
#                                  )

def prepare_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# Функция, выполняющая эксперимент на выбранном датасете
def run_experiment(dataset_id, rule_variant=5, num_epochs=30, lr=3e-1, batch_size=16):
    # Загружаем датасет
    X, y, feature_names = load_dataset(dataset_id)
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = prepare_data(X, y)

    # Инициализируем модель с начальным lr
    model = DempsterShaferModel(input_size=X_train_tensor.shape[1], num_classes=2, lr=lr)
    print("feature_names:", feature_names)
    model.generate_rules(X_train_tensor, feature_names, rule_variant=rule_variant)
    print(f"Количество правил: {len(model.rules)}")
    model.optimizer = torch.optim.Adam(model.rules_masses, lr=model.lr)

    num_samples = X_train_tensor.size(0)
    n_batches = (num_samples + batch_size - 1) // (batch_size * 4)
    criterion_net = torch.nn.CrossEntropyLoss()

    epochs_list = []
    test_accuracy_history = []
    test_loss_history = []
    rules_masses_history = []

    best_test_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        perm = torch.randperm(num_samples)
        X_train_shuf = X_train_tensor[perm]
        y_train_shuf = y_train_tensor[perm]

        for b in range(n_batches):
            start_b = b * batch_size
            end_b = start_b + batch_size
            xb = X_train_shuf[start_b:end_b]
            yb = y_train_shuf[start_b:end_b]

            # Forward pass
            final_m, chain_states = model.forward(xb, shuffle_combine=True)
            loss = criterion_net(final_m[:, :2], yb)
            train_loss += loss.item()

            # Backward and optimization
            model = model.backward_optim(final_m, loss, chain_states)

        avg_train_loss = train_loss / n_batches

        # Test evaluation
        model.eval()
        with torch.no_grad():
            final_test, _ = model(X_test_tensor)
            acc_te, f1_te, roc_te, ap_te = compute_metrics(final_test[:, :2], y_test_tensor)
            loss_test = criterion_net(final_test[:, :2], y_test_tensor).item()

        epochs_list.append(epoch + 1)
        test_accuracy_history.append(acc_te)
        test_loss_history.append(loss_test)
        rules_masses_history.append([param.clone().detach().cpu().numpy() for param in model.rules_masses])
        # model.lr *= 0.8

        if epoch % 10 == 0:
            model.save_model(path=generate_model_path(dataset_id, rule_variant))
            model.save_rules(f'My_code/{dataset_id}:/{rule_variant}:/rules_saved.txt')
            print(f"Saved epoch {epoch+1}")

        print(f"Epoch {epoch+1}/{num_epochs}: TrainLoss={avg_train_loss:.4f} | TestLoss={loss_test:.4f} | TestAcc={acc_te:.3f} | TestF1={f1_te:.3f}")

    total_time = time.time() - start_time
    model.eval()
    with torch.no_grad():
        final_test, _ = model(X_test_tensor)
        acc_te, f1_te, roc_te, ap_te = compute_metrics(final_test[:, :2], y_test_tensor)

    results = {
        "Dataset_ID": dataset_id,
        "Rule_Variant": rule_variant,
        "Accuracy": acc_te,
        "F1_Score": f1_te,
        "ROC_AUC": roc_te,
        "Avg_Precision": ap_te,
        "Training_Time": total_time,
        "Min_Loss": best_test_loss
    }

    history = {
        "epoch": epochs_list,
        "test_accuracy": test_accuracy_history,
        "test_loss": test_loss_history,
        "training_time": total_time,
        "Dataset_ID": dataset_id,
        "Rule_Variant": rule_variant
    }
    return results, history, rules_masses_history


# -----------------------
# Func for saving data
# -----------------------
def load_history(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return pd.DataFrame()

def append_history(new_df, filename):
    existing_df = load_history(filename)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(filename, index=False)
    return combined_df

def load_summary(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return pd.DataFrame()

def append_summary(new_df, filename):
    existing_df = load_summary(filename)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(filename, index=False)
    return combined_df

# -----------------------
# Main part
# -----------------------
if __name__ == '__main__':
    dataset_id = 6
    rule_variants = [3, 5]  # 3 или 5

    # Параметры эксперимента:
    num_epochs = 30
    lr = 4e-1
    batch_size = 16

    for current_rule_variant in rule_variants:
        # # Запуск эксперимента
        results, history, _ = run_experiment(dataset_id, rule_variant=current_rule_variant,
                                              num_epochs=num_epochs, lr=lr, batch_size=batch_size)

        # Сохраняем результаты эксперимента
        df_result = pd.DataFrame([results])
        df_result = append_summary(df_result, "My_code/results/experiment_summary.csv")

        # Сохраняем историю по эпохам
        df_history = pd.DataFrame({
            "Epoch": history["epoch"],
            "Test_Accuracy": history["test_accuracy"],
            "Test_Loss": history["test_loss"],
            "Dataset_ID": history["Dataset_ID"],
            "Rule_Variant": history["Rule_Variant"]
        })
        df_history = append_history(df_history, "My_code/results/experiment_history.csv")

    print("\n=== Итоговые результаты экспериментов ===")
    print(load_summary("My_code/results/experiment_summary.csv"))

    df_all_hist = load_history("My_code/results/experiment_history.csv")
    unique_datasets = df_all_hist["Dataset_ID"].unique()

    # df = load_summary("experiment_history.csv")
    # df = df.iloc[:-60]
    # df.to_csv("experiment_history.csv", index=False)
    #
    #
    # df = load_summary("experiment_summary.csv")
    # df = df.iloc[:-2]
    # df.to_csv("experiment_summary.csv", index=False)

    # Получаем список уникальных датасетов

    # Для каждого датасета строим отдельную фигуру с двумя подграфиками и сохраняем её в файл
    for ds in unique_datasets:
        ds_data = df_all_hist[df_all_hist["Dataset_ID"] == ds]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # График точности (Accuracy)
        for rule_var, grp in ds_data.groupby("Rule_Variant"):
            label = f"RV {rule_var}"
            axes[0].plot(grp["Epoch"], grp["Test_Accuracy"], marker='o', label=label)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Test Accuracy")
        axes[0].set_title(f"Dataset {ds}: Test Accuracy vs Epoch")
        axes[0].legend()

        # График потерь (Loss)
        for rule_var, grp in ds_data.groupby("Rule_Variant"):
            label = f"RV {rule_var}"
            axes[1].plot(grp["Epoch"], grp["Test_Loss"], marker='o', label=label)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Test Loss")
        axes[1].set_title(f"Dataset {ds}: Test Loss vs Epoch")
        axes[1].legend()

        plt.tight_layout()
        # Сохраняем изображение в файл, например, с именем dataset_{ds}_metrics.png
        filename = f"My_code/results/dataset_{ds}_loss.png"
        plt.savefig(filename)
        print(f"Сохранён график для Dataset {ds} в файл {filename}")
        plt.close(fig)
