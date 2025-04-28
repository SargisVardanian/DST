# =====================================================================
# Функция загрузки датасета по номеру
# 1: Brain Tumor – пытаемся загрузить с OpenML (если не удаётся, генерируем синтетически)
# 2: Breast Cancer – load_breast_cancer из sklearn
# 3: Gaussian – make_classification
# 4: Rectangle – равномерное распределение в [-1,1]^2, класс по знаку y
# 5: Uniform – равномерное распределение в [-5,5]^3, класс по знаку первого признака
# =====================================================================
from sklearn.datasets import load_breast_cancer, make_classification, fetch_openml
import pandas as pd


def load_dataset(dataset_id):
    if dataset_id == 1:
        # Brain Tumor – пытаемся загрузить локальный файл brain_tumor.csv с разделителем ';'
        csv_path = "/Users/sargisvardanyan/PycharmProjects/DST/df_breast-cancer-wisconsin.csv"
        df = pd.read_csv(csv_path, delimiter=',', on_bad_lines='skip')
        # df.drop("distance_norm", axis=1, inplace=True)
        # df.rename(columns={'labels_clustering': 'labels'}, inplace=True)
        # median_val = df['distance_norm'].median()
        # df['distance_norm'] = df['distance_norm'].fillna(median_val)
        # df.drop("distance_norm", axis=1, inplace=True)
        # df.to_csv(csv_path, index=False)

        print(df.head())
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        print("X: ", X.shape)
        print("y: ", y.shape)
        feature_names = df.columns[:-1].tolist()
        return X, y, feature_names
    elif dataset_id == 2:
        # Breast Cancer – используем датасет из sklearn
        df = load_breast_cancer()

        # print(df.head())
        X = df.data
        y = df.target
        feature_names = df.feature_names.tolist()
        return X, y, feature_names
    elif dataset_id == 3:
        # Gaussian – если CSV существует, загружаем его; иначе генерируем синтетически
        csv_path = "/Users/sargisvardanyan/PycharmProjects/DST/df_gaussian.csv"
        df = pd.read_csv(csv_path, delimiter=',', on_bad_lines='skip')

        print(df.head())
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        print("X: ", X.shape)
        print("y: ", y.shape)
        feature_names = df.columns[:-1].tolist()
        return X, y, feature_names
    elif dataset_id == 4:
        # Rectangle – если CSV существует, загружаем его; иначе генерируем синтетически
        csv_path = "/Users/sargisvardanyan/PycharmProjects/DST/df_rectangle.csv"
        df = pd.read_csv(csv_path, delimiter=',', on_bad_lines='skip')

        print(df.head())
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        print("X: ", X.shape)
        print("y: ", y.shape)
        feature_names = df.columns[:-1].tolist()
        return X, y, feature_names
    elif dataset_id == 5:
        # Uniform – если CSV существует, загружаем его; иначе генерируем синтетически
        csv_path = "/Users/sargisvardanyan/PycharmProjects/DST/df_uniform.csv"
        df = pd.read_csv(csv_path, delimiter=',', on_bad_lines='skip')

        print(df.head())
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        print("X: ", X.shape)
        print("y: ", y.shape)
        feature_names = df.columns[:-1].tolist()
        return X, y, feature_names
    elif dataset_id == 6:
        # Uniform – если CSV существует, загружаем его; иначе генерируем синтетически
        csv_path = "/Users/sargisvardanyan/PycharmProjects/DST/df_wine.csv"
        df = pd.read_csv(csv_path, delimiter=',', on_bad_lines='skip')

        print(df.head())
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        print("X: ", X.shape)
        print("y: ", y.shape)
        feature_names = df.columns[:-1].tolist()
        return X, y, feature_names
    else:
        raise ValueError("Неизвестный ID датасета")
