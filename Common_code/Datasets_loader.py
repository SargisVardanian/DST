import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def load_dataset(csv_path: str | Path, *, normalize: bool = True):
    """
    Universal CSV loader  →  (X, y, feature_names)

    • автоматически определяет столбец-цель (label/target)
    • удаляет ID-подобные колонки
    • целочисленно кодирует категориальные признаки
    • (optionally) Z-нормирует **только** числовые колонки
    --------------------------------------------------------------------
    Returns
    -------
    X            : np.ndarray, shape (n_samples, n_features)
    y            : np.ndarray, shape (n_samples,)
    feature_names: list[str]
    """

    csv_path = Path(csv_path)

    # 1) читаем CSV, пытаясь sep=',' и ';'
    for sep in (",", ";"):
        try:
            df = pd.read_csv(csv_path, sep=sep, na_values=["?", "NA", "nan"],
                             engine="python")
            if df.shape[1] > 1:
                break
        except Exception:
            continue
    else:
        raise ValueError(f"Cannot read {csv_path} with ',' or ';'")

    # 2) удаляем строки с пропусками
    df = df.dropna(axis=0, how="any").reset_index(drop=True)

    # 3) убираем ID-подобные столбцы
    id_like = [
        c for c in df.columns
        if ("id" in c.lower() or "uid" in c.lower() or "identifier" in c.lower())
        or df[c].is_unique
    ]
    if id_like:
        df = df.drop(columns=id_like)

    # 4) находим столбец-метку
    preferred = {"label","labels","class","target","outcome","diagnosis","y","result"}
    label_col = next((c for c in df.columns if c.strip().lower() in preferred), None)

    if label_col is None:
        # единственный бинарный не-float столбец
        bins = [c for c in df.columns if df[c].nunique()==2 and df[c].dtype!=float]
        if len(bins)==1:
            label_col = bins[0]
        elif len(bins)>1:
            # самый сбалансированный
            label_col = min(
                bins,
                key=lambda c: abs(df[c].value_counts(normalize=True).iloc[0] - 0.5)
            )
    if label_col is None:
        # любой столбец с ≤10 уникальных
        small = [c for c in df.columns if 2<=df[c].nunique()<=10]
        if small:
            label_col = small[-1]
    if label_col is None:
        label_col = df.columns[-1]

    # 5) factorize метки → y
    y, classes_ = pd.factorize(df[label_col], sort=True)
    if len(classes_)<2:
        raise ValueError(f"Too few classes in '{label_col}'")

    # 6) готовим X: отделяем метку
    X_df = df.drop(columns=[label_col])

    # 6a) кодируем **целочисленно** все категориальные (object|category) колонки
    cat_cols = X_df.select_dtypes(include=['object','category']).columns
    for c in cat_cols:
        X_df[c] = pd.Categorical(X_df[c]).codes.astype(float)

    # 6b) нормируем числовые (включая получившиеся integer-коды)
    # if normalize:
    #     num_cols = X_df.select_dtypes(include=[np.number]).columns
    #     if len(num_cols)>0:
    #         scaler = StandardScaler()
    #         X_df[num_cols] = scaler.fit_transform(X_df[num_cols])

    X = X_df.to_numpy(dtype=float)
    feature_names = X_df.columns.tolist()

    # 7) логируем
    print(f"Detected label column: '{label_col}'  →  classes: {classes_.tolist()}")
    print(f"X shape = {X.shape},  y distribution = {np.bincount(y).tolist()}")

    return X, y, feature_names
