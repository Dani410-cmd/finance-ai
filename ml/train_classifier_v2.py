import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.sparse import hstack, csr_matrix
import joblib

# Пути
DATA_PATH = os.path.join("..", "data", "ci_data_ml.csv")
MODEL_PATH = os.path.join("..", "backend_models", "classifier_v2.pkl")
VECTORIZER_PATH = os.path.join("..", "backend_models", "vectorizer_v2.pkl")


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Переименуем под себя под единый стиль
    df = df.rename(columns={
        "Date": "date",
        "Category": "category",
        "RefNo": "ref",
        "Withdrawal": "withdraw",
        "Deposit": "deposit",
        "Balance": "balance"
    })

    # Уберём дубль даты, если есть
    if "Date.1" in df.columns:
        df = df.drop(columns=["Date.1"])

    # Преобразуем дату (на будущее)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # Чистим пропуски
    df["ref"] = df["ref"].fillna("")
    df["withdraw"] = df["withdraw"].fillna(0)
    df["deposit"] = df["deposit"].fillna(0)

    # amount: пополнение +, расход -
    df["amount"] = df["deposit"] - df["withdraw"]
    df["abs_amount"] = df["amount"].abs()
    df["is_income"] = (df["amount"] > 0).astype(int)

    # выбрасываем строки без категории
    before = len(df)
    df = df.dropna(subset=["category"])
    after = len(df)
    print(f"Удалено строк без категории: {before - after}")

    print("Колонки после подготовки:", list(df.columns))

    return df


def build_features(df: pd.DataFrame):
    # Текстовая фича
    text_series = df["ref"].astype(str)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )
    X_text = vectorizer.fit_transform(text_series)

    # Числовые фичи
    num_cols = ["amount", "abs_amount", "is_income"]
    X_num = df[num_cols].values  # shape (n_samples, 3)

    # Превращаем числовые фичи в sparse и склеиваем
    X_num_sparse = csr_matrix(X_num)
    X = hstack([X_text, X_num_sparse])

    return X, vectorizer


def train_classifier_v2():
    df = load_data()

    # Целевая переменная
    y = df["category"]

    # Признаки
    X, vectorizer = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, n_jobs=-1)

    print("Обучаем улучшенный классификатор (LogReg + фичи)...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("=== CLASSIFICATION REPORT V2 ===")
    print(classification_report(y_test, y_pred))

    # Сохраняем
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Улучшенная модель сохранена в:", MODEL_PATH)
    print("Векторизатор V2 сохранён в:", VECTORIZER_PATH)


if __name__ == "__main__":
    train_classifier_v2()