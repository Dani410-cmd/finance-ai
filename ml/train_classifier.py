import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# Пути
DATA_PATH = os.path.join("..", "data", "ci_data.csv")
MODEL_PATH = os.path.join("..", "backend_models", "classifier.pkl")
VECTORIZER_PATH = os.path.join("..", "backend_models", "vectorizer.pkl")


def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Колонки в датасете:", list(df.columns))

    # Переименовываем английские колонки в наши стандартные
    df = df.rename(columns={
        'Date': 'date',
        'Category': 'category',
        'RefNo': 'ref',
        'Withdrawal': 'withdraw',
        'Deposit': 'deposit',
        'Balance': 'balance'
    })

    # Удаляем лишние дублирующие колонки (например Date.1)
    if 'Date.1' in df.columns:
        df = df.drop(columns=['Date.1'])

    # Заполняем пропуски
    df['ref'] = df['ref'].fillna('')
    df['withdraw'] = df['withdraw'].fillna(0)
    df['deposit'] = df['deposit'].fillna(0)

    # Признак amount — полезен
    df['amount'] = df['deposit'] - df['withdraw']

    print("Переименованные колонки:", list(df.columns))
    return df


def train_classifier():
    df = load_data()

    # Удаляем строки без категории
    before = len(df)
    df = df.dropna(subset=["category"])
    after = len(df)
    print(f"Удалено строк без категории: {before - after}")

    # Целевая переменная
    y = df["category"]

    # Признак: текст
    X_text = df["ref"].astype(str)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))

    # Сохраняем модели
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Модель сохранена в:", MODEL_PATH)
    print("Векторизатор сохранён в:", VECTORIZER_PATH)


if __name__ == "__main__":
    train_classifier()