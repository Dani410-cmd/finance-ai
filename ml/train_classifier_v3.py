import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.sparse import hstack, csr_matrix
import joblib

# Пути к файлам
DATA_PATH = os.path.join("..", "data", "ci_data_ml.csv")  # расширенный датасет
MODEL_PATH = os.path.join("..", "backend_models", "classifier_v3.pkl")
VECTORIZER_PATH = os.path.join("..", "backend_models", "vectorizer_v3.pkl")


def map_to_final_category(raw_cat: str) -> str:
    """
    Маппинг исходных категорий из датасета (Food, Misc, Rent, Shopping, Salary, Transport, Subscriptions)
    в финальные 6 категорий под приложение.

    Финальные внутренние категории:
    - food       (Еда)
    - transport  (Транспорт)
    - stores     (Магазины)
    - monthly    (Ежемесячные платежи: аренда, подписки и т.п.)
    - income     (Доход)
    - other      (Прочее, в т.ч. Misc, переводы и т.п.)
    """
    if raw_cat == "Food":
        return "food"
    if raw_cat == "Transport":
        return "transport"
    if raw_cat in ("Rent", "Subscriptions"):
        return "monthly"
    if raw_cat == "Salary":
        return "income"
    if raw_cat == "Shopping":
        return "stores"
    # Всё остальное (в т.ч. Misc) -> прочее
    return "other"


def load_data():
    print(f"Читаем датасет: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Удаляем полностью пустые строки
    df = df.dropna(how="all")

    # Переименуем колонки под удобные имена
    df = df.rename(columns={
        "Date": "date",
        "Category": "category",
        "RefNo": "ref",
        "Withdrawal": "withdraw",
        "Deposit": "deposit",
        "Balance": "balance"
    })

    # В исходном датасете была колонка Date.1 — дубликат, удаляем если есть
    if "Date.1" in df.columns:
        df = df.drop(columns=["Date.1"])

    # Приводим категории к финальному набору
    df["category"] = df["category"].apply(map_to_final_category)

    # Парсим дату
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # Заполняем пропуски
    df["ref"] = df["ref"].fillna("").astype(str)
    df["withdraw"] = df["withdraw"].fillna(0)
    df["deposit"] = df["deposit"].fillna(0)

    # Считаем amount: доходы +, расходы -
    df["amount"] = df["deposit"] - df["withdraw"]
    df["abs_amount"] = df["amount"].abs()
    df["is_income"] = (df["amount"] > 0).astype(int)

    before = len(df)
    df = df.dropna(subset=["category"])
    after = len(df)

    print(f"Удалено строк без категории: {before - after}")
    print("Колонки после подготовки:", list(df.columns))
    print("Распределение финальных категорий:")
    print(df["category"].value_counts())

    return df


def build_features(df: pd.DataFrame):
    # Текстовая часть — описание операции (ref)
    text_series = df["ref"].astype(str)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )
    X_text = vectorizer.fit_transform(text_series)

    # Числовые признаки
    num_cols = ["amount", "abs_amount", "is_income"]
    X_num = df[num_cols].values
    X_num_sparse = csr_matrix(X_num)

    # Склеиваем текст + числа
    X = hstack([X_text, X_num_sparse])
    return X, vectorizer


def train_classifier_v3():
    df = load_data()

    y = df["category"]
    X, vectorizer = build_features(df)

    # Разделение на train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Логистическая регрессия как основной классификатор
    model = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        multi_class="auto"
    )

    print("Обучаем классификатор V3 (финальные категории)...")
    model.fit(X_train, y_train)

    # Оценка качества
    y_pred = model.predict(X_test)
    print("=== CLASSIFICATION REPORT V3 ===")
    print(classification_report(y_test, y_pred, digits=3))

    # Сохраняем артефакты для backend
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("Модель V3 сохранена в:", MODEL_PATH)
    print("Векторизатор V3 сохранён в:", VECTORIZER_PATH)


if __name__ == "__main__":
    train_classifier_v3()