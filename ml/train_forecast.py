import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib

# Пути
DATA_PATH = os.path.join("..", "data", "ci_data.csv")
FORECAST_MODEL_PATH = os.path.join("..", "backend_models", "forecast_model.pkl")


def load_time_series():
    df = pd.read_csv(DATA_PATH)

    # Переименуем под себя
    df = df.rename(columns={
        "Date": "date",
        "Balance": "balance"
    })

    # Приводим дату к datetime и сортируем
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    # Если в один день несколько транзакций — берём последний баланс на день
    daily = df.groupby("date")["balance"].last().reset_index()

    daily.set_index("date", inplace=True)

    print("Пример временного ряда:")
    print(daily.head())

    return daily["balance"]


def train_forecast_model():
    ts = load_time_series()

    # Простая ARIMA(1,1,1)
    print("Обучаем модель ARIMA...")
    model = ARIMA(ts, order=(1, 1, 1))
    model_fit = model.fit()

    print(model_fit.summary())

    # Сохраняем модель
    joblib.dump(model_fit, FORECAST_MODEL_PATH)
    print("Модель прогноза сохранена в:", FORECAST_MODEL_PATH)

    # Прогноз на 7 дней вперёд
    forecast = model_fit.forecast(steps=7)
    print("Прогноз на 7 дней вперёд:")
    print(forecast)


if __name__ == "__main__":
    train_forecast_model()