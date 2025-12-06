import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join("..", "data", "ci_data.csv")


def load_data():
    """Загружаем и подготавливаем CSV."""
    df = pd.read_csv(DATA_PATH)

    # Переводим названия колонок в нижний регистр
    df.columns = [c.lower() for c in df.columns]

    # Считаем amount = income - expense
    df["withdraw"] = df.get("withdrawal", df.get("withdraw", 0)).fillna(0)
    df["deposit"] = df.get("deposit", 0).fillna(0)
    df["amount"] = df["deposit"] - df["withdraw"]

    # Превращаем даты
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    return df.dropna(subset=["date"])


def compute_summary(df: pd.DataFrame):
    """Создаём финансовую сводку для отчёта/демо."""
    # Текущий баланс — последняя строка
    current_balance = df["balance"].iloc[-1] if "balance" in df.columns else df["amount"].cumsum().iloc[-1]

    # Средний дневной расход
    expenses = df[df["amount"] < 0].copy()
    expenses["abs_amount"] = expenses["amount"].abs()
    avg_daily_expense = expenses.groupby(df["date"].dt.date)["abs_amount"].sum().mean()

    if np.isnan(avg_daily_expense):
        avg_daily_expense = 0  # на случай, если расходов нет

    # Рекомендуемая подушка = X дней * средний расход
    reserve_days = 30
    recommended_cushion = avg_daily_expense * reserve_days

    # Статус безопасности
    if current_balance >= recommended_cushion:
        risk_status = "ОК"
    elif current_balance >= recommended_cushion * 0.6:
        risk_status = "ПОЙДЁТ"
    else:
        risk_status = "ПЛОХО"

    # Рекомендации
    recommendations = []

    if current_balance < recommended_cushion:
        recommendations.append(
            "Ваш баланс ниже рекомендуемой финансовой подушки — постарайтесь откладывать хотя бы 10–15% дохода."
        )

    if avg_daily_expense > 0 and current_balance < avg_daily_expense * 7:
        recommendations.append(
            "Ваших средств хватит менее чем на неделю расходов — избегайте необязательных трат."
        )

    if not recommendations:
        recommendations.append("Всё выглядит стабильно — продолжайте следить за финансами.")

    return {
        "current_balance": round(current_balance, 2),
        "avg_daily_expense": round(avg_daily_expense, 2),
        "recommended_cushion": round(recommended_cushion, 2),
        "cushion_percent": round((current_balance / recommended_cushion) * 100, 1) if recommended_cushion > 0 else 100,
        "status": risk_status,
        "recommendations": recommendations,
    }


def print_summary(summary: dict):
    """Красиво печатаем результат."""
    print("=== Финансовая сводка ===")
    print(f"Текущий баланс: {summary['current_balance']}")
    print(f"Средний дневной расход: {summary['avg_daily_expense']}")
    print(f"Рекомендуемая финансовая подушка: {summary['recommended_cushion']}")
    print(f"Процент заполненности подушки: {summary['cushion_percent']}%")
    print(f"Статус: {summary['status']}")
    print("\nРекомендации:")
    for i, rec in enumerate(summary["recommendations"], 1):
        print(f"{i}. {rec}")


if __name__ == "__main__":
    df = load_data()
    summary = compute_summary(df)
    print_summary(summary)
