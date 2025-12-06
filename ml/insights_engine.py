# ml/insights_engine.py

from typing import List, Dict
from datetime import datetime

def generate_insights_and_recommendations(
    transactions: List[Dict],
    monthly_budget: float,
    current_balance: float,
    avg_daily_expense: float
):
    """
    Финансовые инсайты:
      - траты за месяц
      - сравнение с бюджетом
      - заполненность финансовой подушки
      - топовые категории трат
      - автоматические рекомендации
    """

    # ----------------------------------------
    # 1. СЧИТАЕМ РАСХОДЫ ЗА МЕСЯЦ
    # ----------------------------------------
    total_expense = 0.0
    category_totals: Dict[str, float] = {}

    for tx in transactions:
        withdraw = tx.get("withdraw", 0) or 0
        deposit = tx.get("deposit", 0) or 0
        category = tx.get("category", "other")

        amount = deposit - withdraw

        # Учитываем только РАСХОДЫ
        if amount < 0:
            expense = -amount
            total_expense += expense
            category_totals[category] = category_totals.get(category, 0) + expense

    # ----------------------------------------
    # 2. БЮДЖЕТ И СТАТУС
    # ----------------------------------------
    budget_left = monthly_budget - total_expense

    if budget_left >= monthly_budget * 0.3:
        budget_status = "ok"        # всё хорошо
    elif budget_left >= 0:
        budget_status = "warning"   # скоро закончится
    else:
        budget_status = "danger"    # превышение бюджета

    # ----------------------------------------
    # 3. ФИНАНСОВАЯ ПОДУШКА
    # ----------------------------------------
    recommended_cushion = avg_daily_expense * 30  # рекомендуем минимум 1 месяц
    if recommended_cushion <= 0:
        recommended_cushion = 1  # защита от деления на ноль

    cushion_percent = min(100, round(current_balance / recommended_cushion * 100))

    if cushion_percent >= 70:
        cushion_status = "good"
    elif cushion_percent >= 40:
        cushion_status = "medium"
    else:
        cushion_status = "low"

    # ----------------------------------------
    # 4. РЕКОМЕНДАЦИИ ДЛЯ ПОЛЬЗОВАТЕЛЯ
    # ----------------------------------------
    recommendations = []

    # Бюджет
    if budget_status == "danger":
        recommendations.append(
            "Ваши расходы превышают бюджет — сократите необязательные траты."
        )
    elif budget_status == "warning":
        recommendations.append(
            "Вы приближаетесь к лимиту бюджета. Контролируйте расходы."
        )

    # Фин. подушка
    if cushion_status == "low":
        recommendations.append(
            "Финансовая подушка слишком мала — рекомендуется откладывать 10–15% дохода."
        )
    elif cushion_status == "medium":
        recommendations.append(
            "Финансовая подушка на среднем уровне — продолжайте откладывать."
        )

    # Топовые траты
    if category_totals:
        top_cat = max(category_totals.items(), key=lambda x: x[1])
        recommendations.append(
            f"Больше всего расходов в этом месяце — категория '{top_cat[0]}'."
        )

    # ----------------------------------------
    # 5. ИТОГОВЫЙ JSON ДЛЯ BACKEND → FRONTEND
    # ----------------------------------------
    return {
        "monthly_expense": round(total_expense, 2),
        "monthly_budget": round(monthly_budget, 2),
        "budget_left": round(budget_left, 2),
        "budget_status": budget_status,

        "recommended_cushion": round(recommended_cushion, 2),
        "cushion_percent": cushion_percent,
        "cushion_status": cushion_status,

        "categories": category_totals,
        "recommendations": recommendations,
    }
