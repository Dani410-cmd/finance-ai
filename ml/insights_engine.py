# ml/insights_engine.py

from typing import List, Dict

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
        category = tx.get("category") or "other"

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

    if monthly_budget > 0:
        budget_used_percent = round((total_expense / monthly_budget) * 100, 2)
    else:
        budget_used_percent = 0.0
    
    if budget_left < 0:
        budget_status = "danger"  # превышение бюджета
    elif budget_used_percent >= 90:
        budget_status = "warning"  # бюджет почти исчерпан
    elif budget_used_percent >= 70:
        budget_status = "watch"  # требуется внимательность
    else:
        budget_status = "ok"  # всё хорошо

    # ----------------------------------------
    # 3. ФИНАНСОВАЯ ПОДУШКА
    # ----------------------------------------
    recommended_cushion = avg_daily_expense * 30  # рекомендуем минимум 1 месяц

    if recommended_cushion > 0:
        safety_pillow_percent = round((current_balance / recommended_cushion) * 100, 2)
    else:
        safety_pillow_percent = 100.0  # если расходов нет — подушка покрыта

    if safety_pillow_percent >= 100:
        cushion_status = "good"
    elif safety_pillow_percent >= 60:
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
    elif budget_status in ("warning", "watch"):
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
    else:
        recommendations.append(
            "Продолжайте придерживаться текущего темпа накоплений — подушка сформирована."
        )
    # Топовые траты
    top_categories = []
    if category_totals and total_expense > 0:
        sorted_cats = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
        for category, amount in sorted_cats[:5]:
            share = round((amount / total_expense) * 100, 2)
            top_categories.append(
                {"category": category, "amount": round(amount, 2), "share": share}
            )
        recommendations.append(
            f"Больше всего расходов в этом месяце — категория '{top_categories[0]['category']}'."
        )
    else:
        top_categories = []

    # ----------------------------------------
    # 5. ИТОГОВЫЙ JSON ДЛЯ BACKEND → FRONTEND
    # ----------------------------------------
    return {
        "monthly_expense": round(total_expense, 2),
        "monthly_budget": round(monthly_budget, 2),
        "budget_left": round(budget_left, 2),
        "budget_status": budget_status,
        "budget_used_percent": budget_used_percent,

        "current_balance": round(current_balance, 2),
        "avg_daily_expense": round(avg_daily_expense, 2),
        "recommended_cushion": round(recommended_cushion, 2),
        "safety_pillow_percent": safety_pillow_percent,
        "safety_pillow_status": cushion_status,

        "categories": {k: round(v, 2) for k, v in category_totals.items()},
        "top_categories": top_categories,
        "recommendations": recommendations,
    }
