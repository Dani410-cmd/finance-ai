import os
import re
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix

# ==== ЗАГРУЗКА МОДЕЛЕЙ ВЕРНО ПО ПУТЯМ ====

# Папка, где лежит этот файл (py_project/ml/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Корень проекта (py_project/)
ROOT_DIR = os.path.dirname(BASE_DIR)

# Правильные пути к моделям
MODEL_PATH = os.path.join(ROOT_DIR, "backend_models", "classifier_v3.pkl")
VECTORIZER_PATH = os.path.join(ROOT_DIR, "backend_models", "vectorizer_v3.pkl")

# Загружаем модели
classifier = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Человеческие названия категорий
HUMAN_LABELS_RU = {
    "food": "Еда",
    "transport": "Транспорт",
    "stores": "Магазины",
    "monthly": "Ежемесячные платежи",
    "income": "Доход",
    "other": "Прочее",
}


# ----------------------------------------
# 1. RULE-BASED правила
# ----------------------------------------

def apply_rules(ref: str, withdraw: float, deposit: float):
    """
    Простые правила категоризации.
    Если попали под правило — сразу возвращаем категорию.
    """
    text = ref.lower()

    # 1) Доходы
    if any(w in text for w in ["salary", "paycheck", "income", "зарплата"]):
        return "income"

    # 2) Ежемесячные расходы (подписки, интернет, связь, аренда)
    if any(w in text for w in [
        "netflix", "spotify", "youtube premium", "apple music",
        "yandex plus", "subscription", "mobile internet", "phone plan",
        "hbo", "amazon prime", "rent", "аренда", "квартира", "жкх", "internet"
    ]):
        return "monthly"

    # 3) Транспорт
    if any(w in text for w in [
        "uber", "taxi", "bolt", "yandex go", "metro", "bus", "train", "fuel", "gas"
    ]):
        return "transport"

    # 4) Магазины
    if any(w in text for w in [
        "ozon", "wildberries", "wb", "amazon", "aliexpress",
        "zara", "h&m", "mall", "shopping", "store", "marketplace"
    ]):
        return "stores"

    # 5) Еда
    if any(w in text for w in [
        "mcdonald", "kfc", "burger king", "starbucks",
        "coffee", "cafe", "restaurant", "pizza", "sushi"
    ]):
        return "food"

    # 6) Большие поступления без описания → доход
    amount = deposit - withdraw
    if amount > 5000 and not text.strip():
        return "income"

    # 7) Большие списания без описания → аренда/ежемесячные
    if amount < -5000 and not text.strip():
        return "monthly"

    return None


# ----------------------------------------
# 2. ML-функции
# ----------------------------------------

def build_features_for_one(ref: str, withdraw: float, deposit: float):
    amount = deposit - withdraw
    abs_amount = abs(amount)
    is_income = 1 if amount > 0 else 0

    # текстовые фичи
    X_text = vectorizer.transform([ref])

    # числовые фичи
    num_features = np.array([[amount, abs_amount, is_income]])
    X_num_sparse = csr_matrix(num_features)

    return hstack([X_text, X_num_sparse])


def ml_predict(ref: str, withdraw: float, deposit: float):
    X = build_features_for_one(ref, withdraw, deposit)

    proba = classifier.predict_proba(X)[0]
    classes = classifier.classes_

    pairs = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)
    best_cat, best_p = pairs[0]

    # уверенность
    if best_p >= 0.7:
        level = "high"
    elif best_p >= 0.4:
        level = "medium"
    else:
        level = "low"

    human_label = HUMAN_LABELS_RU.get(best_cat, best_cat)

    return {
        "category_internal": best_cat,
        "category_human": human_label,
        "confidence": float(best_p),
        "confidence_level": level,
        "top3": [(cat, float(p)) for cat, p in pairs[:3]],
    }


# ----------------------------------------
# 3. Гибридный предикт (rules + ML)
# ----------------------------------------

def smart_predict_transaction(ref: str, withdraw: float, deposit: float):
    # 1. Правила
    rule_cat = apply_rules(ref, withdraw, deposit)

    if rule_cat:
        human_label = HUMAN_LABELS_RU.get(rule_cat, rule_cat)
        return {
            "source": "rules",
            "category_internal": rule_cat,
            "category_human": human_label,
            "confidence": 1.0,
            "confidence_level": "high",
            "top3": [(rule_cat, 1.0)],
            "needs_user_review": False,
        }

    # 2. ML
    ml_res = ml_predict(ref, withdraw, deposit)
    ml_res.update({
        "source": "ml",
        "needs_user_review": ml_res["confidence_level"] in ("medium", "low"),
    })

    return ml_res


# ----------------------------------------
# DEMO
# ----------------------------------------

if __name__ == "__main__":
    print("Демо гибридных предсказаний V3 (rules + ML)\n")

    examples = [
        ("Salary from company", 0, 1500),
        ("Netflix subscription", 9.99, 0),
        ("Uber trip", 11.3, 0),
        ("Ozon order electronics", 120, 0),
        ("Starbucks Coffee", 6.5, 0),
        ("Unknown payment XJ293", 47.2, 0),
    ]

    for ref, w, d in examples:
        res = smart_predict_transaction(ref, w, d)
        print("Операция:", ref, "| withdraw=", w, "deposit=", d)
        print(" → Источник:", res["source"])
        print(" → Внутренняя категория:", res["category_internal"])
        print(" → Для пользователя:", res["category_human"])
        print(" → Уверенность:", f"{res['confidence']:.2f}", f"({res['confidence_level']})")
        print(" → Нужно ли подтверждение пользователя:", res["needs_user_review"])
        print(" → Top-3:", res["top3"])
        print("-" * 60)