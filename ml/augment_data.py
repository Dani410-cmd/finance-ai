import os
import numpy as np
import pandas as pd

DATA_PATH = os.path.join("..", "data", "ci_data.csv")
OUTPUT_PATH = os.path.join("..", "data", "ci_data_ml.csv")

# Какие категории будем усиливать и какими суммами
CATEGORY_CONFIG = {
    "Rent": {
        "refs": [
            "Apartment rent",
            "Monthly rent payment",
            "Landlord transfer",
            "House rental fee",
            "Rent for flat"
        ],
        "income": False,
        "amount_range": (500, 2000),
    },
    "Shopping": {
        "refs": [
            "Ozon order",
            "Wildberries purchase",
            "Amazon purchase",
            "AliExpress order",
            "Zara store",
            "H&M store",
            "Electronics store",
            "Clothes store",
            "Shoe store",
            "Household goods store",
            "Cosmetics shop",
            "Mall shopping",
            "Retail store",
            "Online marketplace order",
        ],
        "income": False,
        "amount_range": (20, 300),
    },
    "Transport": {
        "refs": [
            "Uber trip",
            "Taxi ride",
            "Metro card refill",
            "Bus ticket",
            "Train fare"
        ],
        "income": False,
        "amount_range": (3, 30),
    },
    "Salary": {
        "refs": [
            "Monthly salary",
            "Salary from company",
            "Paycheck",
            "Income transfer",
            "Employer payment"
        ],
        "income": True,
        "amount_range": (800, 2500),
    },
    "Subscriptions": {
        "refs": [
            "Netflix subscription",
            "Spotify subscription",
            "YouTube Premium",
            "Apple Music subscription",
            "Yandex Plus",
            "Mobile internet plan",
            "Phone plan monthly fee",
            "Streaming service payment",
            "HBO Max monthly",
            "Amazon Prime Video"
        ],
        "income": False,
        "amount_range": (5, 50),
    },
}


def generate_synthetic_rows(category: str, n_rows: int) -> list[dict]:
    cfg = CATEGORY_CONFIG[category]
    refs = cfg["refs"]
    income = cfg["income"]
    lo, hi = cfg["amount_range"]

    rows = []
    rng = np.random.default_rng(42)

    for _ in range(n_rows):
        day = int(rng.integers(1, 29))
        date_str = f"{day}/1/2023"

        amount = float(rng.uniform(lo, hi))
        amount = round(amount, 2)

        if income:
            withdraw = 0.0
            deposit = amount
        else:
            withdraw = amount
            deposit = 0.0

        ref = rng.choice(refs)

        row = {
            "Date": date_str,
            "Category": category,
            "RefNo": ref,
            "Date.1": date_str,
            "Withdrawal": withdraw,
            "Deposit": deposit,
            "Balance": 0.0,
        }
        rows.append(row)

    return rows


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(how="all")

    print("Размер исходного датасета:", len(df))
    print("Распределение по категориям до:")
    print(df["Category"].value_counts())

    TARGET_PER_CLASS = {
        "Rent": 40,
        "Shopping": 120,
        "Transport": 40,
        "Salary": 40,
        "Subscriptions": 40,
    }

    synthetic_rows = []

    for category, cfg in CATEGORY_CONFIG.items():
        target_count = TARGET_PER_CLASS.get(category, 40)
        current_count = (df["Category"] == category).sum()
        to_add = target_count - current_count

        if to_add > 0:
            print(f"Категория {category}: есть {current_count}, нужно добавить {to_add} синтетических примеров.")
            synthetic_rows.extend(generate_synthetic_rows(category, to_add))
        else:
            print(f"Категория {category}: уже достаточно примеров ({current_count}).")

    if synthetic_rows:
        df_syn = pd.DataFrame(synthetic_rows)
        df_all = pd.concat([df, df_syn], ignore_index=True)
    else:
        df_all = df

    print("Размер нового датасета:", len(df_all))
    print("Распределение по категориям после:")
    print(df_all["Category"].value_counts())

    df_all.to_csv(OUTPUT_PATH, index=False)
    print("Новый датасет сохранён в:", OUTPUT_PATH)


if __name__ == "__main__":
    main()