from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_income_rule():
    payload = {"ref": "Salary from company", "withdraw": 0, "deposit": 50000}
    response = client.post("/classify", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["source"] == "rules"
    assert data["category_internal"] == "income"
    assert data["category_human"]
    assert isinstance(data["confidence"], (int, float))


def test_insights_smoke():
    payload = {
        "transactions": [
            {"date": "2024-12-01", "ref": "Groceries", "withdraw": 3000, "deposit": 0},
            {"date": "2024-12-02", "ref": "Salary", "withdraw": 0, "deposit": 90000},
        ],
        "monthly_budget": 60000,
        "current_balance": 150000,
        "avg_daily_expense": 2500,
    }

    response = client.post("/insights", json=payload)

    assert response.status_code == 200
    data = response.json()

    expected_keys = {
        "monthly_expense",
        "monthly_budget",
        "budget_left",
        "budget_status",
        "budget_used_percent",
        "current_balance",
        "avg_daily_expense",
        "recommended_cushion",
        "safety_pillow_percent",
        "safety_pillow_status",
        "categories",
        "top_categories",
        "recommendations",
    }

    assert expected_keys.issubset(data.keys())
    assert isinstance(data.get("top_categories"), list)
    assert isinstance(data.get("recommendations"), list)