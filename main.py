import os
from datetime import date
from typing import List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.smart_predict_v3 import smart_predict_transaction
from ml.insights_engine import generate_insights_and_recommendations

# === FastAPI-приложение ===

app = FastAPI(title="Personal Finance AI Backend")

# === CORS, чтобы фронтенд мог ходить на API ===

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "https://z8j7l9.csb.app",
        "https://*.csb.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# === Модель для прогноза баланса ===

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_MODEL_PATH = os.path.join(BASE_DIR, "backend_models", "forecast_model.pkl")

forecast_model = None
try:
    forecast_model = joblib.load(FORECAST_MODEL_PATH)
except Exception as e:
    # Не падаем при импорте, просто логируем — /forecast вернет 500
    print(f"[WARN] Не удалось загрузить модель прогноза: {e}")


# === Pydantic-модели ===

class Transaction(BaseModel):
    date: date
    ref: str
    withdraw: float = 0.0
    deposit: float = 0.0
    category: Optional[str] = None


class ClassifyRequest(BaseModel):
    ref: str
    withdraw: float = 0.0
    deposit: float = 0.0


class ClassifyResponse(BaseModel):
    source: str
    category_internal: str
    category_human: str
    confidence: float
    confidence_level: str
    needs_user_review: bool


class InsightsRequest(BaseModel):
    transactions: List[Transaction]
    monthly_budget: float
    current_balance: float
    avg_daily_expense: float


class ForecastRequest(BaseModel):
    days: int = Field(7, ge=1, le=60)


class ForecastPoint(BaseModel):
    day: int
    balance: float


class ForecastResponse(BaseModel):
    forecast: List[ForecastPoint]


# === Служебные эндпоинты ===

@app.get("/")
def root():
    return {"status": "ok", "service": "Personal Finance AI Backend"}


@app.get("/health")
def health():
    return {"status": "ok"}


# === ML-эндпоинты ===

@app.post("/classify", response_model=ClassifyResponse)
def classify_transaction(req: ClassifyRequest):
    """
    Классификация одной транзакции:
    вход: ref + withdraw/deposit
    выход: категория + уверенность
    """
    res = smart_predict_transaction(req.ref, req.withdraw, req.deposit)
    return ClassifyResponse(
        source=res["source"],
        category_internal=res["category_internal"],
        category_human=res["category_human"],
        confidence=res["confidence"],
        confidence_level=res["confidence_level"],
        needs_user_review=res["needs_user_review"],
    )


@app.post("/insights")
def get_insights(req: InsightsRequest):
    """
    Инсайты по набору транзакций.
    ВАЖНО: структура ответа соответствует ml/insights_engine.py
    """
    tx = [t.dict() for t in req.transactions]
    result = generate_insights_and_recommendations(
        transactions=tx,
        monthly_budget=req.monthly_budget,
        current_balance=req.current_balance,
        avg_daily_expense=req.avg_daily_expense,
    )
    return result


@app.post("/forecast", response_model=ForecastResponse)
def get_forecast(req: ForecastRequest):
    """
    Прогноз баланса на N дней вперед.
    Использует ARIMA-модель из backend_models/forecast_model.pkl.
    """
    if forecast_model is None:
        raise HTTPException(
            status_code=500,
            detail="Модель прогноза не загружена на сервере",
        )

    steps = req.days
    # ARIMA-модель возвращает последовательность предсказаний
    preds = forecast_model.forecast(steps=steps)

    points = [
        ForecastPoint(day=i + 1, balance=float(preds[i]))
        for i in range(steps)
    ]
    return ForecastResponse(forecast=points)
