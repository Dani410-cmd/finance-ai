import os
from datetime import date
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationInfo, field_validator

from ml.forecast import generate_forecast
from ml.insights_engine import generate_insights_and_recommendations
from ml.smart_predict_v3 import smart_predict_transaction

app = FastAPI(title="Personal Finance AI Backend")

default_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://financeai.app",
]

frontend_origins = [origin for origin in os.getenv("FRONTEND_ORIGINS", "").split(",") if origin]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[*default_origins, *frontend_origins],
    allow_origin_regex=r"https?://localhost(:\d+)?|https://.*\.csb\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


class Transaction(BaseModel):
    date: date
    ref: str
    withdraw: float = 0.0
    deposit: float = 0.0
    category: Optional[str] = None

    @field_validator("withdraw", "deposit")
    @classmethod
    def validate_amount(cls, value: float, info: ValidationInfo):
        if value < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return value


class ClassifyRequest(BaseModel):
    ref: str
    withdraw: float = 0.0
    deposit: float = 0.0

    @field_validator("withdraw", "deposit")
    @classmethod
    def validate_amount(cls, value: float, info: ValidationInfo):
        if value < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return value


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

    @field_validator("monthly_budget", "avg_daily_expense", "current_balance")
    @classmethod
    def validate_non_negative(cls, value: float, info: ValidationInfo):
        if value < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return value



class TopCategory(BaseModel):
    category: str
    amount: float
    share: float


class InsightsResponse(BaseModel):
    monthly_expense: float
    monthly_budget: float
    budget_left: float
    budget_status: str
    budget_used_percent: float
    current_balance: float
    avg_daily_expense: float
    recommended_cushion: float
    safety_pillow_percent: float
    safety_pillow_status: str
    categories: Dict[str, float]
    top_categories: List[TopCategory]
    recommendations: List[str]


class ForecastItem(BaseModel):
    day: int
    balance: float


class ForecastRequest(BaseModel):
    days: int = 7

    @field_validator("days")
    @classmethod
    def validate_days(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("days must be positive")
        return value


@app.post("/classify", response_model=ClassifyResponse)
def classify_transaction(req: ClassifyRequest):
    res = smart_predict_transaction(req.ref, req.withdraw, req.deposit)
    return ClassifyResponse(
        source=res["source"],
        category_internal=res["category_internal"],
        category_human=res["category_human"],
        confidence=res["confidence"],
        confidence_level=res["confidence_level"],
        needs_user_review=res["needs_user_review"],
    )


@app.post("/insights", response_model=InsightsResponse)
def get_insights(req: InsightsRequest):
    tx = [t.model_dump() for t in req.transactions]
    result = generate_insights_and_recommendations(
        transactions=tx,
        monthly_budget=req.monthly_budget,
        current_balance=req.current_balance,
        avg_daily_expense=req.avg_daily_expense,
    )
    return InsightsResponse(**result)


@app.post("/forecast", response_model=List[ForecastItem])
def forecast_balance(req: ForecastRequest):
    try:
        forecast_data = generate_forecast(req.days)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="forecast model not found")

    return [ForecastItem(**item) for item in forecast_data]