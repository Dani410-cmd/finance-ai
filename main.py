from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from datetime import date

from ml.smart_predict_v3 import smart_predict_transaction
from ml.insights_engine import generate_insights_and_recommendations

app = FastAPI(title="Personal Finance AI Backend")

@app.get("/")
def root():
    return {"status": "ok", "service": "Personal Finance AI Backend"}

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

@app.post("/classify", response_model=ClassifyResponse)
def classify_transaction(req: ClassifyRequest):
    res = smart_predict_transaction(req.ref, req.withdraw, req.deposit)
    return ClassifyResponse(**{
        "source": res["source"],
        "category_internal": res["category_internal"],
        "category_human": res["category_human"],
        "confidence": res["confidence"],
        "confidence_level": res["confidence_level"],
        "needs_user_review": res["needs_user_review"],
    })

@app.post("/insights")
def get_insights(req: InsightsRequest):
    tx = [t.dict() for t in req.transactions]
    result = generate_insights_and_recommendations(
        transactions=tx,
        monthly_budget=req.monthly_budget,
        current_balance=req.current_balance,
        avg_daily_expense=req.avg_daily_expense,
    )
    return result
