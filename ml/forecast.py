from pathlib import Path
from typing import List, Dict

import joblib


FORECAST_MODEL_PATH = Path(__file__).resolve().parent.parent / "backend_models" / "forecast_model.pkl"


def load_forecast_model():
    if not FORECAST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Forecast model not found at {FORECAST_MODEL_PATH}")
    return joblib.load(FORECAST_MODEL_PATH)


def generate_forecast(days: int = 7) -> List[Dict[str, float]]:
    model = load_forecast_model()
    forecast = model.forecast(steps=days)
    return [{"day": i + 1, "balance": float(balance)} for i, balance in enumerate(forecast)]