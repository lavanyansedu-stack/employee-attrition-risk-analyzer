from pathlib import Path
import joblib
import pandas as pd

from .config import MODEL_PATH, RISK_LOW, RISK_MEDIUM


def load_model(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python -m src.train_model"
        )
    return joblib.load(path)


def risk_band(probability: float) -> str:
    if probability <= RISK_LOW:
        return "Low Risk"
    if probability <= RISK_MEDIUM:
        return "Medium Risk"
    return "High Risk"


def predict_employee(employee: dict, model=None):
    model = model or load_model()
    row = pd.DataFrame([employee])

    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(model.predict(row)[0])

    return {
        "risk_score": probability,
        "risk_percent": round(probability * 100, 1),
        "risk_level": risk_band(probability),
        "predicted_class": prediction,
    }
