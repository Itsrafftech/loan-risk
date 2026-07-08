"""
Loads the trained model bundle and turns raw applicant inputs into a risk
prediction plus a short, human-readable explanation of the top contributing
factors.
"""

import os

import joblib
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "loan_model.joblib")

# Risk probability -> label thresholds.
LOW_RISK_MAX = 0.15
MEDIUM_RISK_MAX = 0.40

# Whether a higher value of a feature pushes risk up (+1) or down (-1).
RISK_DIRECTION = {
    "age": -1,
    "MonthlyIncome": -1,
    "NumberOfDependents": 1,
    "DebtRatio": 1,
    "RevolvingUtilizationOfUnsecuredLines": 1,
    "NumberOfOpenCreditLinesAndLoans": 1,
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 1,
    "NumberOfTimes90DaysLate": 1,
    "TotalPastDueCount": 1,
    "IncomePerDependent": -1,
}

FACTOR_LABELS = {
    "age": "Usia",
    "MonthlyIncome": "Penghasilan Bulanan",
    "NumberOfDependents": "Jumlah Tanggungan",
    "DebtRatio": "Rasio Utang",
    "RevolvingUtilizationOfUnsecuredLines": "Tingkat Penggunaan Kredit",
    "NumberOfOpenCreditLinesAndLoans": "Jumlah Pinjaman Aktif",
    "NumberOfTime30-59DaysPastDueNotWorse": "Keterlambatan 30 Hari",
    "NumberOfTime60-89DaysPastDueNotWorse": "Keterlambatan 60 Hari",
    "NumberOfTimes90DaysLate": "Keterlambatan 90 Hari",
    "TotalPastDueCount": "Total Insiden Keterlambatan",
    "IncomePerDependent": "Penghasilan per Tanggungan",
}


def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model belum ditemukan di {model_path}. Jalankan `python model/train.py` terlebih dahulu."
        )
    return joblib.load(model_path)


def _engineer_features(raw_input: dict) -> dict:
    features = dict(raw_input)
    features["TotalPastDueCount"] = (
        raw_input["NumberOfTime30-59DaysPastDueNotWorse"]
        + raw_input["NumberOfTime60-89DaysPastDueNotWorse"]
        + raw_input["NumberOfTimes90DaysLate"]
    )
    features["IncomePerDependent"] = raw_input["MonthlyIncome"] / (
        raw_input["NumberOfDependents"] + 1
    )
    return features


def risk_level(probability: float) -> str:
    if probability < LOW_RISK_MAX:
        return "Rendah"
    if probability < MEDIUM_RISK_MAX:
        return "Sedang"
    return "Tinggi"


def top_risk_factors(features: dict, bundle: dict, top_n: int = 3):
    """Ranks features by how much they likely push this applicant's risk up,
    combining global model importance with how far the applicant's value is
    from the training population's typical value."""
    importance = bundle["feature_importance"]
    stats = bundle["feature_stats"]

    scored = []
    for feature, value in features.items():
        stat = stats.get(feature)
        if stat is None:
            continue
        z = (value - stat["median"]) / stat["std"]
        direction = RISK_DIRECTION.get(feature, 0)
        contribution = importance.get(feature, 0) * direction * z
        if contribution > 0:
            scored.append((feature, contribution, value))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [
        {"feature": feature, "label": FACTOR_LABELS.get(feature, feature), "value": value}
        for feature, _, value in scored[:top_n]
    ]


def predict_risk(raw_input: dict, bundle: dict = None):
    """raw_input must contain the 9 base feature keys (see model/train.py::BASE_FEATURES)."""
    if bundle is None:
        bundle = load_model()

    features = _engineer_features(raw_input)
    row = pd.DataFrame([features])[bundle["feature_names"]]

    pipeline = bundle["pipeline"]
    probability = float(pipeline.predict_proba(row)[0, 1])
    level = risk_level(probability)
    factors = top_risk_factors(features, bundle, top_n=3)

    return {
        "probability": probability,
        "risk_level": level,
        "top_factors": factors,
        "model_name": bundle["model_name"],
    }
