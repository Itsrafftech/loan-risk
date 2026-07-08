"""
Trains the loan default risk model.

Loads real "Give Me Some Credit" data from data/cs-training.csv if present,
otherwise generates a synthetic dataset with comparable features. Cleans the
data, engineers a couple of extra features, trains Logistic Regression,
Random Forest, and XGBoost, evaluates each, and saves the best-performing
pipeline (by ROC-AUC) plus metrics/metadata to model/loan_model.joblib.
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODEL_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_KAGGLE_PATH = os.path.join(DATA_DIR, "cs-training.csv")
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "sample_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "loan_model.joblib")

TARGET = "SeriousDlqin2yrs"

BASE_FEATURES = [
    "age",
    "MonthlyIncome",
    "NumberOfDependents",
    "DebtRatio",
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

ENGINEERED_FEATURES = ["TotalPastDueCount", "IncomePerDependent"]
FEATURE_NAMES = BASE_FEATURES + ENGINEERED_FEATURES


def generate_synthetic_data(n=12000, random_state=42):
    """Generates a synthetic borrower dataset with realistic, correlated risk factors."""
    rng = np.random.default_rng(random_state)

    age = rng.normal(45, 13, n).clip(18, 90).round().astype(int)
    # Indonesian monthly income in Rupiah (median ~Rp 5,000,000), rounded to the
    # nearest Rp 10,000 like real salary figures tend to be.
    monthly_income = (
        rng.lognormal(mean=np.log(5_000_000), sigma=0.6, size=n)
        .clip(500_000, 100_000_000)
        .round(-4)
    )
    dependents = rng.poisson(0.9, n).clip(0, 10)
    revolving_util = (rng.beta(2, 5, n) * 1.3).round(4)
    debt_ratio = rng.gamma(2, 0.25, n).clip(0, 3).round(4)
    open_credit_lines = rng.poisson(8, n).clip(0, 40)
    late_30_59 = rng.poisson(0.25, n).clip(0, 20)
    late_60_89 = rng.poisson(0.12, n).clip(0, 20)
    late_90 = rng.poisson(0.10, n).clip(0, 20)

    # Nonlinear bumps so tree-based models have signal that a purely linear
    # model can't fully capture (mirrors real-world threshold effects).
    high_util_high_debt = ((revolving_util > 0.9) & (debt_ratio > 0.8)).astype(float)
    young_and_thin_credit = ((age < 26) & (open_credit_lines < 3)).astype(float)
    repeat_offender = (late_90 >= 2).astype(float)

    logit = (
        -3.7
        + 2.8 * revolving_util
        + 1.5 * debt_ratio.clip(0, 2)
        + 0.85 * late_30_59
        + 1.25 * late_60_89
        + 1.55 * late_90
        - 0.38 * np.log1p(monthly_income / 1_000_000)
        - 0.018 * age
        + 0.08 * dependents
        + 0.015 * open_credit_lines
        + 1.1 * high_util_high_debt
        + 0.9 * young_and_thin_credit
        + 1.0 * repeat_offender
    )
    prob_default = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, prob_default)

    df = pd.DataFrame(
        {
            "age": age,
            "MonthlyIncome": monthly_income,
            "NumberOfDependents": dependents,
            "DebtRatio": debt_ratio,
            "RevolvingUtilizationOfUnsecuredLines": revolving_util,
            "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
            "NumberOfTimes90DaysLate": late_90,
            TARGET: target,
        }
    )

    # Sprinkle in a few missing values, like the real Kaggle dataset has.
    missing_income_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    df.loc[missing_income_idx, "MonthlyIncome"] = np.nan
    missing_dep_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[missing_dep_idx, "NumberOfDependents"] = np.nan

    return df


def load_raw_data():
    if os.path.exists(RAW_KAGGLE_PATH):
        print(f"Loading real Kaggle dataset from {RAW_KAGGLE_PATH}")
        df = pd.read_csv(RAW_KAGGLE_PATH, index_col=0)
        return df
    print("Kaggle dataset not found, generating synthetic dataset instead.")
    return generate_synthetic_data()


def clean_data(df):
    df = df.copy()

    # Missing values: median imputation.
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    # Outliers: cap known problem columns at sane bounds (based on the well-known
    # quirks of this dataset, e.g. past-due counts of 96/98 are data-entry codes).
    df["DebtRatio"] = df["DebtRatio"].clip(upper=df["DebtRatio"].quantile(0.99))
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=2.0)
    for col in [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]:
        df[col] = df[col].clip(upper=20)
    df["NumberOfOpenCreditLinesAndLoans"] = df["NumberOfOpenCreditLinesAndLoans"].clip(
        upper=40
    )
    df["age"] = df["age"].clip(lower=18, upper=100)
    df["MonthlyIncome"] = df["MonthlyIncome"].clip(upper=df["MonthlyIncome"].quantile(0.995))

    return df


def engineer_features(df):
    df = df.copy()
    df["TotalPastDueCount"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    return df


def build_pipelines():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.08,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def get_feature_importance(pipeline, feature_names):
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        raw = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        raw = np.abs(clf.coef_[0])
    else:
        raw = np.zeros(len(feature_names))
    raw = raw / raw.sum() if raw.sum() > 0 else raw
    return dict(sorted(zip(feature_names, raw.tolist()), key=lambda kv: kv[1], reverse=True))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    raw_df = load_raw_data()
    if TARGET not in raw_df.columns:
        raise ValueError(f"Expected target column '{TARGET}' not found in dataset.")

    clean_df = clean_data(raw_df)
    full_df = engineer_features(clean_df)
    full_df = full_df[FEATURE_NAMES + [TARGET]]

    # Persist the working dataset for reference / reproducibility.
    full_df.to_csv(SAMPLE_DATA_PATH, index=False)
    print(f"Saved dataset ({len(full_df)} rows) to {SAMPLE_DATA_PATH}")
    print(f"Default rate: {full_df[TARGET].mean():.2%}")

    X = full_df[FEATURE_NAMES]
    y = full_df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipelines = build_pipelines()
    metrics = {}
    roc_curves = {}
    fitted = {}

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        fitted[name] = pipeline

        print(
            f"  accuracy={metrics[name]['accuracy']:.4f}  "
            f"roc_auc={metrics[name]['roc_auc']:.4f}  "
            f"precision={metrics[name]['precision']:.4f}  "
            f"recall={metrics[name]['recall']:.4f}  "
            f"f1={metrics[name]['f1']:.4f}"
        )

    best_name = max(metrics, key=lambda name: metrics[name]["roc_auc"])
    best_pipeline = fitted[best_name]
    print(f"\nBest model: {best_name} (ROC-AUC={metrics[best_name]['roc_auc']:.4f})")

    feature_stats = {
        col: {"median": float(X_train[col].median()), "std": float(X_train[col].std()) or 1.0}
        for col in FEATURE_NAMES
    }

    bundle = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "feature_names": FEATURE_NAMES,
        "metrics": metrics,
        "roc_curves": roc_curves,
        "feature_importance": get_feature_importance(best_pipeline, FEATURE_NAMES),
        "feature_stats": feature_stats,
        "test_size": len(X_test),
        "default_rate": float(y.mean()),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved best model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
