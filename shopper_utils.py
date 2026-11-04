"""Shared helpers for the Online Shoppers purchase-intent project.

``train_models.py`` and ``app.py`` both import this module, so the numbers that
appear in the README and the numbers shown in the Streamlit app are produced by
exactly the same code.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"
RAW_CSV = ROOT / "data" / "online_shoppers_intention.csv"

TARGET = "Revenue"
CLASS_NAMES = ["No purchase", "Purchase"]  # list position == class id (0 / 1)

# The 17 model inputs, in the column order of the UCI file.
FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

# How every feature is pre-processed inside each saved pipeline.
NUMERIC_COLS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Weekend",
]  # median-impute + standardise
CODE_COLS = ["OperatingSystems", "Browser", "Region", "TrafficType"]  # integer IDs -> one-hot
TEXT_COLS = ["Month", "VisitorType"]  # strings -> one-hot
assert sorted(NUMERIC_COLS + CODE_COLS + TEXT_COLS) == sorted(FEATURES)

# Display name -> file written by train_models.py
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# (column, kind, meaning) - used for the data dictionary shown in the app.
FEATURE_INFO = [
    ("Administrative", "count", "Administrative pages visited in the session"),
    ("Administrative_Duration", "time", "Total time spent on administrative pages"),
    ("Informational", "count", "Informational pages visited in the session"),
    ("Informational_Duration", "time", "Total time spent on informational pages"),
    ("ProductRelated", "count", "Product-related pages visited in the session"),
    ("ProductRelated_Duration", "time", "Total time spent on product-related pages"),
    ("BounceRates", "ratio", "Average bounce rate of the pages visited"),
    ("ExitRates", "ratio", "Average exit rate of the pages visited"),
    ("PageValues", "number", "Average page value of the pages visited"),
    ("SpecialDay", "0 - 1", "Closeness of the visit to a special day (e.g. Mother's Day)"),
    ("Month", "text", "Month of the visit (Feb ... Dec; 'June' is written in full)"),
    ("OperatingSystems", "ID", "Operating-system identifier"),
    ("Browser", "ID", "Browser identifier"),
    ("Region", "ID", "Visitor region identifier"),
    ("TrafficType", "ID", "Traffic-source identifier"),
    ("VisitorType", "text", "Returning_Visitor, New_Visitor or Other"),
    ("Weekend", "0 / 1", "1 if the visit happened on a weekend"),
    (TARGET, "0 / 1", "TARGET - 1 if the session ended in a purchase"),
]

_TRUE_TOKENS = {"true", "t", "yes", "y", "1", "1.0"}
_FALSE_TOKENS = {"false", "f", "no", "n", "0", "0.0"}


# --------------------------------------------------------------------------- data prep
def _as_flag(value) -> float:
    if not isinstance(value, str) and pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if value else 0.0
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return 1.0
    if token in _FALSE_TOKENS:
        return 0.0
    return np.nan


def to_flag(series: pd.Series) -> pd.Series:
    """TRUE/FALSE, yes/no and 1/0 style values -> 1.0 / 0.0 (NaN when unrecognised)."""
    return pd.Series([_as_flag(v) for v in series], index=series.index, dtype="float64")


def _clean_text(value):
    if not isinstance(value, str) and pd.isna(value):
        return np.nan
    text = str(value).strip()
    return text if text else np.nan


def tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip stray whitespace from column names (e.g. ' Revenue')."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def missing_columns(df: pd.DataFrame) -> list[str]:
    """Required columns (17 features + target) that are absent from ``df``."""
    return [c for c in [*FEATURES, TARGET] if c not in df.columns]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 17 model inputs in training order and with the training dtypes.

    Extra columns and column order in ``df`` are ignored; blank cells become NaN
    and are imputed inside the saved pipelines.
    """
    absent = [c for c in FEATURES if c not in df.columns]
    if absent:
        raise ValueError(f"Missing feature column(s): {', '.join(absent)}")
    X = pd.DataFrame(index=df.index)
    for col in FEATURES:
        series = df[col]
        if col == "Weekend":
            X[col] = to_flag(series)
        elif col in TEXT_COLS:
            X[col] = pd.Series([_clean_text(v) for v in series], index=df.index, dtype="object")
        else:
            numeric = pd.to_numeric(series, errors="coerce").astype("float64")
            X[col] = numeric.replace([np.inf, -np.inf], np.nan)
    return X


def prepare_target(df: pd.DataFrame) -> pd.Series:
    """Return the ``Revenue`` column as a 0/1 integer Series (raises ValueError if invalid)."""
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    flags = to_flag(df[TARGET])
    bad = int(flags.isna().sum())
    if bad:
        raise ValueError(
            f"'{TARGET}' has {bad} value(s) that are blank or not TRUE/FALSE (1/0)."
        )
    return flags.astype(int)


# --------------------------------------------------------------------------- scoring
def positive_scores(model, X) -> np.ndarray:
    """Score of the 'Purchase' class: probability when available, else the decision value."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, list(model.classes_).index(1)]
    return np.asarray(model.decision_function(X))


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Accuracy, AUC, Precision, Recall, F1 and MCC (positive class = Purchase)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    two_classes = np.unique(y_true).size == 2
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_score)) if two_classes else float("nan"),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def evaluate_model(model, X, y) -> dict:
    """Predict with a saved pipeline (no fitting) and return metrics + raw outputs."""
    y_pred = np.asarray(model.predict(X)).astype(int)
    y_score = positive_scores(model, X)
    return {"metrics": compute_metrics(y, y_pred, y_score), "y_pred": y_pred, "y_score": y_score}


def evaluate_all(models: dict, X, y) -> dict:
    return {name: evaluate_model(model, X, y) for name, model in models.items()}


def metrics_table(results: dict) -> pd.DataFrame:
    """One row per model, columns exactly as required by the assignment table."""
    rows = [{"ML Model Name": name, **res["metrics"]} for name, res in results.items()]
    return pd.DataFrame(rows, columns=["ML Model Name", *METRICS])


# --------------------------------------------------------------------------- persistence
def load_models(model_dir: Path | str = MODEL_DIR) -> dict:
    """Load the five saved pipelines (raw feature DataFrame in, class label out)."""
    model_dir = Path(model_dir)
    return {name: joblib.load(model_dir / filename) for name, filename in MODEL_FILES.items()}


def load_model_info(model_dir: Path | str = MODEL_DIR) -> dict:
    path = Path(model_dir) / "model_info.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_reference_metrics(model_dir: Path | str = MODEL_DIR) -> pd.DataFrame | None:
    path = Path(model_dir) / "test_metrics.csv"
    return pd.read_csv(path) if path.exists() else None
