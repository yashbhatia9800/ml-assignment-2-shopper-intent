"""Train, tune, evaluate and save the five classifiers.

Run from the project root::

    python train_models.py                      # writes model/ and test_data.csv
    python train_models.py --out-dir lab_run    # same outputs in a separate folder
    python train_models.py --no-tune            # fixed hyper-parameters, much faster

Workflow: stratified 80/20 split -> 5-fold grid search on the TRAINING part only
(scored with MCC) -> one evaluation on the untouched 20 % hold-out -> save the
fitted pipelines, the hold-out CSV and the metrics.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import shopper_utils as su

SEED = 20261104
TEST_SIZE = 0.20
CV_FOLDS = 5
MCC_SCORER = make_scorer(matthews_corrcoef)


# --------------------------------------------------------------------------- pipeline
def make_preprocessor() -> ColumnTransformer:
    """Impute + scale the numeric columns, impute + one-hot encode the nominal ones."""

    def numeric() -> Pipeline:
        return Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )

    def nominal() -> Pipeline:
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

    return ColumnTransformer(
        [
            ("numeric", numeric(), su.NUMERIC_COLS),
            ("codes", nominal(), su.CODE_COLS),
            ("text", nominal(), su.TEXT_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()), ("clf", estimator)])


def default_models() -> dict:
    """Fixed hyper-parameters used by --no-tune."""
    return {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=3000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=7, min_samples_leaf=10, random_state=SEED
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0),
    }


def search_space() -> dict:
    """name -> (estimator, grid). Grid keys address the 'clf' step of the pipeline."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000),
            {
                "clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 10, 100],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [3, 4, 5, 6, 8, 10, 15, None],
                "clf__min_samples_leaf": [1, 5, 10, 25, 50, 100],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "clf__n_neighbors": [3, 5, 7, 11, 15, 25, 35],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]},
        ),
        "Support Vector Machine": (
            # AUC for the SVM is computed from decision_function scores
            # (SVC(probability=True) is deprecated in scikit-learn 1.9).
            SVC(kernel="rbf"),
            {
                "clf__C": [1, 5, 10, 50, 100],
                "clf__gamma": [0.01, 0.003, 0.001],
                "clf__class_weight": [None, "balanced"],
            },
        ),
    }


def fit_model(name, estimator, grid, X_train, y_train, tune, cv, n_jobs):
    """Fit one pipeline (optionally grid-searched) and describe what was chosen."""
    started = time.perf_counter()
    pipe = build_pipeline(estimator)
    info: dict = {"file": su.MODEL_FILES[name]}
    if tune:
        search = GridSearchCV(
            pipe, grid, scoring=MCC_SCORER, cv=cv, n_jobs=n_jobs, refit=True
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best = search.best_index_
        info["candidates"] = len(search.cv_results_["params"])
        info["cv_mcc_mean"] = float(search.cv_results_["mean_test_score"][best])
        info["cv_mcc_std"] = float(search.cv_results_["std_test_score"][best])
    else:
        model = pipe.fit(X_train, y_train)
        info["candidates"] = 1
    clf_params = model.named_steps["clf"].get_params()
    info["selected_params"] = {
        key.removeprefix("clf__"): clf_params[key.removeprefix("clf__")] for key in grid
    }
    info["fit_seconds"] = round(time.perf_counter() - started, 1)
    return model, info


# --------------------------------------------------------------------------- helpers
def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialise {type(obj).__name__}")


def _fmt_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


# --------------------------------------------------------------------------- main routine
def run(out_dir=su.ROOT, data_path=su.RAW_CSV, tune=True, n_jobs=-1, verbose=True):
    """Train everything, write the artefacts and return the hold-out metrics table."""
    log = print if verbose else (lambda *args, **kwargs: None)
    out_dir = Path(out_dir).resolve()
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Place online_shoppers_intention.csv (UCI id 468) there."
        )
    raw = pd.read_csv(data_path)
    df = raw.drop_duplicates().reset_index(drop=True)
    removed = len(raw) - len(df)
    log(f"Loaded {len(raw):,} rows, dropped {removed} exact duplicates -> {len(df):,} rows")

    X = su.prepare_features(df)
    y = su.prepare_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    log(
        f"Train {len(X_train):,} rows ({y_train.mean():.2%} purchases) | "
        f"hold-out test {len(X_test):,} rows ({y_test.mean():.2%} purchases)"
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    spaces = search_space()
    defaults = default_models()
    fitted, details = {}, {}
    for name in su.MODEL_FILES:
        estimator, grid = spaces[name]
        if not tune:
            estimator = defaults[name]
        n_candidates = len(ParameterGrid(grid)) if tune else 1
        log(f"\n[{name}] fitting ({n_candidates} candidate(s) x {CV_FOLDS} folds)..." if tune
            else f"\n[{name}] fitting with fixed hyper-parameters...")
        model, info = fit_model(name, estimator, grid, X_train, y_train, tune, cv, n_jobs)
        fitted[name], details[name] = model, info
        score = f", CV MCC {info['cv_mcc_mean']:.4f}" if tune else ""
        log(f"  chosen: {_fmt_params(info['selected_params'])}{score}  ({info['fit_seconds']}s)")

    # hold-out file that ships with the repo (raw, un-scaled values; Revenue as 0/1)
    test_raw = df.loc[X_test.index, su.FEATURES].copy()
    test_raw["Weekend"] = su.to_flag(test_raw["Weekend"]).astype(int)
    test_raw[su.TARGET] = y_test.to_numpy()
    test_csv = out_dir / "test_data.csv"
    test_raw.to_csv(test_csv, index=False)

    for name, model in fitted.items():
        joblib.dump(model, model_dir / su.MODEL_FILES[name], compress=3)

    # Reload everything from disk and score the CSV exactly the way the app does.
    reloaded = su.load_models(model_dir)
    frame = pd.read_csv(test_csv)
    disk_table = su.metrics_table(
        su.evaluate_all(reloaded, su.prepare_features(frame), su.prepare_target(frame))
    )
    memory_table = su.metrics_table(su.evaluate_all(fitted, X_test, y_test))
    if not np.allclose(
        disk_table[su.METRICS].to_numpy(float),
        memory_table[su.METRICS].to_numpy(float),
        atol=1e-9,
        equal_nan=True,
    ):
        raise RuntimeError("Saved models / test_data.csv do not reproduce the in-memory metrics")
    disk_table.to_csv(model_dir / "test_metrics.csv", index=False)

    info_doc = {
        "dataset": {
            "name": "Online Shoppers Purchasing Intention (UCI id 468)",
            "target": su.TARGET,
            "raw_rows": int(len(raw)),
            "duplicates_removed": int(removed),
            "rows_used": int(len(df)),
            "n_features": len(su.FEATURES),
            "positive_rate": float(y.mean()),
        },
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "settings": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "selection_metric": "MCC",
            "tuned": bool(tune),
        },
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
        "models": details,
    }
    (model_dir / "model_info.json").write_text(
        json.dumps(info_doc, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )

    log("\nHold-out test metrics (20 % split, models reloaded from disk):")
    log(disk_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log(f"\nSaved to {model_dir} and {test_csv}")
    return disk_table


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the five shopper-intent classifiers.")
    parser.add_argument("--data", default=str(su.RAW_CSV), help="path to online_shoppers_intention.csv")
    parser.add_argument(
        "--out-dir",
        default=str(su.ROOT),
        help="folder that receives model/ and test_data.csv (default: project root)",
    )
    parser.add_argument("--no-tune", action="store_true", help="skip the grid search")
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel workers for the grid search")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.out_dir, args.data, tune=not args.no_tune, n_jobs=args.n_jobs)
