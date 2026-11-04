"""Extra analysis quoted in the README observations.

    python analysis.py

Reads the saved models in model/ and the hold-out file test_data.csv - nothing is re-trained.
Prints confusion matrices, the inputs the interpretable models lean on, bootstrap intervals for
MCC / F1 on the hold-out rows, and the correlation of a few redundant feature pairs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

import shopper_utils as su
import train_models as tm

N_BOOT = 2000
BOOT_SEED = 20261104
COMPARISONS = [
    ("Support Vector Machine", "Decision Tree"),
    ("Support Vector Machine", "Logistic Regression"),
    ("Support Vector Machine", "KNN"),
    ("Support Vector Machine", "Naive Bayes"),
    ("Decision Tree", "KNN"),
    ("KNN", "Naive Bayes"),
]
CORRELATED_PAIRS = [
    ("BounceRates", "ExitRates"),
    ("ProductRelated", "ProductRelated_Duration"),
    ("Administrative", "Administrative_Duration"),
]
ZERO_HEAVY = ["Administrative", "Informational", "PageValues", "BounceRates"]


def interval(values) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def dataset_facts(raw: pd.DataFrame) -> pd.DataFrame:
    """Print the counts quoted in the README and return the de-duplicated frame."""
    deduped = raw.drop_duplicates().reset_index(drop=True)
    print("Dataset")
    for label, frame in (("UCI file", raw), ("after dropping exact duplicates", deduped)):
        bought = int(su.prepare_target(frame).sum())
        print(
            f"  {label}: {len(frame):,} rows | purchases {bought:,} ({bought / len(frame):.2%})"
            f" | no purchase {len(frame) - bought:,}"
        )
    print(f"  exact duplicate rows dropped: {len(raw) - len(deduped)}")
    categories = {col: int(deduped[col].nunique()) for col in su.CODE_COLS + su.TEXT_COLS}
    print("  categories per one-hot column: " + ", ".join(f"{c}={n}" for c, n in categories.items()))
    n_numeric = len(su.NUMERIC_COLS)
    print(
        f"  model inputs: {n_numeric} numeric + {sum(categories.values())} one-hot"
        f" = {n_numeric + sum(categories.values())}"
    )
    return deduped


def holdout_checks(deduped: pd.DataFrame, shipped: pd.DataFrame) -> None:
    """test_data.csv must be the stratified hold-out rows; also re-fit the default Naive Bayes."""
    X, y = su.prepare_features(deduped), su.prepare_target(deduped)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=tm.TEST_SIZE, stratify=y, random_state=tm.SEED
    )
    expected = deduped.loc[X_test.index, su.FEATURES].reset_index(drop=True)
    expected["Weekend"] = su.to_flag(expected["Weekend"]).astype(int)
    expected[su.TARGET] = y_test.to_numpy()
    pd.testing.assert_frame_equal(
        expected, shipped[su.FEATURES + [su.TARGET]].reset_index(drop=True), check_dtype=False
    )
    print(f"\ntest_data.csv == the {len(X_test):,} stratified hold-out rows of the UCI file "
          f"(training rows: {len(X_train):,}; none shared)")

    default_nb = Pipeline([("prep", tm.make_preprocessor()), ("clf", GaussianNB())]).fit(X_train, y_train)
    scores = su.evaluate_model(default_nb, X_test, y_test)["metrics"]
    print("Naive Bayes with the default var_smoothing=1e-9 (same split and preprocessing)")
    print("  " + "  ".join(f"{name}={scores[name]:.4f}" for name in su.METRICS))


def main() -> None:
    raw = pd.read_csv(su.RAW_CSV)
    deduped = dataset_facts(raw)
    models = su.load_models()
    info = su.load_model_info()
    shipped = pd.read_csv(su.SAMPLE_CSV)
    holdout_checks(deduped, shipped)
    frame = su.tidy_columns(shipped)
    X, y = su.prepare_features(frame), su.prepare_target(frame).to_numpy()
    pred = {name: model.predict(X) for name, model in models.items()}
    print()

    print(f"Hold-out rows: {len(y):,} | purchases: {y.sum():,} ({y.mean():.2%})")
    print(f"Always answering 'no purchase' already scores accuracy {1 - y.mean():.4f}\n")

    print("Confusion matrices on the hold-out rows")
    for name, p in pred.items():
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        print(f"  {name:<24} TN={tn:<5} FP={fp:<4} FN={fn:<4} TP={tp:<4} predicted purchases={fp + tp}")

    print("\nWhat the interpretable models lean on")
    tree = models["Decision Tree"]
    shares = pd.Series(tree[-1].feature_importances_, index=tree[:-1].get_feature_names_out())
    print(f"  Decision tree: depth {tree[-1].get_depth()}, {tree[-1].get_n_leaves()} leaves; largest impurity-reduction shares")
    for feature, share in shares.nlargest(5).items():
        print(f"      {feature:<34} {share:.3f}")
    logreg = models["Logistic Regression"]
    coef = pd.Series(logreg[-1].coef_[0], index=logreg[:-1].get_feature_names_out())
    print("  Logistic regression: largest coefficients (numeric inputs are standardised)")
    for feature, weight in coef.reindex(coef.abs().nlargest(5).index).items():
        print(f"      {feature:<34} {weight:+.3f}")
    svm = models["Support Vector Machine"][-1]
    n_sv, n_train = int(svm.n_support_.sum()), info["split"]["train_rows"]
    print(f"  SVM: {n_sv:,} support vectors = {n_sv / n_train:.1%} of the {n_train:,} training rows")
    print(f"  Model inputs after preprocessing: {len(tree[:-1].get_feature_names_out())}")

    rng = np.random.default_rng(BOOT_SEED)
    mcc = {name: [] for name in pred}
    f1 = {name: [] for name in pred}
    for _ in range(N_BOOT):
        rows = rng.integers(0, len(y), len(y))
        for name, p in pred.items():
            mcc[name].append(matthews_corrcoef(y[rows], p[rows]))
            f1[name].append(f1_score(y[rows], p[rows], zero_division=0))

    print(f"\nBootstrap ({N_BOOT:,} resamples of the hold-out rows), 95% intervals")
    for name in pred:
        m_lo, m_hi = interval(mcc[name])
        f_lo, f_hi = interval(f1[name])
        print(f"  {name:<24} MCC [{m_lo:.3f}, {m_hi:.3f}]   F1 [{f_lo:.3f}, {f_hi:.3f}]")
    print("  Paired MCC differences (first minus second)")
    for first, second in COMPARISONS:
        diff = np.array(mcc[first]) - np.array(mcc[second])
        lo, hi = interval(diff)
        print(f"      {first} - {second}: {diff.mean():+.3f}  95% [{lo:+.3f}, {hi:+.3f}]")

    print("\nRedundant feature pairs (Pearson r, full UCI file)")
    for a, b in CORRELATED_PAIRS:
        print(f"  {a} vs {b}: {raw[a].corr(raw[b]):.2f}")

    print("\nShare of sessions whose value is exactly 0 (full UCI file)")
    for col in ZERO_HEAVY:
        print(f"  {col}: {(raw[col] == 0).mean():.0%}")


if __name__ == "__main__":
    main()
