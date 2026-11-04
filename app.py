"""Shopper Intent Lab - Streamlit front-end for the five saved classifiers.

Nothing is trained here. The app loads the pipelines written by ``train_models.py``,
scores the CSV you upload (or the bundled hold-out sample) and displays the results.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

import shopper_utils as su

st.set_page_config(page_title="Shopper Intent Lab", page_icon="🛒", layout="wide")
sns.set_theme(style="white", context="notebook")

METRIC_HELP = {
    "Accuracy": "Share of all sessions classified correctly. Misleading when only ~15% of sessions buy.",
    "AUC": "Chance that a random buyer is scored above a random non-buyer (threshold-free ranking quality).",
    "Precision": "Of the sessions predicted to buy, the share that really bought.",
    "Recall": "Of the sessions that really bought, the share the model found.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Matthews correlation coefficient, from -1 to +1. Stays honest on imbalanced classes.",
}


# --------------------------------------------------------------------------- cached work
@st.cache_resource(show_spinner="Loading the five saved models ...")
def get_models() -> dict:
    return su.load_models()


@st.cache_data(show_spinner="Scoring every model on your file ...")
def score_all(frame: pd.DataFrame):
    """Predict with the saved models only - no fitting anywhere in this app."""
    X = su.prepare_features(frame)
    y = su.prepare_target(frame)
    return y.to_numpy(), su.evaluate_all(get_models(), X, y)


@st.cache_data
def get_reference():
    return su.load_model_info(), su.load_reference_metrics()


# --------------------------------------------------------------------------- helpers
def fmt(value: float) -> str:
    return "n/a" if value is None or np.isnan(value) else f"{value:.4f}"


def read_upload(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    try:
        frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    return su.tidy_columns(frame)


def confusion_figure(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    totals = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, totals, out=np.zeros(cm.shape, dtype=float), where=totals > 0)
    labels = np.array([[f"{cm[i, j]:,}\n{pct[i, j]:.1%}" for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="GnBu",
        cbar=False,
        linewidths=2,
        linecolor="white",
        xticklabels=su.CLASS_NAMES,
        yticklabels=su.CLASS_NAMES,
        annot_kws={"fontsize": 12},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Counts and % of each actual class", fontsize=10)
    fig.tight_layout()
    return fig


def roc_figure(y_true, y_score, name, auc):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.plot(fpr, tpr, lw=2.2, color="#0f766e", label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey", label="Chance")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("ROC curve", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def comparison_figure(table: pd.DataFrame):
    long = table.melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 4.4))
    sns.barplot(data=long, x="Metric", y="Score", hue="ML Model Name", palette="Set2", ax=ax)
    ax.set_ylim(min(0.0, float(np.nanmin(long["Score"])) - 0.02), 1.0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=5, frameon=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- load models
try:
    models = get_models()
except FileNotFoundError:
    st.error("Saved models were not found in the `model/` folder. Run `python train_models.py` first.")
    st.stop()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🛒 Shopper Intent Lab")
    st.caption("Will this browsing session end in a purchase?")
    model_name = st.selectbox("1 · Choose a classifier", list(su.MODEL_FILES))
    uploaded = st.file_uploader(
        "2 · Upload test data (CSV)",
        type=["csv"],
        help="Upload only the test split - the free hosting tier has limited capacity.",
    )
    st.download_button(
        "⬇️ Download the sample test_data.csv",
        data=su.SAMPLE_CSV.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )
    with st.expander("Columns the CSV must contain"):
        st.markdown(
            "All 17 features **and** the target:\n\n"
            + ", ".join(f"`{c}`" for c in [*su.FEATURES, su.TARGET])
            + "\n\n`Revenue` may be `TRUE/FALSE` or `1/0`. Column order and extra columns are ignored."
        )
    st.caption("Models are loaded from `model/*.joblib`; nothing is re-trained in this app.")

# --------------------------------------------------------------------------- read data
st.title("Shopper Intent Lab")
st.caption(
    "Five classifiers trained on the UCI *Online Shoppers Purchasing Intention* data. "
    "Pick a model, upload a test CSV and inspect how well it predicts purchases."
)

try:
    if uploaded is None:
        frame = su.tidy_columns(pd.read_csv(su.SAMPLE_CSV))
        source = "the bundled sample `test_data.csv` (20 % hold-out split)"
    else:
        frame = read_upload(uploaded)
        source = f"your upload `{uploaded.name}`"
except pd.errors.EmptyDataError:
    st.error("The uploaded file is empty.")
    st.stop()
except (pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"Could not read the file as CSV: {exc}")
    st.stop()

absent = su.missing_columns(frame)
if absent:
    st.error(
        "The file is missing required column(s): "
        + ", ".join(f"`{c}`" for c in absent)
        + ". It needs all 17 feature columns plus the `Revenue` target."
    )
    st.stop()
if frame.empty:
    st.error("The file has a header but no data rows.")
    st.stop()

try:
    y_true, results = score_all(frame)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # unexpected input the models cannot score
    st.error(f"Could not score this file ({type(exc).__name__}: {exc}).")
    st.stop()

n_rows, n_pos = len(frame), int(y_true.sum())
st.info(
    f"**Data source:** {source} · **{n_rows:,}** rows · **{n_pos:,}** purchases ({n_pos / n_rows:.1%})"
)
if n_pos in (0, n_rows):
    st.warning("The file contains a single class, so AUC and the ROC curve are not defined.")

table = su.metrics_table(results)
res = results[model_name]

tab_model, tab_compare, tab_rows, tab_about = st.tabs(
    ["🎯 Selected model", "⚖️ Compare all five", "🔎 Row-level predictions", "ℹ️ About & data"]
)

# --------------------------------------------------------------------------- tab 1
with tab_model:
    st.subheader(model_name)
    for col, key in zip(st.columns(6), su.METRICS):
        col.metric(key, fmt(res["metrics"][key]), help=METRIC_HELP[key], border=True)

    left, right = st.columns(2)
    with left:
        st.markdown("##### Confusion matrix")
        fig = confusion_figure(y_true, res["y_pred"])
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("##### ROC curve")
        if np.isnan(res["metrics"]["AUC"]):
            st.info("ROC needs both classes in the file.")
        else:
            fig = roc_figure(y_true, res["y_score"], model_name, res["metrics"]["AUC"])
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("##### Classification report")
    report = classification_report(
        y_true,
        res["y_pred"],
        labels=[0, 1],
        target_names=su.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    accuracy = report.pop("accuracy", None)
    report_df = pd.DataFrame(report).T
    if accuracy is not None:
        report_df.loc["accuracy"] = [np.nan, np.nan, accuracy, float(n_rows)]
    st.table(
        report_df.style.format(
            {"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:,.0f}"},
            na_rep="–",
        )
    )

# --------------------------------------------------------------------------- tab 2
with tab_compare:
    st.subheader("All five models on the same data")
    styled = (
        table.set_index("ML Model Name")
        .style.format("{:.4f}", na_rep="n/a")
        .highlight_max(axis=0, color="#bbf7d0")
    )
    st.dataframe(styled, width="stretch")
    st.caption("Green = best value in each column.")

    best = table.loc[table["MCC"].idxmax()]
    st.success(
        f"🏆 Highest MCC on this file: **{best['ML Model Name']}** - MCC {best['MCC']:.4f}, "
        f"AUC {fmt(best['AUC'])}, F1 {best['F1']:.4f}, accuracy {best['Accuracy']:.4f}."
    )
    st.caption(
        "MCC is the headline metric: only about 15 % of sessions end in a purchase, "
        "so accuracy alone rewards a model that mostly answers 'no purchase'."
    )
    fig = comparison_figure(table)
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------------------------------- tab 3
with tab_rows:
    st.subheader(f"Predictions - {model_name}")
    view = pd.DataFrame(
        {
            "Actual label": [su.CLASS_NAMES[i] for i in y_true],
            "Predicted label": [su.CLASS_NAMES[i] for i in res["y_pred"]],
            "Purchase score": np.round(res["y_score"], 4),
            "Correct?": y_true == res["y_pred"],
        }
    )
    view = pd.concat([view, frame.reset_index(drop=True)], axis=1)
    if st.checkbox("Show only misclassified rows"):
        view = view[~view["Correct?"]]
    st.write(f"{len(view):,} row(s) shown")
    st.dataframe(view, width="stretch", height=420)
    st.download_button(
        "⬇️ Download these predictions (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{su.MODEL_FILES[model_name].split('.')[0]}.csv",
        mime="text/csv",
    )
    if model_name == "Support Vector Machine":
        st.caption("Purchase score for the SVM is its signed decision value (positive = purchase).")
    else:
        st.caption("Purchase score is the model's predicted probability of a purchase.")

# --------------------------------------------------------------------------- tab 4
with tab_about:
    info, reference = get_reference()
    st.subheader("Dataset")
    if info:
        ds, split, cfg = info["dataset"], info["split"], info["settings"]
        a, b, c, d = st.columns(4)
        a.metric("Rows used", f"{ds['rows_used']:,}", help=f"{ds['duplicates_removed']} exact duplicates removed")
        b.metric("Input features", ds["n_features"])
        c.metric("Purchase rate", f"{ds['positive_rate']:.1%}")
        d.metric("Train / test rows", f"{split['train_rows']:,} / {split['test_rows']:,}")
        st.caption(
            f"Stratified {1 - cfg['test_size']:.0%}/{cfg['test_size']:.0%} split (seed {cfg['seed']}); "
            f"hyper-parameters chosen with {cfg['cv_folds']}-fold CV on the training part, scored by "
            f"{cfg['selection_metric']}."
        )
    st.dataframe(
        pd.DataFrame(su.FEATURE_INFO, columns=["Column", "Type", "Meaning"]),
        width="stretch",
        hide_index=True,
    )

    if reference is not None:
        st.subheader("Reference metrics from training")
        st.caption("Hold-out results written by train_models.py - upload the bundled sample to reproduce them.")
        st.dataframe(
            reference.set_index("ML Model Name").style.format("{:.4f}", na_rep="n/a"),
            width="stretch",
        )

    if info:
        st.subheader("Selected hyper-parameters")
        rows = [
            {
                "Model": name,
                "Hyper-parameters": ", ".join(f"{k}={v}" for k, v in meta["selected_params"].items()),
                "CV MCC (train)": meta.get("cv_mcc_mean"),
            }
            for name, meta in info["models"].items()
        ]
        st.dataframe(
            pd.DataFrame(rows).style.format({"CV MCC (train)": "{:.4f}"}, na_rep="-"),
            width="stretch",
            hide_index=True,
        )
        trained_with = info["versions"]["scikit-learn"]
        if trained_with.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            st.warning(
                f"Models were trained with scikit-learn {trained_with} but this app runs "
                f"{sklearn.__version__}; results may differ."
            )
        st.caption(
            f"Trained with scikit-learn {trained_with} on Python {info['versions']['python']} · "
            f"running scikit-learn {sklearn.__version__}."
        )
