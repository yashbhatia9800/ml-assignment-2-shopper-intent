# Online Shoppers Purchase-Intent Classification

Machine Learning - Assignment 2. Five classifiers on one public UCI dataset, scored with six metrics and served through an interactive Streamlit app.

## a. Problem statement

An online shop would like to know, while a visitor is still browsing, whether the session will end in a purchase, so that a discount, a chat prompt or a product recommendation goes to the visitors who need it instead of to everyone.

This project predicts that outcome from the behaviour logged during a single browsing session: pages visited and time spent per page category, Google Analytics bounce / exit / page-value figures, the date, and visitor and traffic details. It is a **binary classification** task (`Revenue` = 1 when the session ended in a purchase, 0 otherwise) with **imbalanced classes** (about 15.6 % purchases).

Five classifiers - Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Naive Bayes and Support Vector Machine - are tuned and trained on the same data and compared with Accuracy, AUC, Precision, Recall, F1 and the Matthews Correlation Coefficient (MCC). A model that always answers "no purchase" already scores an accuracy of 0.8435 on the hold-out data, so MCC, which also penalises missed buyers and false alarms, is used as the deciding metric.

## b. Dataset description

| Property | Value |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository, dataset id 468 - <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset> |
| Citation | Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5F88Q> |
| Licence | CC BY 4.0 |
| Instances | 12,330 browsing sessions (assignment minimum: 500) |
| Features | 17 inputs (10 numeric, 7 categorical) plus the target (assignment minimum: 12 features) |
| Target | `Revenue` - did the session end in a purchase? |
| Class balance | 10,422 no purchase (84.5 %) and 1,908 purchase (15.5 %) |
| Missing values | none in the UCI file (the preprocessing still imputes, so uploaded files with gaps work) |
| Collection | Each session belongs to a different user over one year, to avoid bias towards a campaign, special day, user profile or period (as described by the dataset's creators) |

| Feature | Type | Meaning |
|---|---|---|
| `Administrative` | count | Administrative pages visited in the session |
| `Administrative_Duration` | time | Total time spent on administrative pages |
| `Informational` | count | Informational pages visited in the session |
| `Informational_Duration` | time | Total time spent on informational pages |
| `ProductRelated` | count | Product-related pages visited in the session |
| `ProductRelated_Duration` | time | Total time spent on product-related pages |
| `BounceRates` | ratio | Average bounce rate of the pages visited |
| `ExitRates` | ratio | Average exit rate of the pages visited |
| `PageValues` | number | Average page value of the pages visited |
| `SpecialDay` | 0 - 1 | Closeness of the visit to a special day (for example Mother's Day) |
| `Month` | text | Month of the visit |
| `OperatingSystems` | ID | Operating-system identifier |
| `Browser` | ID | Browser identifier |
| `Region` | ID | Visitor region identifier |
| `TrafficType` | ID | Traffic-source identifier |
| `VisitorType` | text | `Returning_Visitor`, `New_Visitor` or `Other` |
| `Weekend` | 0 / 1 | 1 if the visit happened on a weekend |
| **`Revenue`** | 0 / 1 | **Target** - 1 if the session ended in a purchase |

**Data preparation**

1. 125 exact duplicate rows are dropped (all of them "no purchase" sessions), leaving 12,205 sessions with 1,908 purchases (15.63 %).
2. A stratified 80 / 20 split with a fixed seed (`20261104`) gives 9,764 training rows and 2,441 hold-out rows (382 purchases, 15.65 %). The hold-out rows are shipped as `test_data.csv`; they are never used for fitting or tuning.
3. Preprocessing, stored inside every saved model: the 11 numeric columns (the 10 numeric features and the 0/1 `Weekend` flag) get median imputation and standardisation; `Month`, `VisitorType` and the ID-coded `OperatingSystems`, `Browser`, `Region` and `TrafficType` get most-frequent imputation and one-hot encoding (unseen categories are ignored). Together that gives 74 model inputs.
4. Tuning: a 5-fold stratified grid search on the training rows only, scored with MCC. The best setting is refit on all training rows and evaluated once on the hold-out rows.

## c. GitHub Repository Link

**<https://github.com/yashbhatia9800/ml-assignment-2-shopper-intent>**

```text
ml-assignment-2-shopper-intent/
|-- app.py                  Streamlit app: loads the saved models, never re-trains
|-- requirements.txt        pinned dependencies for local runs and Streamlit Community Cloud
|-- README.md               this file
|-- test_data.csv           the 2,441 hold-out rows (17 features + Revenue) to upload in the app
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- svm.joblib
|   |-- model_info.json     split sizes, selected hyper-parameters, CV scores, library versions
|   `-- test_metrics.csv    hold-out metrics written by train_models.py
|-- shopper_utils.py        feature lists, cleaning and metric helpers shared by trainer and app
|-- train_models.py         split, grid search, evaluation and saving of the five models
|-- analysis.py             confusion matrices, bootstrap intervals and correlations quoted below
|-- ML_Assignment_2.ipynb   notebook that re-runs the whole pipeline (written for BITS Virtual Lab)
|-- data/online_shoppers_intention.csv   raw UCI file
`-- .streamlit/config.toml  app theme
```

Every `.joblib` file is a complete scikit-learn `Pipeline` (preprocessing plus classifier), so the imputers, scaler and one-hot encoder travel with the model and no separate preprocessing files are needed.

## d. Models used

All five models use the same split, the same preprocessing and the same tuning protocol. The metrics below are computed on the 2,441 hold-out rows of `test_data.csv`, with "purchase" as the positive class. They are exactly what the app shows when `test_data.csv` is uploaded. AUC comes from `predict_proba`, except for the SVM, where it comes from the signed `decision_function` score.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8533 | 0.9083 | 0.5220 | 0.7461 | 0.6142 | 0.5400 |
| Decision Tree | 0.8947 | 0.9155 | 0.6855 | 0.6047 | 0.6426 | 0.5827 |
| KNN | 0.8816 | 0.8152 | 0.7163 | 0.4031 | 0.5159 | 0.4788 |
| Naive Bayes | 0.8247 | 0.8382 | 0.4582 | 0.6597 | 0.5408 | 0.4478 |
| Support Vector Machine | 0.8755 | 0.9094 | 0.5786 | 0.7513 | 0.6538 | 0.5867 |

### Observations on model performance

Confusion counts (TP / FP / FN) on the 2,441 hold-out rows, of which 382 are real purchases: Logistic Regression 285 / 261 / 97, Decision Tree 231 / 106 / 151, KNN 154 / 61 / 228, Naive Bayes 252 / 298 / 130, Support Vector Machine 287 / 209 / 95.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Finds most buyers (recall 0.7461, 285 of 382) but flags 546 sessions as purchases, so 261 are false alarms and precision is only 0.5220. Class weighting and strong regularisation (C = 0.01) push it towards recall. Its AUC (0.9083) is within 0.01 of the best (0.9155), so it ranks sessions almost as well as the best model; its MCC (0.5400) is held back by the false alarms. PageValues has by far the largest coefficient (+1.632 per standard deviation, next largest +0.594). Fast and easy to explain. |
| Decision Tree | Best accuracy (0.8947) and best AUC (0.9155). Cross-validation chose a small tree (depth 5, 30 leaves, no class weighting) and 77 % of its impurity reduction comes from a single feature, PageValues, so a handful of rules is enough on this data. It is the conservative model: precision 0.6855 (only 106 false alarms) but recall 0.6047 (151 buyers missed), giving MCC 0.5827, a hair below the SVM. Fully interpretable and cheap to run. |
| KNN | Highest precision (0.7163: 154 of the 215 sessions it flags really buy) but the lowest recall (0.4031, 228 of 382 buyers missed) and the lowest AUC (0.8152), so F1 is only 0.5159. With k = 7 and Euclidean distance over 74 standardised or one-hot columns, every column pulls equally on the distance, which probably dilutes the one decisive feature (PageValues) among many weak ones - a plausible reason, not tested here. Its accuracy (0.8816) looks good mainly because 84 % of sessions are "no purchase". It keeps the whole training set and offers no model to inspect. |
| Naive Bayes | Weakest overall: lowest accuracy (0.8247, below the 0.8435 of always answering "no purchase"), lowest precision (0.4582) and lowest MCC (0.4478), although its MCC gap to KNN is within sampling noise. It over-predicts purchases (550 flagged against 382 real, 298 false alarms). Its assumptions fit this data badly: Gaussian features for inputs that are mostly zero (78 % of sessions have PageValues = 0, 79 % have Informational = 0) and for one-hot columns, and independence for strongly correlated inputs (BounceRates and ExitRates r = 0.91, ProductRelated and ProductRelated_Duration r = 0.86). It needed heavy smoothing (var_smoothing = 0.1); with the default 1e-9 the same pipeline reaches accuracy 0.2511 at recall 0.9921. Trains in seconds. |
| Support Vector Machine | Best F1 (0.6538), best recall (0.7513, 287 of 382 buyers) and best MCC (0.5867). The RBF kernel with balanced class weights (C = 10, gamma = 0.003) finds about as many buyers as Logistic Regression (287 against 285) with 52 fewer false alarms (209 against 261), and clearly beats Naive Bayes (35 more buyers found, 89 fewer false alarms). Costs: the slowest to tune (its grid search took about 99 s against 14 s or less for the others on the machine used), 4,286 support vectors (43.9 % of the training rows), and no native probabilities, so its AUC (0.9094) is computed from decision scores. |
| **Overall, Winner for your dataset?** | **Support Vector Machine.** It has the highest MCC (0.5867) and F1 (0.6538) and the best recall (0.7513). MCC decides because 84 % of the sessions are "no purchase", which flatters accuracy. A paired bootstrap on the hold-out rows shows the SVM beating Logistic Regression (MCC difference +0.047, 95 % interval +0.020 to +0.074), KNN (+0.108, +0.059 to +0.156) and Naive Bayes (+0.139, +0.095 to +0.182), but only tying with the Decision Tree (+0.004, -0.033 to +0.041). The tree has the best accuracy and AUC, a slightly higher cross-validated MCC (0.6065 against 0.5952) and is simpler and cheaper, so choose the SVM to catch the most buyers and the Decision Tree if interpretability and speed matter more. |

### Hyper-parameters and training protocol

| Model | Grid searched (candidates) | Selected | CV MCC (mean +/- std over 5 folds) |
|---|---|---|---|
| Logistic Regression | `C` in {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 10, 100} x `class_weight` in {None, balanced} (18) | `C=0.01`, `class_weight=balanced` | 0.5501 +/- 0.0153 |
| Decision Tree | `criterion` in {gini, entropy} x `max_depth` in {3, 4, 5, 6, 8, 10, 15, None} x `min_samples_leaf` in {1, 5, 10, 25, 50, 100} x `class_weight` in {None, balanced} (192) | `criterion=gini`, `max_depth=5`, `min_samples_leaf=25`, `class_weight=None` | 0.6065 +/- 0.0203 |
| KNN | `n_neighbors` in {3, 5, 7, 11, 15, 25, 35} x `weights` in {uniform, distance} x `p` in {1, 2} (28) | `n_neighbors=7`, `weights=uniform`, `p=2` | 0.4451 +/- 0.0160 |
| Naive Bayes (Gaussian) | `var_smoothing` in {1e-9, 1e-6, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10} (10) | `var_smoothing=0.1` | 0.4122 +/- 0.0182 |
| Support Vector Machine (RBF) | `C` in {1, 5, 10, 50, 100} x `gamma` in {0.01, 0.003, 0.001} x `class_weight` in {None, balanced} (30) | `C=10`, `gamma=0.003`, `class_weight=balanced` | 0.5952 +/- 0.0160 |

Every selected numeric value (`C`, `gamma`, `max_depth`, `min_samples_leaf`, `n_neighbors`, `var_smoothing`) lies strictly inside its grid, so no search ran into the edge of a range. All other settings are scikit-learn defaults, except `max_iter=3000` for Logistic Regression and `random_state=20261104` for the Decision Tree. The selected hyper-parameters are also shown in the app's "About & data" tab and stored in `model/model_info.json`.

## Streamlit app

`app.py` loads the five saved pipelines from `model/` and computes every number from the uploaded CSV with those models. Nothing is trained or fitted inside the app.

| Required feature | Where it is in the app |
|---|---|
| Dataset upload (CSV) | Sidebar uploader for the test CSV (the 17 feature columns plus `Revenue`). Without an upload the bundled `test_data.csv` is used. A missing column, an empty file or an unusable target gives a clear message instead of a crash. |
| Model selection dropdown | Sidebar "Choose a classifier": all five models. |
| Evaluation metrics | Six metric cards (Accuracy, AUC, Precision, Recall, F1, MCC) for the selected model, plus a "Compare all five" tab with the same table for every model and a bar chart. |
| Confusion matrix / classification report | "Selected model" tab: confusion matrix, ROC curve and classification report. The "Row-level predictions" tab lists the prediction and purchase score of every row (downloadable as CSV). |

Uploaded files may use `TRUE`/`FALSE` or `1`/`0` in `Revenue` and `Weekend`; column order does not matter and extra columns are ignored. Upload only the test data: the free Streamlit tier has limited capacity.

## Run it yourself

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Reproduce the training and the extra analysis:

```bash
python train_models.py                      # full grid search; rewrites model/, test_data.csv, model/test_metrics.csv
python train_models.py --out-dir lab_run    # same, written to a separate folder
python train_models.py --no-tune            # fixed hyper-parameters, much faster
python analysis.py                          # confusion matrices, bootstrap intervals, correlations used above
```

The full grid search takes a few minutes (the SVM dominates). The saved models were built with scikit-learn 1.9.1 on Python 3.12, and scikit-learn pickles can only be loaded reliably with the same version, which is why `requirements.txt` pins exact versions.

## Deploy on Streamlit Community Cloud

1. Go to <https://streamlit.io/cloud> and sign in with GitHub.
2. Click **New app** and select this repository.
3. Choose branch `main` and main file `app.py`.
4. Open *Advanced settings* and select Python 3.12, the version the models were built with (choose it before deploying).
5. Click **Deploy**.

## Notes and limitations

- The results come from one stratified hold-out of 2,441 rows with 382 purchases, so they carry sampling noise: the 95 % bootstrap intervals for MCC are about 0.08 to 0.10 wide (for example 0.545 to 0.626 for the SVM). Differences smaller than that, such as SVM against Decision Tree or KNN against Naive Bayes, should not be over-read. The bootstrap (2,000 resamples, `python analysis.py`) resamples the hold-out rows only; the models are not re-trained for each resample.
- No decision-threshold tuning was done: every model uses its default cut-off (probability 0.5, or decision value 0 for the SVM), so the precision-recall balance reflects each model's calibration and class weighting. Another threshold would trade precision for recall.
- Hyper-parameters were chosen on the training rows only; the hold-out rows were scored once per model.
- The data comes from a single retailer's year of sessions, so the findings may not transfer to other shops or periods.

## Dataset citation

Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5F88Q> (licence CC BY 4.0).
