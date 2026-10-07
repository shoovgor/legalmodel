import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/Users/purevjavgerel/Desktop/analysis/research/analysis"
DATA_PATH = os.path.join(BASE_DIR, "cases/output.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

print("Loading data...")
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# EXACT replication of reggression.py data prep
model_df = df[df["case_level"].isin(["appellate", "supervisory"])].copy()
model_df = model_df[model_df["appeal_outcome"].notna()].copy()
model_df["шийдвэр_хэвээр"] = (model_df["appeal_outcome"] == "upheld").astype(int)

model_df["нэхэмжлэлийн_дүн_байгаа_эсэх"] = model_df["claim_amount_requested"].notna().astype(int)
model_df["нэхэмжлэлийн_дүн_log"] = np.log1p(model_df["claim_amount_requested"].fillna(0))

model_df["нэхэмжлэгч"] = model_df["plaintiff_type"].fillna("тодорхойгүй")
model_df["хариуцагч"] = model_df["defendant_type"].fillna("тодорхойгүй")
model_df["шүүхийн_шат"] = model_df["case_level"].fillna("тодорхойгүй")
model_df["талуудын_харилцаа"] = model_df["нэхэмжлэгч"] + "_vs_" + model_df["хариуцагч"]
model_df["гомдол_гаргагч"] = model_df["appealing_party"].fillna("тодорхойгүй")

model_df["холбогдох_шийдвэрийн_тоо"] = (
    model_df["referenced_decisions"]
    .fillna("")
    .apply(lambda x: len(x.split("|")) if x else 0)
)

print("Table 1: Dataset Summary")
if 'decision_date' in model_df.columns:
    model_df['year'] = pd.to_datetime(model_df['decision_date'], errors='coerce').dt.year

appellate_summary = {
    "total_observations": len(model_df),
    "appellate_count": (model_df["case_level"] == "appellate").sum(),
    "supervisory_count": (model_df["case_level"] == "supervisory").sum(),
    "upheld_count": model_df["шийдвэр_хэвээр"].sum(),
    "not_upheld_count": len(model_df) - model_df["шийдвэр_хэвээр"].sum(),
    "upheld_percentage": round(model_df["шийдвэр_хэвээр"].mean() * 100, 2)
}
pd.DataFrame([appellate_summary]).to_csv(os.path.join(TABLES_DIR, "18_appellate_dataset_summary.csv"), index=False)

plt.figure(figsize=(6, 5))
sns.countplot(x="шийдвэр_хэвээр", data=model_df, palette="Set2")
plt.title("Appellate Target Distribution")
plt.xticks([0, 1], ["Not Upheld (0)", "Upheld (1)"])
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "18_appellate_target_distribution.png"), dpi=300)
plt.close()


print("Table 2: Logistic Configuration")
config = [
    {"parameter": "model", "value": "LogisticRegression"},
    {"parameter": "class_weight", "value": "balanced"},
    {"parameter": "input_transformations", "value": "StandardScaler() on all inputs"},
    {"parameter": "target_encoding_configuration", "value": "NONE (Methodology claims M=5 target encoding, but implementation uses One-Hot Encoding)"},
    {"parameter": "categorical_encoding_actual", "value": "pd.get_dummies(drop_first=True)"},
    {"parameter": "CV_configuration", "value": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"},
    {"parameter": "regularization_penalty", "value": "l2"},
    {"parameter": "C", "value": "1.0"},
    {"parameter": "solver", "value": "lbfgs (default)"},
    {"parameter": "max_iter", "value": "100 (default)"},
    {"parameter": "random_state", "value": "42"}
]
pd.DataFrame(config).to_csv(os.path.join(TABLES_DIR, "19_logistic_configuration.csv"), index=False)


print("Building Model...")
feature_cols_numeric = ["нэхэмжлэлийн_дүн_log", "нэхэмжлэлийн_дүн_байгаа_эсэх", "холбогдох_шийдвэрийн_тоо"]
dummies = pd.get_dummies(
    model_df[["нэхэмжлэгч", "хариуцагч", "шүүхийн_шат", "гомдол_гаргагч"]],
    drop_first=True,
)

X = pd.concat([model_df[feature_cols_numeric].reset_index(drop=True),
               dummies.reset_index(drop=True)], axis=1)
y = model_df["шийдвэр_хэвээр"].reset_index(drop=True)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", random_state=42)),
])

# Fit on all data for odds ratios (as original script did)
pipe.fit(X, y)
lr_model = pipe.named_steps["lr"]

print("Table 3: Odds Ratios")
# Scaler affects the coefficients! We need to un-scale to get true odds ratios for unit changes.
# Actually, the user asked to reproduce the project's methodology. The original script printed:
# coef_df = pd.DataFrame({"Хувьсагч": X.columns, "Коэффициент": lr_model.coef_[0], "Нөлөөлөл (Odds Ratio)": np.exp(lr_model.coef_[0])})
# So it reported odds ratios on the SCALED features! This is a methodological issue, 
# meaning ORs represent 1 standard deviation change, not 1 unit change.
coefs = lr_model.coef_[0]
odds_ratios = np.exp(coefs)
directions = ["Positive" if c > 0 else "Negative" for c in coefs]

or_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient_beta": coefs,
    "odds_ratio": odds_ratios,
    "direction": directions
}).sort_values("odds_ratio", ascending=False)
or_df.to_csv(os.path.join(TABLES_DIR, "20_odds_ratios.csv"), index=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="odds_ratio", y="feature", data=or_df, palette="coolwarm")
plt.axvline(x=1, color='black', linestyle='--')
plt.title("Odds Ratios (Note: Features were StandardScaled)")
plt.xlabel("Odds Ratio (exp(beta))")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "20_odds_ratios.png"), dpi=300)
plt.close()


print("Table 4 & 5: Regression Performance & Confusion Matrix")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get raw predictions for metrics
y_pred = cross_val_predict(pipe, X, y, cv=cv)
y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

# Replicate original script's single metric manually, but also add fold-level for Table 4
fold_acc, fold_prec, fold_rec, fold_f1, fold_auc = [], [], [], [], []
for train_idx, test_idx in cv.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    p = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", random_state=42))])
    p.fit(X_tr, y_tr)
    pred = p.predict(X_te)
    prob = p.predict_proba(X_te)[:, 1]
    
    rep = classification_report(y_te, pred, output_dict=True)
    fold_acc.append(rep["accuracy"])
    fold_prec.append(rep["1"]["precision"])
    fold_rec.append(rep["1"]["recall"])
    fold_f1.append(rep["1"]["f1-score"])
    fold_auc.append(roc_auc_score(y_te, prob))

perf_df = pd.DataFrame({
    "fold": range(1, 6),
    "accuracy": fold_acc,
    "precision_upheld": fold_prec,
    "recall_upheld": fold_rec,
    "f1_upheld": fold_f1,
    "roc_auc": fold_auc
})
# Append mean row
mean_row = perf_df.mean().to_dict()
mean_row["fold"] = "Mean"
perf_df = pd.concat([perf_df, pd.DataFrame([mean_row])], ignore_index=True)
perf_df.to_csv(os.path.join(TABLES_DIR, "21_logistic_cv_results.csv"), index=False)

# Boxplot of fold metrics
melted_perf = perf_df[perf_df.fold != "Mean"].melt(id_vars="fold", var_name="metric", value_name="score")
plt.figure(figsize=(8, 5))
sns.boxplot(x="metric", y="score", data=melted_perf, palette="Set3")
plt.title("5-Fold CV Performance (Appellate Model)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "21_logistic_cv_performance.png"), dpi=300)
plt.close()

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual_Not_Upheld", "Actual_Upheld"], columns=["Pred_Not_Upheld", "Pred_Upheld"])
cm_df.to_csv(os.path.join(TABLES_DIR, "22_logistic_confusion_matrix.csv"))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred Not Upheld", "Pred Upheld"], yticklabels=["Actual Not Upheld", "Actual Upheld"])
plt.title("CV Confusion Matrix (Appellate Model)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "22_logistic_confusion_matrix.png"), dpi=300)
plt.close()


print("Table 6: ROC and PR Curves")
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Appellate)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "23_roc_curve.png"), dpi=300)
plt.close()

precision, recall, _ = precision_recall_curve(y, y_prob)
pr_auc = auc(recall, precision)
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Appellate)')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "24_precision_recall_curve.png"), dpi=300)
plt.close()

print("Generating Final Report Markdown")
report_md = """# First-Instance and Appellate Model Analysis Summary

This report summarizes the findings from the comprehensive evaluation of both the 6-class first-instance outcome classifier and the appellate/supervisory logistic regression model.

### 1. Most Strongly Associated Features (Six-Class Classifier)
Before any modeling, statistical association tests (Cramér's V) showed that the strongest categorical/binary predictors of the outcome are the presence of settlement language (`has_settlement`), opposition/withdrawal language (`has_opposition`, `has_withdrawal`), and the specific court location/type. 

### 2. Most Important Features (Model's Perspective)
When evaluated using Permutation Feature Importance on the full `HistGBM` pipeline, the model relies most heavily on structural metadata rather than just text:
1. `judge_case_count` (The caseload/frequency of the judge)
2. `claim_amount_log` (The size of the dispute)
3. `court_type_rural`
4. The TF-IDF term `эвлэрлийн` (related to settlement).

### 3. Hardest Classes to Predict
The minority classes are the hardest to predict. The model exhibits a strong bias toward predicting `granted`. Specifically, `partial` and `denied` are frequently misclassified as `granted`. The uncalibrated HistGBM achieved a macro-F1 of 0.639, but class-specific F1 scores drop heavily for `denied` (0.20) and `partial` (0.57).

### 4. The "Denied" Class Deficit
The `denied` class remains extremely difficult to identify due to massive class imbalance (120 out of 5,153 cases). While the uncalibrated HistGBM catches a few of them (Recall: 17%), applying Sigmoid calibration completely suppresses the model's confidence for this class below 50%. The calibrated model effectively never predicts `denied`.

### 5. Model Calibration
The uncalibrated model suffers from the overconfidence problem identified in the original research report. When predictions are heavily confident (>80%), they are highly accurate (97.7%). However, lower-confidence predictions are extremely noisy. 

### 6. Principal Error Patterns
The dominant error pattern is predicting the majority class (`granted`) when the true outcome is a nuanced minority class. Of the 1,348 total CV errors, 1,206 (89%) involved the model incorrectly guessing `granted`.

### 7. Appellate/Supervisory Regression (Odds Ratios)
The logistic regression model for appellate/supervisory cases yielded the following key insights (Note: ORs reflect changes per standard deviation due to scaling):
* Cases heard at the **supervisory** level are modeled to have higher odds of being upheld compared to appellate level.
* When the **defendant** is the appealing party, the odds of the original decision being upheld are higher.
* The number of **referenced decisions** has a negative odds ratio; more references lower the modeled odds of the decision being upheld.

### 8. Important Methodological Limitations
1. **Appellate Encoding Methodology Mismatch**: The project documentation and descriptive statistics state that "Smoothed Target Encoding with M=5" was used for categorical variables in the appellate model. However, the actual `LogisticRegression` pipeline implementation uses global One-Hot Encoding (`pd.get_dummies`) instead. The model does not use target encodings.
2. **Appellate Scaled Odds Ratios**: The original appellate implementation applied `StandardScaler()` to all features (including one-hot encoded dummies) *before* fitting the Logistic Regression, and then directly exponentiated the coefficients to report Odds Ratios. As a result, the reported Odds Ratios represent the effect of a 1 standard deviation increase in the feature (which is mathematically difficult to interpret for one-hot binary flags), rather than a 1 unit increase. 
3. **Causal Fallacy**: None of the coefficients, permutation importances, or odds ratios presented here represent causal effects. They are purely predictive associations. 
"""

with open(os.path.join(REPORTS_DIR, "model_analysis_summary.md"), "w") as f:
    f.write(report_md)

print("--- APPELLATE GENERATION COMPLETE ---")
