"""
Phase 2 — Model Evaluation
Computes outputs 07–12 from the analysis plan.
Re-runs cross-validation and captures per-fold scores, confusion matrix,
feature importance, and top TF-IDF terms.

Run: python3 analysis_phase2.py
"""

import numpy as np
import pandas as pd
import warnings, os

warnings.filterwarnings("ignore")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import features

CSV_PATH = "output.csv"
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
fi = df[(df.case_level == "first_instance") & (df.civil_outcome.notna())].copy()
print(f"Training on {len(fi)} labeled first_instance rows")

X, median_claim, judge_counts = features.build_training_features(fi)
y = fi["civil_outcome"]
labels = sorted(y.unique())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

preprocess = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), features.FEATURES_CAT),
        ("text", TfidfVectorizer(max_features=1000, min_df=5, ngram_range=(1, 2), sublinear_tf=True), features.TEXT_COL),
    ],
    remainder="passthrough",
    sparse_threshold=0,
)

# ══════════════════════════════════════════════════════════════
# 07 — Cross-Validation Fold Scores
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("07 — Cross-Validation Fold Scores")
print("=" * 60)

models = {}

# Naive baseline
majority = y.mode()[0]
majority_preds = pd.Series([majority] * len(y))
naive_f1 = f1_score(y, majority_preds, average="macro")
print(f"Naive baseline (always '{majority}'): macro-F1 = {naive_f1:.3f}")

# Logistic Regression
lr = LogisticRegression(max_iter=2000, class_weight="balanced")
pipe_lr = Pipeline([("prep", preprocess), ("scale", StandardScaler(with_mean=False)), ("clf", lr)])
scores_lr = cross_val_score(pipe_lr, X, y, cv=cv, scoring="f1_macro")
preds_lr = cross_val_predict(pipe_lr, X, y, cv=cv)
print(f"\nLogistic Regression fold scores: {[round(s, 3) for s in scores_lr]}")
print(f"  mean={scores_lr.mean():.3f} ± {scores_lr.std():.3f}")

# HistGradientBoosting
gbm = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.05, max_depth=6,
    class_weight="balanced", random_state=42
)
pipe_gbm = Pipeline([("prep", preprocess), ("clf", gbm)])
scores_gbm = cross_val_score(pipe_gbm, X, y, cv=cv, scoring="f1_macro")
preds_gbm = cross_val_predict(pipe_gbm, X, y, cv=cv)
print(f"\nHistGBM fold scores: {[round(s, 3) for s in scores_gbm]}")
print(f"  mean={scores_gbm.mean():.3f} ± {scores_gbm.std():.3f}")

# RandomForest for comparison
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
pipe_rf = Pipeline([("prep", preprocess), ("clf", rf)])
scores_rf = cross_val_score(pipe_rf, X, y, cv=cv, scoring="f1_macro")
preds_rf = cross_val_predict(pipe_rf, X, y, cv=cv)
print(f"\nRandomForest fold scores: {[round(s, 3) for s in scores_rf]}")
print(f"  mean={scores_rf.mean():.3f} ± {scores_rf.std():.3f}")

# CalibratedClassifierCV (production model)
pipe_cal_base = Pipeline([("prep", preprocess), ("clf", gbm)])
# For CV scoring of calibrated model, we need to do it manually
# since CalibratedClassifierCV with cv=5 already does internal CV
scores_cal = []
preds_cal = pd.Series(index=y.index, dtype=str)
for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    cal = CalibratedClassifierCV(
        Pipeline([("prep", preprocess), ("clf", HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=6,
            class_weight="balanced", random_state=42
        ))]),
        cv=3, method="sigmoid"  # inner CV=3 to avoid too-small folds
    )
    cal.fit(X_train, y_train)
    fold_preds = cal.predict(X_test)
    fold_f1 = f1_score(y_test, fold_preds, average="macro")
    scores_cal.append(fold_f1)
    preds_cal.iloc[test_idx] = fold_preds

scores_cal = np.array(scores_cal)
print(f"\nCalibrated HistGBM fold scores: {[round(s, 3) for s in scores_cal]}")
print(f"  mean={scores_cal.mean():.3f} ± {scores_cal.std():.3f}")

# Save fold scores
fold_df = pd.DataFrame({
    "fold": range(1, 6),
    "logistic_regression": scores_lr.round(3),
    "histgbm": scores_gbm.round(3),
    "random_forest": scores_rf.round(3),
    "calibrated_histgbm": scores_cal.round(3),
})
fold_df.to_csv(f"{OUTPUT_DIR}/07_cv_fold_scores.csv", index=False, encoding="utf-8-sig")

# Save fold accuracy
fold_acc_lr = []
fold_acc_gbm = []
fold_acc_rf = []
fold_acc_cal = []
for train_idx, test_idx in cv.split(X, y):
    y_test = y.iloc[test_idx]
    fold_acc_lr.append(accuracy_score(y_test, preds_lr[test_idx]))
    fold_acc_gbm.append(accuracy_score(y_test, preds_gbm[test_idx]))
    fold_acc_rf.append(accuracy_score(y_test, preds_rf[test_idx]))
    fold_acc_cal.append(accuracy_score(y_test, preds_cal.iloc[test_idx]))

fold_acc_df = pd.DataFrame({
    "fold": range(1, 6),
    "logistic_regression": np.round(fold_acc_lr, 3),
    "histgbm": np.round(fold_acc_gbm, 3),
    "random_forest": np.round(fold_acc_rf, 3),
    "calibrated_histgbm": np.round(fold_acc_cal, 3),
})
fold_acc_df.to_csv(f"{OUTPUT_DIR}/07_cv_fold_accuracy.csv", index=False, encoding="utf-8-sig")
print()

# ══════════════════════════════════════════════════════════════
# 08 — Model Comparison
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("08 — Model Comparison")
print("=" * 60)

comparison_rows = [
    {"model": "naive_baseline", "macro_f1": round(naive_f1, 3),
     "accuracy": round((majority_preds.values == y.values).mean(), 3),
     "weighted_f1": round(f1_score(y, majority_preds, average="weighted"), 3)},
    {"model": "logistic_regression", "macro_f1": round(f1_score(y, preds_lr, average="macro"), 3),
     "accuracy": round(accuracy_score(y, preds_lr), 3),
     "weighted_f1": round(f1_score(y, preds_lr, average="weighted"), 3)},
    {"model": "random_forest", "macro_f1": round(f1_score(y, preds_rf, average="macro"), 3),
     "accuracy": round(accuracy_score(y, preds_rf), 3),
     "weighted_f1": round(f1_score(y, preds_rf, average="weighted"), 3)},
    {"model": "histgbm", "macro_f1": round(f1_score(y, preds_gbm, average="macro"), 3),
     "accuracy": round(accuracy_score(y, preds_gbm), 3),
     "weighted_f1": round(f1_score(y, preds_gbm, average="weighted"), 3)},
    {"model": "calibrated_histgbm", "macro_f1": round(f1_score(y, preds_cal, average="macro"), 3),
     "accuracy": round(accuracy_score(y, preds_cal), 3),
     "weighted_f1": round(f1_score(y, preds_cal, average="weighted"), 3)},
]
comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(f"{OUTPUT_DIR}/08_model_comparison.csv", index=False, encoding="utf-8-sig")
print(comp_df.to_string(index=False))
print()

# ══════════════════════════════════════════════════════════════
# 09 — Per-Class Precision/Recall/F1 (HistGBM)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("09 — Per-Class Metrics (HistGBM, via cross_val_predict)")
print("=" * 60)

report = classification_report(y, preds_gbm, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T
report_df = report_df[report_df.index.isin(labels + ["macro avg", "weighted avg", "accuracy"])]
report_df = report_df.round(3)
report_df.to_csv(f"{OUTPUT_DIR}/09_per_class_metrics.csv", encoding="utf-8-sig")
print(classification_report(y, preds_gbm, zero_division=0))
print()

# Also do calibrated HistGBM
print("--- Calibrated HistGBM per-class metrics ---")
report_cal = classification_report(y, preds_cal, output_dict=True, zero_division=0)
report_cal_df = pd.DataFrame(report_cal).T
report_cal_df = report_cal_df[report_cal_df.index.isin(labels + ["macro avg", "weighted avg", "accuracy"])]
report_cal_df = report_cal_df.round(3)
report_cal_df.to_csv(f"{OUTPUT_DIR}/09b_per_class_metrics_calibrated.csv", encoding="utf-8-sig")
print(classification_report(y, preds_cal, zero_division=0))
print()

# ══════════════════════════════════════════════════════════════
# 10 — Confusion Matrix (HistGBM)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("10 — Confusion Matrix (HistGBM)")
print("=" * 60)

cm = pd.DataFrame(confusion_matrix(y, preds_gbm, labels=labels), index=labels, columns=labels)
cm.to_csv(f"{OUTPUT_DIR}/10_confusion_matrix.csv", encoding="utf-8-sig")
print("Rows=true, Cols=predicted:")
print(cm.to_string())
print()

cm_cal = pd.DataFrame(confusion_matrix(y, preds_cal, labels=labels), index=labels, columns=labels)
cm_cal.to_csv(f"{OUTPUT_DIR}/10b_confusion_matrix_calibrated.csv", encoding="utf-8-sig")
print("--- Calibrated HistGBM Confusion Matrix ---")
print(cm_cal.to_string())
print()

# ══════════════════════════════════════════════════════════════
# 11 — Feature Importance (HistGBM via Permutation Importance)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("11 — Feature Importance (HistGBM via Permutation Importance)")
print("=" * 60)

from sklearn.inspection import permutation_importance

# Fit the full pipeline to get feature importance
pipe_gbm_full = Pipeline([("prep", preprocess), ("clf", gbm)])
pipe_gbm_full.fit(X, y)

# Get feature names from the ColumnTransformer
prep = pipe_gbm_full.named_steps["prep"]
cat_names = prep.named_transformers_["cat"].get_feature_names_out(features.FEATURES_CAT).tolist()
text_names = prep.named_transformers_["text"].get_feature_names_out().tolist()
num_names = [c for c in X.columns if c not in features.FEATURES_CAT + [features.TEXT_COL]]
all_feature_names = cat_names + text_names + num_names

clf_model = pipe_gbm_full.named_steps["clf"]

# Compute permutation importance on the transformed data
X_transformed = prep.transform(X)
# To save time, we can run permutation importance on the classifier directly
result = permutation_importance(clf_model, X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed, y, n_repeats=1, random_state=42)
importances = result.importances_mean

imp_df = pd.DataFrame({
    "feature": all_feature_names[:len(importances)],
    "importance": importances,
}).sort_values("importance", ascending=False)

imp_df.to_csv(f"{OUTPUT_DIR}/11_feature_importance.csv", index=False, encoding="utf-8-sig")
print("Top 30 features:")
print(imp_df.head(30).to_string(index=False))
print(f"\n... ({len(imp_df)} total features)")
print()

# ══════════════════════════════════════════════════════════════
# 12 — Top TF-IDF Terms
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("12 — Top TF-IDF Terms")
print("=" * 60)

# Extract TF-IDF terms from the fitted vectorizer
tfidf = prep.named_transformers_["text"]
vocab = tfidf.get_feature_names_out()

# Get TF-IDF feature importance from the HistGBM
tfidf_start = len(cat_names)
tfidf_end = tfidf_start + len(text_names)
tfidf_importances = importances[tfidf_start:tfidf_end]

tfidf_imp = pd.DataFrame({
    "term": text_names[:len(tfidf_importances)],
    "importance": tfidf_importances,
}).sort_values("importance", ascending=False)

tfidf_imp.to_csv(f"{OUTPUT_DIR}/12_top_tfidf_terms.csv", index=False, encoding="utf-8-sig")
print("Top 30 TF-IDF terms by HistGBM importance:")
print(tfidf_imp.head(30).to_string(index=False))
print()

print("=" * 60)
print("Phase 2 complete. Outputs saved to:", OUTPUT_DIR)
print("=" * 60)
