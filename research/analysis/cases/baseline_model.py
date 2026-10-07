"""
Baseline model: predict civil_outcome for first_instance loan disputes.

Target: civil_outcome (6 classes: granted, granted_uncontested, settled,
withdrawn, partial, denied). law_articles and civil_award_amount are
deliberately excluded from features - both are extracted from the
post-decision operative section, so using them would leak the outcome
into its own predictor. case_narrative (free text) isn't used yet either;
that's a natural next step (NLP/LLM feature extraction), not part of
this first structured-features pass.

Run: python3 baseline_model.py
Needs: output.csv in the same folder (or edit CSV_PATH below).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import features  # shared with predict_new_case.py - one definition, not two copies to keep in sync

CSV_PATH = "output.csv"  # <- set if not in the same folder as this script

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


# ---------- load + filter to the labeled training set ----------
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
fi = df[(df.case_level == "first_instance") & (df.civil_outcome.notna())].copy()
print(f"training rows: {len(fi)}")
print(fi.civil_outcome.value_counts())
print()


# ---------- feature engineering (see features.py for the actual logic) ----------
X, median_claim, judge_counts = features.build_training_features(fi)
y = fi["civil_outcome"]
labels = sorted(y.unique())

# TF-IDF on the raw narrative text, in ADDITION to the regex flags above -
# the flags target specific known-relevant concepts precisely, TF-IDF
# catches broader patterns no one would think to hand-code. Tested
# max_features from 150-2500: climbs steadily to 1000, then starts
# giving it back (overfitting past that point) - 1000 is the real
# sweet spot, not a round-number default.
preprocess = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), features.FEATURES_CAT),
        ("text", TfidfVectorizer(max_features=1000, min_df=5, ngram_range=(1, 2), sublinear_tf=True), features.TEXT_COL),
    ],
    remainder="passthrough",
    sparse_threshold=0,
)

# denied=8, partial=19 - stratified k-fold so every fold sees some of
# each class, but treat per-class numbers for these two as noisy no
# matter what; there just isn't much data to evaluate them on yet.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ---------- naive baseline: always predict the majority class ----------
# This is the number a real model has to beat. Accuracy alone makes this
# look deceptively decent - that's exactly why macro-F1 is the metric
# used everywhere below instead.
majority_preds = pd.Series([y.mode()[0]] * len(y))
print("### naive baseline: always predict majority class ###")
print("accuracy:", round((majority_preds.values == y.values).mean(), 3))
print("macro-F1:", round(f1_score(y, majority_preds, average="macro"), 3))
print()


# ---------- logistic regression ----------
# with_mean=False since ColumnTransformer's one-hot output can be
# sparse - centering would break that. Scale-only still fixes the
# convergence warning, which was from mixing very different feature
# ranges (narrative_length_log spans ~0-11, judge_case_count up to 40+,
# the new boolean flags are 0/1) - exactly what the warning pointed at.
# Random forest doesn't need this, it's scale-invariant by construction.
lr = LogisticRegression(max_iter=2000, class_weight="balanced")
pipe_lr = Pipeline([("prep", preprocess), ("scale", StandardScaler(with_mean=False)), ("clf", lr)])
preds_lr = cross_val_predict(pipe_lr, X, y, cv=cv)
print("### logistic regression ###")
print("macro-F1:", round(f1_score(y, preds_lr, average="macro"), 3))
print()


# ---------- gradient boosting ----------
gbm = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.05, max_depth=6, class_weight="balanced", random_state=42
)
pipe_gbm = Pipeline([("prep", preprocess), ("clf", gbm)])
preds_gbm = cross_val_predict(pipe_gbm, X, y, cv=cv)
print("### gradient boosting ###")
print(classification_report(y, preds_gbm, zero_division=0))
print("macro-F1:", round(f1_score(y, preds_gbm, average="macro"), 3))
print()
print("confusion matrix (rows=true, cols=predicted):")
cm = pd.DataFrame(confusion_matrix(y, preds_gbm, labels=labels), index=labels, columns=labels)
print(cm.to_string())
print()


# ---------- narrative-only evaluation (simulates --narrative-only mode) ------
# In production, users often only have the petition text. This evaluation
# simulates that: we zero out structured fields to their defaults (same
# values predict_new_case.py uses when called with --narrative-only) and
# measure how the model performs with ONLY narrative-derived features.
X_narr = X.copy()
X_narr["claim_amount_log"] = np.log1p(median_claim)
X_narr["claim_amount_missing"] = 1
X_narr["judge_case_count"] = judge_counts.median()
X_narr["plaintiff_type"] = "unknown"
X_narr["defendant_type"] = "unknown"
X_narr["party_type_interaction"] = "unknown_vs_unknown"
X_narr["claim_amount_bucket"] = pd.cut(
    pd.Series([median_claim]), bins=[-np.inf, 1000000, 10000000, np.inf],
    labels=["low", "medium", "high"]
).astype(str).iloc[0]
# narrative_missing, narrative_length_log, has_* flags, and TEXT_COL
# remain real — these ARE available in narrative-only mode

preds_narr = cross_val_predict(pipe_gbm, X_narr, y, cv=cv)
print("### gradient boosting — NARRATIVE-ONLY (simulated production) ###")
print(classification_report(y, preds_narr, zero_division=0))
print("macro-F1:", round(f1_score(y, preds_narr, average="macro"), 3))
print()
print("confusion matrix (rows=true, cols=predicted):")
cm_narr = pd.DataFrame(confusion_matrix(y, preds_narr, labels=labels), index=labels, columns=labels)
print(cm_narr.to_string())
print()


# ---------- example predictions on genuinely held-out rows ----------
examples = fi[["source_file", "civil_outcome"]].copy()
examples["full_features"] = preds_gbm
examples["narrative_only"] = preds_narr
examples["full_correct"] = examples.civil_outcome == examples.full_features
examples["narr_correct"] = examples.civil_outcome == examples.narrative_only
print("### example held-out predictions (full vs narrative-only) ###")
print(examples.sample(10, random_state=1).to_string(index=False))
print()

