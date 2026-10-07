"""
Phase 3 — Calibration & Error Analysis
Computes outputs 13–15 from the analysis plan.
"""

import numpy as np
import pandas as pd
import warnings, os

warnings.filterwarnings("ignore")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import features

CSV_PATH = "output.csv"
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
fi = df[(df.case_level == "first_instance") & (df.civil_outcome.notna())].copy()
print(f"Working with {len(fi)} labeled first_instance rows")

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

gbm = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.05, max_depth=6,
    class_weight="balanced", random_state=42
)

# ══════════════════════════════════════════════════════════════
# Generate Calibrated Probabilities via CV
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("Generating Calibrated Probabilities via CV...")
print("=" * 60)

preds_cal = pd.Series(index=y.index, dtype=str)
preds_cal_proba = pd.DataFrame(index=y.index, columns=labels, dtype=float)

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    cal = CalibratedClassifierCV(
        Pipeline([("prep", preprocess), ("clf", HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=6,
            class_weight="balanced", random_state=42
        ))]),
        cv=3, method="sigmoid"
    )
    cal.fit(X_train, y_train)
    fold_preds = cal.predict(X_test)
    fold_proba = cal.predict_proba(X_test)
    
    preds_cal.iloc[test_idx] = fold_preds
    preds_cal_proba.iloc[test_idx] = fold_proba

# Extract top confidence and top predicted class
top_conf = preds_cal_proba.max(axis=1)
pred_class = preds_cal_proba.idxmax(axis=1)

# ══════════════════════════════════════════════════════════════
# 13 — Calibration Data
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("13 — Calibration Data")
print("=" * 60)

bins = np.linspace(0, 1, 11)
cal_rows = []

for c in labels:
    # True binary array for this class
    y_true_bin = (y == c).astype(int)
    # Probabilities predicted for this class
    y_prob = preds_cal_proba[c]
    
    for i in range(len(bins)-1):
        bin_min = bins[i]
        bin_max = bins[i+1]
        
        # Mask for predictions falling in this bin
        if i == len(bins) - 2: # include max in last bin
            in_bin = (y_prob >= bin_min) & (y_prob <= bin_max)
        else:
            in_bin = (y_prob >= bin_min) & (y_prob < bin_max)
            
        count = in_bin.sum()
        if count > 0:
            fraction_pos = y_true_bin[in_bin].mean()
            mean_pred_prob = y_prob[in_bin].mean()
        else:
            fraction_pos = np.nan
            mean_pred_prob = np.nan
            
        cal_rows.append({
            "class": c,
            "bin_min": bin_min,
            "bin_max": bin_max,
            "count": count,
            "mean_predicted_prob": mean_pred_prob,
            "fraction_positives": fraction_pos
        })

cal_df = pd.DataFrame(cal_rows)
cal_df.to_csv(f"{OUTPUT_DIR}/13_calibration_data.csv", index=False, encoding="utf-8-sig")
print(cal_df.head(10).to_string(index=False))
print("...")
print()


# ══════════════════════════════════════════════════════════════
# 14 — Confidence Distribution
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("14 — Confidence Distribution")
print("=" * 60)

conf_rows = []
for i in range(len(bins)-1):
    bin_min = bins[i]
    bin_max = bins[i+1]
    
    if i == len(bins) - 2:
        in_bin = (top_conf >= bin_min) & (top_conf <= bin_max)
    else:
        in_bin = (top_conf >= bin_min) & (top_conf < bin_max)
        
    correct_in_bin = (y[in_bin] == pred_class[in_bin]).sum()
    total_in_bin = in_bin.sum()
    
    conf_rows.append({
        "bin_min": bin_min,
        "bin_max": bin_max,
        "count": total_in_bin,
        "accuracy": correct_in_bin / total_in_bin if total_in_bin > 0 else np.nan
    })

conf_df = pd.DataFrame(conf_rows)
conf_df.to_csv(f"{OUTPUT_DIR}/14_confidence_distribution.csv", index=False, encoding="utf-8-sig")
print(conf_df.to_string(index=False))
print()


# ══════════════════════════════════════════════════════════════
# 15 — Error Analysis
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("15 — Error Analysis")
print("=" * 60)

errors_mask = y != pred_class
errors_df = fi[errors_mask].copy()

# Add predicted class and confidence
errors_df["predicted_outcome"] = pred_class[errors_mask]
errors_df["confidence"] = top_conf[errors_mask]

# Reorder columns to put true/pred side by side
cols = ["source_file", "civil_outcome", "predicted_outcome", "confidence", 
        "dispute_type", "plaintiff_type", "defendant_type", "claim_amount_requested"]
errors_df = errors_df[cols]

errors_df.to_csv(f"{OUTPUT_DIR}/15_error_analysis.csv", index=False, encoding="utf-8-sig")
print(f"Total misclassified cases: {len(errors_df)} ({len(errors_df)/len(fi)*100:.1f}%)")
print("Top error pairs (True -> Predicted):")
error_pairs = errors_df.groupby(["civil_outcome", "predicted_outcome"]).size().sort_values(ascending=False)
print(error_pairs.head(10).to_string())

print()
print("=" * 60)
print("Phase 3 complete. Outputs saved to:", OUTPUT_DIR)
print("=" * 60)
