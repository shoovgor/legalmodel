"""
Phase 4 — Appellate Regression
Computes outputs 17–18 from the analysis plan.
"""

import numpy as np
import pandas as pd
import warnings, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

warnings.filterwarnings("ignore")

CSV_PATH = "output.csv"
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# ══════════════════════════════════════════════════════════════
# 17 — Appellate Regression Results
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("17 — Appellate Regression Results")
print("=" * 60)

# Filter for appellate and supervisory cases with labeled outcome
app = df[df.case_level.isin(["appellate", "supervisory"]) & df.appeal_outcome.notna()].copy()
print(f"Working with {len(app)} appellate/supervisory cases")

# Binary target: upheld=1, overturned/modified/dismissed=0
app["target_upheld"] = (app.appeal_outcome == "upheld").astype(int)

# Feature engineering
app["claim_amount_log"] = np.log1p(app.claim_amount_requested.fillna(app.claim_amount_requested.median()))
app["claim_amount_missing"] = app.claim_amount_requested.isna().astype(int)

# Count referenced decisions
def count_refs(x):
    if pd.isna(x): return 0
    return len(x.split("|"))
app["referenced_decision_count"] = app.referenced_decisions.apply(count_refs)

# Dummies
features_to_dummy = ["plaintiff_type", "defendant_type", "case_level", "appealing_party"]
app[features_to_dummy] = app[features_to_dummy].fillna("unknown")

X_reg = app[["claim_amount_log", "claim_amount_missing", "referenced_decision_count"]]
for col in features_to_dummy:
    dummies = pd.get_dummies(app[col], prefix=col, drop_first=True)
    X_reg = pd.concat([X_reg, dummies], axis=1)

# Ensure numeric boolean columns are float for statsmodels
X_reg = X_reg.astype(float)
y_reg = app["target_upheld"].astype(float)

# CV Accuracy using scikit-learn
lr = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', random_state=42, max_iter=1000)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reg)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(lr, X_scaled, y_reg, cv=cv, scoring='accuracy')

print(f"Logistic Regression CV Accuracy (balanced): {scores.mean():.3f} ± {scores.std():.3f}")

# Statsmodels for coefficients, p-values, odds ratios
X_sm = sm.add_constant(X_reg)
sm_model = sm.Logit(y_reg, X_sm).fit(disp=0)

results_df = pd.DataFrame({
    "coefficient": sm_model.params,
    "std_error": sm_model.bse,
    "p_value": sm_model.pvalues,
    "odds_ratio": np.exp(sm_model.params)
})
results_df.index.name = "feature"

results_df.to_csv(f"{OUTPUT_DIR}/17_appellate_regression_results.csv", encoding="utf-8-sig")
print("\nRegression Coefficients & Odds Ratios:")
print(results_df.round(4).to_string())
print()


# ══════════════════════════════════════════════════════════════
# 18 — Appellate Smoothed Rates
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("18 — Appellate Smoothed Rates")
print("=" * 60)

rates_rows = []

for group_col in ["appealing_party", "case_level", "plaintiff_type", "defendant_type"]:
    grouped = app.groupby(group_col)["target_upheld"].agg(["count", "mean"])
    for val, row in grouped.iterrows():
        # Apply smoothing (Bayesian average)
        # prior mean = overall mean, prior weight = 10 cases
        global_mean = y_reg.mean()
        prior_weight = 10
        
        n = row["count"]
        emp_mean = row["mean"]
        
        smoothed_mean = (n * emp_mean + prior_weight * global_mean) / (n + prior_weight)
        
        rates_rows.append({
            "grouping": group_col,
            "value": val,
            "count": int(n),
            "raw_upheld_rate": emp_mean,
            "smoothed_upheld_rate": smoothed_mean
        })

rates_df = pd.DataFrame(rates_rows)
rates_df.to_csv(f"{OUTPUT_DIR}/18_appellate_smoothed_rates.csv", index=False, encoding="utf-8-sig")
print(rates_df.round(3).to_string(index=False))

print()
print("=" * 60)
print("Phase 4 complete. Outputs saved to:", OUTPUT_DIR)
print("=" * 60)
