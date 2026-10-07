"""
Phase 1 — Data Exploration
Computes outputs 01–06 and 16 (leakage recheck) from the analysis plan.
All computations are directly from output.csv, no model training required.

Run: python3 analysis_phase1.py
"""

import numpy as np
import pandas as pd
import warnings, os

warnings.filterwarnings("ignore")

CSV_PATH = "output.csv"
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
fi = df[(df.case_level == "first_instance") & (df.civil_outcome.notna())].copy()

print(f"Total rows in output.csv: {len(df)}")
print(f"First-instance with labeled civil_outcome: {len(fi)}")
print()

# ══════════════════════════════════════════════════════════════
# 01 — Target Distribution
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("01 — Target Distribution")
print("=" * 60)

target_dist = fi.civil_outcome.value_counts().reset_index()
target_dist.columns = ["civil_outcome", "count"]
target_dist["percentage"] = (target_dist["count"] / target_dist["count"].sum() * 100).round(1)
target_dist.to_csv(f"{OUTPUT_DIR}/01_target_distribution.csv", index=False, encoding="utf-8-sig")
print(target_dist.to_string(index=False))
print()

# ══════════════════════════════════════════════════════════════
# 02 — Dataset Summary
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("02 — Dataset Summary")
print("=" * 60)

summary_rows = []
summary_rows.append({"metric": "total_rows", "value": len(df)})
for lv in ["first_instance", "appellate", "supervisory"]:
    summary_rows.append({"metric": f"rows_{lv}", "value": (df.case_level == lv).sum()})
summary_rows.append({"metric": "first_instance_labeled", "value": len(fi)})
summary_rows.append({"metric": "appellate_supervisory_labeled",
                      "value": df[df.case_level.isin(["appellate", "supervisory"]) & df.appeal_outcome.notna()].shape[0]})
summary_rows.append({"metric": "n_columns", "value": len(df.columns)})
summary_rows.append({"metric": "n_unique_judges", "value": fi.judge.nunique()})
summary_rows.append({"metric": "n_unique_courts", "value": fi.court.nunique()})
summary_rows.append({"metric": "n_classes", "value": fi.civil_outcome.nunique()})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUTPUT_DIR}/02_dataset_summary.csv", index=False, encoding="utf-8-sig")
print(summary_df.to_string(index=False))
print()

# ══════════════════════════════════════════════════════════════
# 03 — Missing Values (first-instance only)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("03 — Missing Values (first-instance subset)")
print("=" * 60)

key_cols = ["source_file", "case_id", "decision_date", "court", "judge",
            "dispute_type", "plaintiff_type", "defendant_type",
            "claim_amount_requested", "law_articles", "case_narrative",
            "civil_outcome", "civil_award_amount"]
missing_rows = []
for c in key_cols:
    n_miss = fi[c].isna().sum()
    pct = fi[c].isna().mean() * 100
    missing_rows.append({"column": c, "n_missing": n_miss, "pct_missing": round(pct, 1),
                         "n_present": len(fi) - n_miss})
missing_df = pd.DataFrame(missing_rows)
missing_df.to_csv(f"{OUTPUT_DIR}/03_missing_values.csv", index=False, encoding="utf-8-sig")
print(missing_df.to_string(index=False))
print()

# ══════════════════════════════════════════════════════════════
# 04 — Feature-Target Crosstab (binary signal rates by outcome)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("04 — Feature-Target Crosstab (binary signal rates by outcome)")
print("=" * 60)

import features

X, median_claim, judge_counts = features.build_training_features(fi)

signal_cols = [c for c in X.columns if c.startswith("has_")]

# Build a DataFrame with outcome + signals
crosstab_data = X[signal_cols].copy()
crosstab_data["civil_outcome"] = fi["civil_outcome"].values

# Activation rate per class
rates = crosstab_data.groupby("civil_outcome")[signal_cols].mean().round(3)
rates_t = rates.T
rates_t.index.name = "signal"
rates_t["overall"] = crosstab_data[signal_cols].mean().round(3).values
rates_t.to_csv(f"{OUTPUT_DIR}/04_feature_target_crosstab.csv", encoding="utf-8-sig")
print(rates_t.to_string())
print()

# ══════════════════════════════════════════════════════════════
# 05 — Feature Distributions by Class (numeric features)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("05 — Feature Distributions by Class")
print("=" * 60)

num_cols = ["claim_amount_log", "claim_amount_missing", "judge_case_count",
            "narrative_missing", "narrative_length_log"]

dist_data = X[num_cols].copy()
dist_data["civil_outcome"] = fi["civil_outcome"].values

stats_rows = []
for outcome in sorted(fi.civil_outcome.unique()):
    subset = dist_data[dist_data.civil_outcome == outcome]
    for col in num_cols:
        stats_rows.append({
            "civil_outcome": outcome,
            "feature": col,
            "count": len(subset),
            "mean": round(subset[col].mean(), 3),
            "std": round(subset[col].std(), 3),
            "median": round(subset[col].median(), 3),
            "min": round(subset[col].min(), 3),
            "max": round(subset[col].max(), 3),
        })

dist_df = pd.DataFrame(stats_rows)
dist_df.to_csv(f"{OUTPUT_DIR}/05_feature_distributions.csv", index=False, encoding="utf-8-sig")
# Print a pivot summary
pivot = dist_df.pivot_table(index="feature", columns="civil_outcome", values="mean")
print("Mean values by class:")
print(pivot.round(3).to_string())
print()

# ══════════════════════════════════════════════════════════════
# 06 — Feature Association Matrix (Cramér's V for categoricals,
#       Phi coefficient for binary pairs)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("06 — Feature Association Matrix")
print("=" * 60)

from scipy.stats import chi2_contingency

def cramers_v(x, y):
    """Cramér's V statistic for categorical-categorical association."""
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    min_dim = min(ct.shape[0], ct.shape[1]) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))

assoc_cols = signal_cols + ["civil_outcome"]
cat_cols_for_assoc = ["plaintiff_type", "defendant_type", "court_type", "claim_amount_bucket"]
assoc_data = X[signal_cols + cat_cols_for_assoc].copy()
assoc_data["civil_outcome"] = fi["civil_outcome"].values

all_assoc_cols = signal_cols + cat_cols_for_assoc + ["civil_outcome"]

# Compute Cramér's V for all pairs vs civil_outcome
assoc_rows = []
for col in signal_cols + cat_cols_for_assoc:
    v = cramers_v(assoc_data[col], assoc_data["civil_outcome"])
    assoc_rows.append({"feature": col, "cramers_v_vs_outcome": round(v, 4)})

assoc_df = pd.DataFrame(assoc_rows).sort_values("cramers_v_vs_outcome", ascending=False)
assoc_df.to_csv(f"{OUTPUT_DIR}/06_feature_association_matrix.csv", index=False, encoding="utf-8-sig")
print("Cramér's V vs civil_outcome:")
print(assoc_df.to_string(index=False))
print()

# ══════════════════════════════════════════════════════════════
# 16 — Leakage Recheck on full dataset
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("16 — Leakage Recheck (full dataset n={})".format(len(df)))
print("=" * 60)

narratives = df["case_narrative"].dropna()
n_narratives = len(narratives)

# Check for operative section markers inside narratives
checks = {
    "ТОГТООХ нь": r"ТОГТООХ\s*нь",
    "ЗАХИРАМЖЛАХ нь": r"ЗАХИРАМЖЛАХ\s*нь",
    "ҮНДЭСЛЭХ нь": r"ҮНДЭСЛЭХ\s*нь",
    "хэрэгсэхгүй болгосугай": "хэрэгсэхгүй болгосугай",
    "хангаж": r"\bхангаж\b",
    "татгалзсугай": "татгалзсугай",
    "хэвээр үлдээ": "хэвээр үлдээ",
}

leak_rows = []
for label, pattern in checks.items():
    if pattern.startswith(r"\b") or r"\s" in pattern:
        count = narratives.str.contains(pattern, regex=True, case=False, na=False).sum()
    else:
        count = narratives.str.contains(pattern, regex=False, case=False, na=False).sum()
    pct = count / n_narratives * 100 if n_narratives > 0 else 0
    leak_rows.append({"marker": label, "count_in_narratives": count,
                      "pct_of_narratives": round(pct, 2), "n_total_narratives": n_narratives})
    status = "✅ CLEAN" if count == 0 else f"⚠️  {count} FOUND"
    print(f"  {label}: {status} ({pct:.1f}%)")

leak_df = pd.DataFrame(leak_rows)
leak_df.to_csv(f"{OUTPUT_DIR}/16_leakage_recheck.csv", index=False, encoding="utf-8-sig")
print()

# Check first-instance narratives specifically
fi_narratives = fi["case_narrative"].dropna()
n_fi_narr = len(fi_narratives)
print(f"First-instance narratives: {n_fi_narr}")
for label, pattern in [("ТОГТООХ нь", r"ТОГТООХ\s*нь"), ("ЗАХИРАМЖЛАХ нь", r"ЗАХИРАМЖЛАХ\s*нь"),
                        ("ҮНДЭСЛЭХ нь", r"ҮНДЭСЛЭХ\s*нь")]:
    count = fi_narratives.str.contains(pattern, regex=True, case=False, na=False).sum()
    status = "✅ CLEAN" if count == 0 else f"⚠️  {count} FOUND"
    print(f"  {label} (first_instance only): {status}")

print()
print("=" * 60)
print("Phase 1 complete. Outputs saved to:", OUTPUT_DIR)
print("=" * 60)
