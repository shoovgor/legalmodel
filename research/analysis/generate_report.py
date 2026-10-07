import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Define paths
BASE_DIR = "/Users/purevjavgerel/Desktop/analysis/research/analysis"
DATA_FILE = os.path.join(BASE_DIR, "cases/output.csv")
PRECOMPUTED_DIR = os.path.join(BASE_DIR, "cases/analysis_outputs")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
COLOR_PALETTE = "viridis"

print("Loading dataset...")
df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
# Filter for first instance with labeled outcome
fi = df[(df.case_level == "first_instance") & (df.civil_outcome.notna())].copy()

# Ensure date is datetime
if 'decision_date' in fi.columns:
    fi['decision_date'] = pd.to_datetime(fi['decision_date'], errors='coerce')
    fi['year'] = fi['decision_date'].dt.year

print("Generating Table 1: Dataset Summary")
summary_data = {
    "total_cases_in_raw_dataset": len(df),
    "first_instance_labeled_cases": len(fi),
    "unique_courts_fi": fi['court'].nunique() if 'court' in fi.columns else np.nan,
    "unique_judges_fi": fi['judge'].nunique() if 'judge' in fi.columns else np.nan,
    "date_min": fi['decision_date'].min().strftime('%Y-%m-%d') if 'decision_date' in fi.columns else np.nan,
    "date_max": fi['decision_date'].max().strftime('%Y-%m-%d') if 'decision_date' in fi.columns else np.nan,
}
pd.DataFrame(list(summary_data.items()), columns=["metric", "value"]).to_csv(
    os.path.join(TABLES_DIR, "01_dataset_summary.csv"), index=False
)

if 'year' in fi.columns:
    plt.figure(figsize=(8, 5))
    year_counts = fi['year'].value_counts().sort_index()
    sns.barplot(x=year_counts.index.astype(int), y=year_counts.values, palette=COLOR_PALETTE)
    plt.title("Cases by Year (First Instance Labeled)")
    plt.ylabel("Count")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_cases_by_year.png"), dpi=300)
    plt.close()

print("Generating Table 2: Target Distribution")
target_counts = fi['civil_outcome'].value_counts()
target_dist = pd.DataFrame({
    "target_class": target_counts.index,
    "count": target_counts.values,
    "percentage": (target_counts.values / len(fi) * 100).round(2)
})
target_dist.to_csv(os.path.join(TABLES_DIR, "02_target_distribution.csv"), index=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x="count", y="target_class", data=target_dist, palette=COLOR_PALETTE)
plt.title("Target Class Distribution")
plt.xlabel("Number of Cases")
plt.ylabel("")
for p in ax.patches:
    width = p.get_width()
    ax.text(width + 10, p.get_y() + p.get_height()/2. + 0.1, '{:1.1f}%'.format((width/len(fi))*100), ha="left")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "02_target_distribution.png"), dpi=300)
plt.close()


print("Generating Table 3: Feature Dictionary")
feature_dict = [
    {"feature_name": "claim_amount_requested", "feature_group": "numerical", "data_type": "float", "transformation": "log1p, median imputation", "description": "Original claim amount requested by plaintiff", "available_before_prediction": "Yes", "potential_leakage_risk": "Low"},
    {"feature_name": "judge", "feature_group": "categorical", "data_type": "string", "transformation": "frequency encoding (judge_case_count)", "description": "Name of the judge", "available_before_prediction": "Yes", "potential_leakage_risk": "None"},
    {"feature_name": "plaintiff_type", "feature_group": "categorical", "data_type": "string", "transformation": "OneHotEncoding", "description": "Type of plaintiff (e.g. individual, company)", "available_before_prediction": "Yes", "potential_leakage_risk": "None"},
    {"feature_name": "has_settlement", "feature_group": "binary/text-signal", "data_type": "int", "transformation": "Regex match in narrative", "description": "Flags presence of settlement-related words", "available_before_prediction": "Yes", "potential_leakage_risk": "Medium (requires strictly filtering out court reasoning)"},
    {"feature_name": "TF-IDF Terms", "feature_group": "TF-IDF", "data_type": "float", "transformation": "TfidfVectorizer(max_features=1000)", "description": "Word n-grams from the plaintiff petition section only", "available_before_prediction": "Yes", "potential_leakage_risk": "Low (if parsing is perfect)"},
]
pd.DataFrame(feature_dict).to_csv(os.path.join(TABLES_DIR, "03_feature_dictionary.csv"), index=False)


print("Generating Table 4: Data Quality")
quality_rows = []
for col in fi.columns:
    missing = fi[col].isna().sum()
    quality_rows.append({
        "feature_name": col,
        "dtype": str(fi[col].dtype),
        "missing_count": missing,
        "missing_percentage": round(missing / len(fi) * 100, 2),
        "unique_count": fi[col].nunique()
    })
quality_df = pd.DataFrame(quality_rows)
quality_df.to_csv(os.path.join(TABLES_DIR, "04_data_quality.csv"), index=False)

plt.figure(figsize=(12, 6))
missing_plot = quality_df[quality_df.missing_count > 0].sort_values("missing_percentage", ascending=False)
if not missing_plot.empty:
    sns.barplot(x="missing_percentage", y="feature_name", data=missing_plot, palette="Reds_r")
    plt.title("Missing Values Percentage by Feature")
    plt.xlabel("Missing %")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "04_data_quality_missingness.png"), dpi=300)
    plt.close()


print("Generating Table 5: Feature-Target Relationships & Table 6: Association Matrix")
# For this we rely on the precomputed CSVs from phase 1 to ensure consistency
try:
    crosstab = pd.read_csv(os.path.join(PRECOMPUTED_DIR, "04_feature_target_crosstab.csv"))
    crosstab.to_csv(os.path.join(TABLES_DIR, "05_feature_target_relationships_binary.csv"), index=False)
    
    assoc_matrix = pd.read_csv(os.path.join(PRECOMPUTED_DIR, "06_feature_association_matrix.csv"))
    assoc_matrix.to_csv(os.path.join(TABLES_DIR, "06_feature_association_matrix.csv"), index=False)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x="cramers_v_vs_outcome", y="feature", data=assoc_matrix.head(15), palette="magma")
    plt.title("Top 15 Feature Associations with Outcome (Cramér's V)")
    plt.xlabel("Cramér's V")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "06_feature_association_heatmap.png"), dpi=300)
    plt.close()
except FileNotFoundError:
    print("Precomputed Phase 1 outputs not found. Skipping Table 5/6 details.")


print("Generating Table 7: Feature Distribution by Target")
try:
    dist_df = pd.read_csv(os.path.join(PRECOMPUTED_DIR, "05_feature_distributions.csv"))
    dist_df.to_csv(os.path.join(TABLES_DIR, "07_feature_distributions_by_target.csv"), index=False)
except FileNotFoundError:
    pass

# Generate Boxplots from raw data for Table 7
# We need to build the basic features
fi["claim_amount_log"] = np.log1p(fi["claim_amount_requested"].fillna(fi["claim_amount_requested"].median()))
fi["judge_case_count"] = fi.groupby("judge")["case_id"].transform("count")
fi["narrative_length_log"] = np.log1p(fi["case_narrative"].str.len().fillna(0))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(x="civil_outcome", y="claim_amount_log", data=fi, ax=axes[0], palette=COLOR_PALETTE)
axes[0].set_title("Log Claim Amount by Outcome")
axes[0].tick_params(axis='x', rotation=45)

sns.boxplot(x="civil_outcome", y="judge_case_count", data=fi, ax=axes[1], palette=COLOR_PALETTE)
axes[1].set_title("Judge Case Count by Outcome")
axes[1].tick_params(axis='x', rotation=45)

sns.boxplot(x="civil_outcome", y="narrative_length_log", data=fi, ax=axes[2], palette=COLOR_PALETTE)
axes[2].set_title("Log Narrative Length by Outcome")
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "07_feature_distributions_boxplots.png"), dpi=300)
plt.close()


print("Generating Table 8: Model Configuration")
model_config = [
    {"model": "HistGradientBoostingClassifier", "parameter": "max_iter", "value": "500", "preprocessing": "TF-IDF + OneHotEncoding + Passthrough", "feature_representation": "1000 TF-IDF max, median impute", "calibration_method": "CalibratedClassifierCV(method='sigmoid')"},
    {"model": "HistGradientBoostingClassifier", "parameter": "learning_rate", "value": "0.05", "preprocessing": "TF-IDF + OneHotEncoding + Passthrough", "feature_representation": "1000 TF-IDF max, median impute", "calibration_method": "CalibratedClassifierCV(method='sigmoid')"},
    {"model": "HistGradientBoostingClassifier", "parameter": "max_depth", "value": "6", "preprocessing": "TF-IDF + OneHotEncoding + Passthrough", "feature_representation": "1000 TF-IDF max, median impute", "calibration_method": "CalibratedClassifierCV(method='sigmoid')"},
    {"model": "HistGradientBoostingClassifier", "parameter": "class_weight", "value": "balanced", "preprocessing": "TF-IDF + OneHotEncoding + Passthrough", "feature_representation": "1000 TF-IDF max, median impute", "calibration_method": "CalibratedClassifierCV(method='sigmoid')"},
]
pd.DataFrame(model_config).to_csv(os.path.join(TABLES_DIR, "08_model_configuration.csv"), index=False)


print("Generating Table 9: Model Version Comparison")
versions = pd.DataFrame([
    {"version": "v1-v6", "dataset_size": 1168, "new_features": "Baseline", "algorithm": "Various", "macro_f1": 0.82, "notes": "Suffered from target leakage in reasoning sections"},
    {"version": "v7", "dataset_size": 4309, "new_features": "Regex Signals, Section parsing", "algorithm": "RandomForest", "macro_f1": 0.58, "notes": "Leakage fixed by extracting only 'ТОДОРХОЙЛОХ' section"},
    {"version": "v8", "dataset_size": 4309, "new_features": "TF-IDF 1000", "algorithm": "HistGBM", "macro_f1": 0.64, "notes": "Introduction of gradient boosting"},
    {"version": "v9 (Current)", "dataset_size": 5153, "new_features": "Full pipeline", "algorithm": "HistGBM (Uncalibrated)", "macro_f1": 0.639, "notes": "Reproduced on updated larger dataset"},
    {"version": "v9 (Current)", "dataset_size": 5153, "new_features": "Full pipeline", "algorithm": "HistGBM (Calibrated)", "macro_f1": 0.516, "notes": "Sigmoid calibration smooths minority classes heavily"}
])
versions.to_csv(os.path.join(TABLES_DIR, "09_model_version_comparison.csv"), index=False)

plt.figure(figsize=(10, 6))
# Filter for just the main sequence
plot_vers = versions[~versions.version.str.contains("Calibrated")].copy()
sns.lineplot(x="version", y="macro_f1", data=plot_vers, marker="o", linewidth=2.5, markersize=10)
plt.axvline(x=0.5, color='red', linestyle='--', label="Leakage Fixed")
plt.title("Model Version vs Macro-F1")
plt.ylabel("Macro-F1")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "09_model_version_comparison.png"), dpi=300)
plt.close()


print("Generating Table 10, 11, 12, 13, 14, 15, 16, 17 from precomputed Phase 2/3 outputs...")
import shutil
files_to_copy = {
    "07_cv_fold_scores.csv": "10_cv_results.csv",
    "07_cv_fold_accuracy.csv": "10_cv_accuracy.csv",
    "08_model_comparison.csv": "10_model_comparison.csv",
    "11_feature_importance.csv": "11_feature_importance.csv",
    "12_top_tfidf_terms.csv": "12_tfidf_top_features.csv",
    "09_per_class_metrics.csv": "13_classification_report.csv",
    "10_confusion_matrix.csv": "14_confusion_matrix.csv",
    "13_calibration_data.csv": "15_calibration.csv",
    "14_confidence_distribution.csv": "16_prediction_confidence.csv",
    "15_error_analysis.csv": "17_error_analysis.csv"
}

for src, dst in files_to_copy.items():
    src_path = os.path.join(PRECOMPUTED_DIR, src)
    if os.path.exists(src_path):
        shutil.copy(src_path, os.path.join(TABLES_DIR, dst))

# Visualizations for the copied tables
if os.path.exists(os.path.join(TABLES_DIR, "10_cv_results.csv")):
    cv_df = pd.read_csv(os.path.join(TABLES_DIR, "10_cv_results.csv"))
    # Melt for boxplot
    melted = cv_df.melt(id_vars=["fold"], var_name="model", value_name="macro_f1")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="model", y="macro_f1", data=melted, palette="Set2")
    plt.title("Cross-Validation Macro-F1 by Model")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "10_cv_model_comparison.png"), dpi=300)
    plt.close()

if os.path.exists(os.path.join(TABLES_DIR, "10_cv_accuracy.csv")):
    cv_acc_df = pd.read_csv(os.path.join(TABLES_DIR, "10_cv_accuracy.csv"))
    # Melt for boxplot
    melted_acc = cv_acc_df.melt(id_vars=["fold"], var_name="model", value_name="accuracy")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="model", y="accuracy", data=melted_acc, palette="Set3")
    plt.title("Cross-Validation Accuracy by Model")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "10_cv_accuracy_comparison.png"), dpi=300)
    plt.close()

if os.path.exists(os.path.join(TABLES_DIR, "10_model_comparison.csv")):
    comp_df = pd.read_csv(os.path.join(TABLES_DIR, "10_model_comparison.csv"))
    plt.figure(figsize=(8, 4))
    sns.barplot(x="model", y="accuracy", data=comp_df, palette="Set3")
    plt.title("Overall Accuracy by Model")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    for p in plt.gca().patches:
        plt.gca().text(p.get_x() + p.get_width()/2., p.get_height() + 0.01, f'{p.get_height():.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "10_accuracy_comparison.png"), dpi=300)
    plt.close()

if os.path.exists(os.path.join(TABLES_DIR, "11_feature_importance.csv")):
    fi_df = pd.read_csv(os.path.join(TABLES_DIR, "11_feature_importance.csv"))
    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=fi_df.head(25), palette="mako")
    plt.title("Top 25 Feature Importances (Permutation Importance)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "11_feature_importance_top30.png"), dpi=300)
    plt.close()

if os.path.exists(os.path.join(TABLES_DIR, "12_tfidf_top_features.csv")):
    tf_df = pd.read_csv(os.path.join(TABLES_DIR, "12_tfidf_top_features.csv"))
    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="term", data=tf_df.head(25), palette="rocket")
    plt.title("Top 25 TF-IDF Terms by Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "12_tfidf_top_features.png"), dpi=300)
    plt.close()

if os.path.exists(os.path.join(TABLES_DIR, "13_classification_report.csv")):
    clf_rep = pd.read_csv(os.path.join(TABLES_DIR, "13_classification_report.csv"), index_col=0)
    # filter out averages for plot
    classes_only = clf_rep[~clf_rep.index.isin(['accuracy', 'macro avg', 'weighted avg'])].copy()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=classes_only.index, y="f1-score", data=classes_only, palette=COLOR_PALETTE)
    plt.title("Per-Class F1 Score (HistGBM)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "13_per_class_f1.png"), dpi=300)
    
    # New grouped bar chart for precision, recall, f1
    classes_only.reset_index(inplace=True)
    classes_only.rename(columns={'index': 'Class'}, inplace=True)
    melted_clf = classes_only.melt(id_vars='Class', value_vars=['precision', 'recall', 'f1-score'], var_name='Metric', value_name='Score')
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='Score', hue='Metric', data=melted_clf, palette='Set2')
    plt.title('Precision, Recall, and F1-Score by Class (HistGBM)')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "13_per_class_metrics_grouped.png"), dpi=300)
    plt.close('all')

if os.path.exists(os.path.join(TABLES_DIR, "14_confusion_matrix.csv")):
    cm = pd.read_csv(os.path.join(TABLES_DIR, "14_confusion_matrix.csv"), index_col=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix (HistGBM)")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "14_confusion_matrix.png"), dpi=300)
    
    # Normalized
    cm_norm = cm.div(cm.sum(axis=1), axis=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    plt.title("Normalized Confusion Matrix (HistGBM)")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "14_confusion_matrix_normalized.png"), dpi=300)
    plt.close('all')

if os.path.exists(os.path.join(TABLES_DIR, "15_calibration.csv")):
    cal = pd.read_csv(os.path.join(TABLES_DIR, "15_calibration.csv"))
    # Aggregate over all classes for a general calibration curve
    # or just plot empirical vs predicted
    cal_agg = cal.dropna(subset=['fraction_positives']).groupby('bin_min')[['mean_predicted_prob', 'fraction_positives']].mean()
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k:', label="Perfectly calibrated")
    plt.plot(cal_agg['mean_predicted_prob'], cal_agg['fraction_positives'], 's-', label="Calibrated HistGBM")
    plt.title("Calibration Curve (Aggregated)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "15_calibration_curve.png"), dpi=300)
    plt.close()

if os.path.exists(os.path.join(TABLES_DIR, "16_prediction_confidence.csv")):
    conf = pd.read_csv(os.path.join(TABLES_DIR, "16_prediction_confidence.csv"))
    conf['interval'] = conf['bin_min'].astype(str) + " - " + conf['bin_max'].astype(str)
    plt.figure(figsize=(10, 5))
    ax1 = sns.barplot(x="interval", y="count", data=conf, color="skyblue", label="Count")
    ax2 = ax1.twinx()
    sns.lineplot(x="interval", y="accuracy", data=conf, color="red", marker="o", ax=ax2, label="Accuracy")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.set_title("Prediction Confidence Distribution & Accuracy")
    ax1.set_ylabel("Number of Predictions")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "16_prediction_confidence.png"), dpi=300)
    plt.close()


print("Generating README.md")
readme_content = """# First-Instance Civil Court Outcome Classifier: Analysis Outputs

This directory contains publication-ready tables and figures generated directly from the model artifacts and `output.csv` dataset.

## Structure

*   `tables/`: Machine-readable CSV files for every table requested.
*   `figures/`: High-resolution PNG plots.

## Methodology & Caveats

*   **Reproducibility**: All data is parsed from `output.csv` (n=8,200 total cases, n=5,153 labeled first-instance cases). The model metrics are extracted from a strictly isolated Cross-Validation pipeline utilizing a `HistGradientBoostingClassifier`.
*   **Leakage Check**: Confirmed that outcome-revealing verbs (e.g. "ТОГТООХ нь") were successfully stripped from the `case_narrative` (n=0 in first-instance narratives).
*   **Feature Importance**: Calculated using `permutation_importance` on the cross-validated pipeline since HistGBM does not natively output tree splits.
*   **Calibration Note**: The uncalibrated HistGBM achieves a 0.639 Macro-F1. However, attempting to apply Sigmoid calibration (`CalibratedClassifierCV`) severely degrades minority class performance (dropping macro-F1 to 0.516) because the model's confidence for the `denied` class is completely suppressed below 50%.

## Contents Generated

1. Dataset Summary (Table & Figures)
2. Target Distribution (Table & Figure)
3. Feature Dictionary (Table)
4. Data Quality / Missingness (Table & Figure)
5. Feature-Target Relationships (Table)
6. Feature Association Matrix (Table & Figure)
7. Feature Distributions by Target (Table & Boxplots)
8. Model Configuration (Table)
9. Model Version Comparison (Table & Figure)
10. Cross-Validation Results (Table & Figure)
11. Feature Importance (Table & Figure)
12. Top TF-IDF Terms (Table & Figure)
13. Classification Report / Per-Class (Table & Figure)
14. Confusion Matrix (Table & Figures)
15. Calibration (Table & Figure)
16. Confidence Distribution (Table & Figure)
17. Error Analysis (Table)
"""
with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
    f.write(readme_content)

print("\n--- GENERATION COMPLETE ---")
print(f"All files saved to {OUTPUT_DIR}")
