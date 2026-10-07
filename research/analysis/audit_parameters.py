import os
import csv

OUT_DIR = "/Users/purevjavgerel/Desktop/analysis/research/analysis/output/reports"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Complete Parameter Inventory
parameters = [
    # A. DATASET & TARGET
    {"Name": "Target Variable", "Group": "Dataset", "Current Value": "civil_outcome (6 classes)", "Is Fixed/Default": "Fixed", "Notes": "The core supervised target."},
    {"Name": "Appellate Target Variable", "Group": "Dataset", "Current Value": "upheld (Binary)", "Is Fixed/Default": "Fixed", "Notes": "Target for appellate regression."},
    
    # C/D. SPLITTING & CV
    {"Name": "CV Folds (Outer)", "Group": "Cross-Validation", "Current Value": "5", "Is Fixed/Default": "Fixed", "Notes": "StratifiedKFold(n_splits=5) used for reporting."},
    {"Name": "CV Shuffle", "Group": "Cross-Validation", "Current Value": "True", "Is Fixed/Default": "Fixed", "Notes": "Ensures random distribution."},
    {"Name": "CV Random State", "Group": "Cross-Validation", "Current Value": "42", "Is Fixed/Default": "Fixed", "Notes": "Standard seed."},
    {"Name": "CalibratedClassifierCV inner folds", "Group": "Cross-Validation", "Current Value": "5 (train), 3 (eval)", "Is Fixed/Default": "Fixed", "Notes": "Inconsistent inner CV for Platt Scaling."},
    
    # E. TEXT PREPROCESSING
    {"Name": "Text Imputation", "Group": "Preprocessing", "Current Value": "Empty String ('')", "Is Fixed/Default": "Fixed", "Notes": "NaNs filled with blank strings before TF-IDF."},
    
    # F. TF-IDF
    {"Name": "TF-IDF max_features", "Group": "TF-IDF", "Current Value": "1000", "Is Fixed/Default": "Explicit", "Notes": "Caps vocabulary. High impact on text signal."},
    {"Name": "TF-IDF min_df", "Group": "TF-IDF", "Current Value": "5", "Is Fixed/Default": "Explicit", "Notes": "Removes extremely rare words."},
    {"Name": "TF-IDF ngram_range", "Group": "TF-IDF", "Current Value": "(1, 2)", "Is Fixed/Default": "Explicit", "Notes": "Unigrams and bigrams used."},
    {"Name": "TF-IDF sublinear_tf", "Group": "TF-IDF", "Current Value": "True", "Is Fixed/Default": "Explicit", "Notes": "Logarithmic term frequency scaling."},
    {"Name": "TF-IDF stop_words", "Group": "TF-IDF", "Current Value": "None", "Is Fixed/Default": "Default", "Notes": "No stop word filtering applied."},
    
    # G. CATEGORICAL
    {"Name": "Categorical Encoder", "Group": "Features", "Current Value": "OneHotEncoder", "Is Fixed/Default": "Explicit", "Notes": "One-hot applied to plaintiff_type, defendant_type, court_type, etc."},
    {"Name": "OHE handle_unknown", "Group": "Features", "Current Value": "'ignore'", "Is Fixed/Default": "Explicit", "Notes": "Prevents crashes on unseen categories."},
    {"Name": "Appellate Encoder", "Group": "Features", "Current Value": "pd.get_dummies", "Is Fixed/Default": "Explicit", "Notes": "Deviates from doc claiming STE M=5."},
    
    # H. NUMERICAL
    {"Name": "Numerical Imputation (Claim)", "Group": "Features", "Current Value": "Median Claim", "Is Fixed/Default": "Explicit", "Notes": "Replaces missing claims with median."},
    {"Name": "Numerical Transformation", "Group": "Features", "Current Value": "np.log1p", "Is Fixed/Default": "Explicit", "Notes": "Applied to claim amounts and text lengths."},
    {"Name": "Claim Amount Bucket", "Group": "Features", "Current Value": "[-inf, 1M, 10M, inf]", "Is Fixed/Default": "Explicit", "Notes": "Discretizes claim amounts into low/medium/high."},
    {"Name": "Appellate Scaling", "Group": "Features", "Current Value": "StandardScaler(with_mean=False)", "Is Fixed/Default": "Explicit", "Notes": "Applied to dummy variables! Skews Odds Ratios."},
    
    # I. SIGNALS
    {"Name": "Regex Signal count", "Group": "Features", "Current Value": "13", "Is Fixed/Default": "Explicit", "Notes": "Hand-engineered boolean features."},
    
    # J. RANDOM FOREST
    {"Name": "RF n_estimators", "Group": "Random Forest", "Current Value": "300", "Is Fixed/Default": "Explicit", "Notes": "Only used for comparison in phase 2."},
    {"Name": "RF class_weight", "Group": "Random Forest", "Current Value": "'balanced'", "Is Fixed/Default": "Explicit", "Notes": "Adjusts for class imbalance."},
    
    # K. LOGISTIC REGRESSION
    {"Name": "LR max_iter", "Group": "Logistic Regression", "Current Value": "2000", "Is Fixed/Default": "Explicit", "Notes": "High max_iter to ensure convergence."},
    {"Name": "LR class_weight", "Group": "Logistic Regression", "Current Value": "'balanced'", "Is Fixed/Default": "Explicit", "Notes": "Handles imbalance."},
    {"Name": "LR penalty", "Group": "Logistic Regression", "Current Value": "l2", "Is Fixed/Default": "Default", "Notes": "Default L2 regularization."},
    {"Name": "LR C", "Group": "Logistic Regression", "Current Value": "1.0", "Is Fixed/Default": "Default", "Notes": "Default regularization strength."},
    
    # L. HISTGRADIENTBOOSTING
    {"Name": "HistGBM max_iter", "Group": "HistGradientBoosting", "Current Value": "500", "Is Fixed/Default": "Explicit", "Notes": "Maximum number of trees."},
    {"Name": "HistGBM learning_rate", "Group": "HistGradientBoosting", "Current Value": "0.05", "Is Fixed/Default": "Explicit", "Notes": "Shrinkage parameter."},
    {"Name": "HistGBM max_depth", "Group": "HistGradientBoosting", "Current Value": "6", "Is Fixed/Default": "Explicit", "Notes": "Limits tree depth."},
    {"Name": "HistGBM class_weight", "Group": "HistGradientBoosting", "Current Value": "'balanced'", "Is Fixed/Default": "Explicit", "Notes": "Crucial for predicting minority classes."},
    {"Name": "HistGBM l2_regularization", "Group": "HistGradientBoosting", "Current Value": "0.0", "Is Fixed/Default": "Default", "Notes": "No L2 penalty applied explicitly."},
    {"Name": "HistGBM min_samples_leaf", "Group": "HistGradientBoosting", "Current Value": "20", "Is Fixed/Default": "Default", "Notes": "Default leaf minimum."},
    
    # N. CALIBRATION
    {"Name": "Calibration Method", "Group": "Calibration", "Current Value": "'sigmoid'", "Is Fixed/Default": "Explicit", "Notes": "Platt Scaling used instead of isotonic."},
    
    # O. PREDICTION / CONFIDENCE
    {"Name": "Precision Thresholds", "Group": "Prediction", "Current Value": "denied=0.5, partial=0.4, etc.", "Is Fixed/Default": "Explicit", "Notes": "Hardcoded thresholds for final prediction assignment."},
    {"Name": "Confidence Margin Threshold", "Group": "Prediction", "Current Value": "0.20", "Is Fixed/Default": "Explicit", "Notes": "Margin between top 2 classes for high confidence."},
    {"Name": "Signal Boosts", "Group": "Prediction", "Current Value": "settled=1.5, granted=1.3, etc.", "Is Fixed/Default": "Explicit", "Notes": "Heuristic posterior probability multipliers applied after model prediction!"},
    
    # Q. EVALUATION
    {"Name": "Primary Metric", "Group": "Evaluation", "Current Value": "macro-F1", "Is Fixed/Default": "Fixed", "Notes": "Main evaluation metric."},
]

def write_csv(filename, fieldnames, data):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

write_csv("complete_parameter_inventory.csv", ["Name", "Group", "Current Value", "Is Fixed/Default", "Notes"], parameters)

experiment = [p for p in parameters if p["Group"] in ("TF-IDF", "HistGradientBoosting", "Logistic Regression", "Calibration", "Prediction", "Features") and "Fixed" not in p["Is Fixed/Default"]]
write_csv("experiment_worthy_parameters.csv", ["Name", "Group", "Current Value", "Is Fixed/Default", "Notes"], experiment)

interactions = [
    {"Parameter 1": "TF-IDF max_features", "Parameter 2": "HistGBM max_depth", "Interaction": "Higher vocab size requires deeper trees to capture interactions."},
    {"Parameter 1": "HistGBM learning_rate", "Parameter 2": "HistGBM max_iter", "Interaction": "Lower learning rate requires higher max_iter to converge."},
    {"Parameter 1": "Appellate Scaling", "Parameter 2": "Logistic Regression Penalty", "Interaction": "StandardScaler shrinks dummy vars, drastically altering the L2 penalty effect and Odds Ratios."},
    {"Parameter 1": "Calibration Method", "Parameter 2": "Precision Thresholds", "Interaction": "Changing calibration from sigmoid to isotonic will invalidate all hardcoded precision thresholds."},
    {"Parameter 1": "TF-IDF ngram_range", "Parameter 2": "TF-IDF min_df", "Interaction": "Adding bigrams (1,2) drastically increases vocab; min_df must be tuned together to prevent noise."},
]
write_csv("parameter_interactions.csv", ["Parameter 1", "Parameter 2", "Interaction"], interactions)

controls = [p for p in parameters if p["Is Fixed/Default"] == "Fixed"]
write_csv("fixed_experiment_controls.csv", ["Name", "Group", "Current Value", "Is Fixed/Default", "Notes"], controls)

def to_markdown_table(data, columns):
    if not data:
        return ""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in data:
        rows.append("| " + " | ".join([str(row.get(col, "")) for col in columns]) + " |")
    return "\n".join([header, separator] + rows)

md_lines = ["# MASTER PARAMETER AUDIT TABLE", "", "## Complete Parameter Inventory", ""]
md_lines.append(to_markdown_table(parameters, ["Name", "Group", "Current Value", "Is Fixed/Default", "Notes"]))
md_lines.extend(["", "## High-Impact Experiment Targets", "These parameters strongly govern the model's behavior and represent the primary search space for tuning."])
md_lines.append(to_markdown_table(experiment, ["Name", "Group", "Current Value", "Is Fixed/Default", "Notes"]))
md_lines.extend(["", "## Known Methodological Discrepancies & Oddities", 
    "- **Appellate StandardScaler on Dummies**: The `StandardScaler(with_mean=False)` is applied to the One-Hot Encoded variables. This scales binary indicators by their variance, drastically distorting the resulting Logistic Regression coefficients and Odds Ratios.",
    "- **Appellate Target Encoding**: Documentation states `Smoothed Target Encoding M=5`, but implementation uses `pd.get_dummies`.",
    "- **Heuristic Signal Boosts**: The `prediction_model.py` script applies arbitrary heuristic multipliers (e.g., `has_settlement` * 1.5) to the `CalibratedClassifierCV` posterior probabilities *after* prediction, destroying calibration.",
    "- **Hardcoded Precision Thresholds**: Final classifications are overridden by arbitrary probability thresholds in `predict_case`."
])

with open(os.path.join(OUT_DIR, "MASTER_PARAMETER_TABLE.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print("Audit complete. Saved 5 artifacts to:", OUT_DIR)
