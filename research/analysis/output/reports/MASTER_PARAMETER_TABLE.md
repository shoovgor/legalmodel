# MASTER PARAMETER AUDIT TABLE

## Complete Parameter Inventory

| Name | Group | Current Value | Is Fixed/Default | Notes |
| --- | --- | --- | --- | --- |
| Target Variable | Dataset | civil_outcome (6 classes) | Fixed | The core supervised target. |
| Appellate Target Variable | Dataset | upheld (Binary) | Fixed | Target for appellate regression. |
| CV Folds (Outer) | Cross-Validation | 5 | Fixed | StratifiedKFold(n_splits=5) used for reporting. |
| CV Shuffle | Cross-Validation | True | Fixed | Ensures random distribution. |
| CV Random State | Cross-Validation | 42 | Fixed | Standard seed. |
| CalibratedClassifierCV inner folds | Cross-Validation | 5 (train), 3 (eval) | Fixed | Inconsistent inner CV for Platt Scaling. |
| Text Imputation | Preprocessing | Empty String ('') | Fixed | NaNs filled with blank strings before TF-IDF. |
| TF-IDF max_features | TF-IDF | 1000 | Explicit | Caps vocabulary. High impact on text signal. |
| TF-IDF min_df | TF-IDF | 5 | Explicit | Removes extremely rare words. |
| TF-IDF ngram_range | TF-IDF | (1, 2) | Explicit | Unigrams and bigrams used. |
| TF-IDF sublinear_tf | TF-IDF | True | Explicit | Logarithmic term frequency scaling. |
| TF-IDF stop_words | TF-IDF | None | Default | No stop word filtering applied. |
| Categorical Encoder | Features | OneHotEncoder | Explicit | One-hot applied to plaintiff_type, defendant_type, court_type, etc. |
| OHE handle_unknown | Features | 'ignore' | Explicit | Prevents crashes on unseen categories. |
| Appellate Encoder | Features | pd.get_dummies | Explicit | Deviates from doc claiming STE M=5. |
| Numerical Imputation (Claim) | Features | Median Claim | Explicit | Replaces missing claims with median. |
| Numerical Transformation | Features | np.log1p | Explicit | Applied to claim amounts and text lengths. |
| Claim Amount Bucket | Features | [-inf, 1M, 10M, inf] | Explicit | Discretizes claim amounts into low/medium/high. |
| Appellate Scaling | Features | StandardScaler(with_mean=False) | Explicit | Applied to dummy variables! Skews Odds Ratios. |
| Regex Signal count | Features | 13 | Explicit | Hand-engineered boolean features. |
| RF n_estimators | Random Forest | 300 | Explicit | Only used for comparison in phase 2. |
| RF class_weight | Random Forest | 'balanced' | Explicit | Adjusts for class imbalance. |
| LR max_iter | Logistic Regression | 2000 | Explicit | High max_iter to ensure convergence. |
| LR class_weight | Logistic Regression | 'balanced' | Explicit | Handles imbalance. |
| LR penalty | Logistic Regression | l2 | Default | Default L2 regularization. |
| LR C | Logistic Regression | 1.0 | Default | Default regularization strength. |
| HistGBM max_iter | HistGradientBoosting | 500 | Explicit | Maximum number of trees. |
| HistGBM learning_rate | HistGradientBoosting | 0.05 | Explicit | Shrinkage parameter. |
| HistGBM max_depth | HistGradientBoosting | 6 | Explicit | Limits tree depth. |
| HistGBM class_weight | HistGradientBoosting | 'balanced' | Explicit | Crucial for predicting minority classes. |
| HistGBM l2_regularization | HistGradientBoosting | 0.0 | Default | No L2 penalty applied explicitly. |
| HistGBM min_samples_leaf | HistGradientBoosting | 20 | Default | Default leaf minimum. |
| Calibration Method | Calibration | 'sigmoid' | Explicit | Platt Scaling used instead of isotonic. |
| Precision Thresholds | Prediction | denied=0.5, partial=0.4, etc. | Explicit | Hardcoded thresholds for final prediction assignment. |
| Confidence Margin Threshold | Prediction | 0.20 | Explicit | Margin between top 2 classes for high confidence. |
| Signal Boosts | Prediction | settled=1.5, granted=1.3, etc. | Explicit | Heuristic posterior probability multipliers applied after model prediction! |
| Primary Metric | Evaluation | macro-F1 | Fixed | Main evaluation metric. |

## High-Impact Experiment Targets
These parameters strongly govern the model's behavior and represent the primary search space for tuning.
| Name | Group | Current Value | Is Fixed/Default | Notes |
| --- | --- | --- | --- | --- |
| TF-IDF max_features | TF-IDF | 1000 | Explicit | Caps vocabulary. High impact on text signal. |
| TF-IDF min_df | TF-IDF | 5 | Explicit | Removes extremely rare words. |
| TF-IDF ngram_range | TF-IDF | (1, 2) | Explicit | Unigrams and bigrams used. |
| TF-IDF sublinear_tf | TF-IDF | True | Explicit | Logarithmic term frequency scaling. |
| TF-IDF stop_words | TF-IDF | None | Default | No stop word filtering applied. |
| Categorical Encoder | Features | OneHotEncoder | Explicit | One-hot applied to plaintiff_type, defendant_type, court_type, etc. |
| OHE handle_unknown | Features | 'ignore' | Explicit | Prevents crashes on unseen categories. |
| Appellate Encoder | Features | pd.get_dummies | Explicit | Deviates from doc claiming STE M=5. |
| Numerical Imputation (Claim) | Features | Median Claim | Explicit | Replaces missing claims with median. |
| Numerical Transformation | Features | np.log1p | Explicit | Applied to claim amounts and text lengths. |
| Claim Amount Bucket | Features | [-inf, 1M, 10M, inf] | Explicit | Discretizes claim amounts into low/medium/high. |
| Appellate Scaling | Features | StandardScaler(with_mean=False) | Explicit | Applied to dummy variables! Skews Odds Ratios. |
| Regex Signal count | Features | 13 | Explicit | Hand-engineered boolean features. |
| LR max_iter | Logistic Regression | 2000 | Explicit | High max_iter to ensure convergence. |
| LR class_weight | Logistic Regression | 'balanced' | Explicit | Handles imbalance. |
| LR penalty | Logistic Regression | l2 | Default | Default L2 regularization. |
| LR C | Logistic Regression | 1.0 | Default | Default regularization strength. |
| HistGBM max_iter | HistGradientBoosting | 500 | Explicit | Maximum number of trees. |
| HistGBM learning_rate | HistGradientBoosting | 0.05 | Explicit | Shrinkage parameter. |
| HistGBM max_depth | HistGradientBoosting | 6 | Explicit | Limits tree depth. |
| HistGBM class_weight | HistGradientBoosting | 'balanced' | Explicit | Crucial for predicting minority classes. |
| HistGBM l2_regularization | HistGradientBoosting | 0.0 | Default | No L2 penalty applied explicitly. |
| HistGBM min_samples_leaf | HistGradientBoosting | 20 | Default | Default leaf minimum. |
| Calibration Method | Calibration | 'sigmoid' | Explicit | Platt Scaling used instead of isotonic. |
| Precision Thresholds | Prediction | denied=0.5, partial=0.4, etc. | Explicit | Hardcoded thresholds for final prediction assignment. |
| Confidence Margin Threshold | Prediction | 0.20 | Explicit | Margin between top 2 classes for high confidence. |
| Signal Boosts | Prediction | settled=1.5, granted=1.3, etc. | Explicit | Heuristic posterior probability multipliers applied after model prediction! |

## Known Methodological Discrepancies & Oddities
- **Appellate StandardScaler on Dummies**: The `StandardScaler(with_mean=False)` is applied to the One-Hot Encoded variables. This scales binary indicators by their variance, drastically distorting the resulting Logistic Regression coefficients and Odds Ratios.
- **Appellate Target Encoding**: Documentation states `Smoothed Target Encoding M=5`, but implementation uses `pd.get_dummies`.
- **Heuristic Signal Boosts**: The `prediction_model.py` script applies arbitrary heuristic multipliers (e.g., `has_settlement` * 1.5) to the `CalibratedClassifierCV` posterior probabilities *after* prediction, destroying calibration.
- **Hardcoded Precision Thresholds**: Final classifications are overridden by arbitrary probability thresholds in `predict_case`.

## Publication 8-Tests Hyperparameter Setup Table

| Test # | Group / Model | Vocab ($V$) | Learning Rate ($\eta$) | Max Depth ($d$) | Trees / Iter ($N$) | Class Weight ($\mathcal{W}$) | Calibration ($\mathcal{C}$) | Inner $k_{cal}$ | Decision / Post-processing Rule |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Naive Baseline** | — | — | — | — | None | None | — | Majority class: `granted` |
| **2** | **Logistic Regression** | 1,000 | — | — | 2,000 | balanced | None | — | $L_2$ penalty, $C=1.0$, `StandardScaler` |
| **3** | **Random Forest** | 1,000 | — | None | 300 | balanced | None | — | Standard bagging ensemble |
| **4** | **HistGBM (Uncalibrated)** | 1,000 | 0.05 | 6 | 500 | balanced | **None (Raw)** | — | Raw gradient boosting logit outputs |
| **5** | **HistGBM + Sigmoid ($k=3$)** | 1,000 | 0.05 | 6 | 500 | balanced | **Sigmoid** | **3** | Platt scaling (CV evaluation model) |
| **6** | **HistGBM + Sigmoid ($k=5$)** | 1,000 | 0.05 | 6 | 500 | balanced | **Sigmoid** | **5** | **Final production classifier** |
| **7** | **HistGBM Calibrated + Rules**| 1,000 | 0.05 | 6 | 500 | balanced | **Sigmoid** | **5** | Margin $\ge 0.20$, $\max(p) \ge 0.50$, per-class cutoffs |
| **8** | **Appellate Logistic Reg** | — | — | — | 2,000 | balanced | None | — | Binary prediction: Upheld (1) vs Other (0) |