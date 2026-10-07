# First-Instance Civil Court Outcome Classifier: Analysis Outputs

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
