# First-Instance and Appellate Model Analysis Summary

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
