#GRun: python3 train_and_save_model.py

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import features

CSV_PATH = "output.csv"

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
fi = df[(df.case_level == "first_instance") & (df.civil_outcome.notna())].copy()
print(f"training on {len(fi)} labeled first_instance rows")

X, median_claim, judge_counts = features.build_training_features(fi)
y = fi["civil_outcome"]

from sklearn.calibration import CalibratedClassifierCV

preprocess = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), features.FEATURES_CAT),
        ("text", TfidfVectorizer(max_features=1000, min_df=5, ngram_range=(1, 2), sublinear_tf=True), features.TEXT_COL),
    ],
    remainder="passthrough",
    sparse_threshold=0,
)
clf = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.05, max_depth=6,
    class_weight="balanced", random_state=42,
)
pipe = Pipeline([("prep", preprocess), ("clf", clf)])
calibrated = CalibratedClassifierCV(pipe, cv=5, method="sigmoid")
calibrated.fit(X, y)

joblib.dump(
    {"pipeline": calibrated, "median_claim": median_claim, "judge_counts": judge_counts, "classes": list(calibrated.classes_)},
    "trained_model.joblib",
)
print("saved -> trained_model.joblib")
print(f"classes: {list(calibrated.classes_)}")