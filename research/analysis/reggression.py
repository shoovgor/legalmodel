import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# 1. Өгөгдөл унших
DATA_PATH = "output.csv"
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# 2. Өгөгдөл шүүх (Зөвхөн давж заалдах болон хяналтын шатны хэргүүд)
model_df = df[df["case_level"].isin(["appellate", "supervisory"])].copy()
model_df = model_df[model_df["appeal_outcome"].notna()].copy()

# 1 = хэвээр үлдээсэн (upheld), 0 = бусад (өөрчилсөн, хүчингүй болгосон, хэрэгсэхгүй болгосон)
model_df["шийдвэр_хэвээр"] = (model_df["appeal_outcome"] == "upheld").astype(int)

# 3. Шинж чанар (Features) бэлтгэх
model_df["нэхэмжлэлийн_дүн_байгаа_эсэх"] = model_df["claim_amount_requested"].notna().astype(int)
model_df["нэхэмжлэлийн_дүн_log"] = np.log1p(model_df["claim_amount_requested"].fillna(0))

model_df["нэхэмжлэгч"] = model_df["plaintiff_type"].fillna("тодорхойгүй")
model_df["хариуцагч"] = model_df["defendant_type"].fillna("тодорхойгүй")
model_df["шүүхийн_шат"] = model_df["case_level"].fillna("тодорхойгүй")
model_df["талуудын_харилцаа"] = model_df["нэхэмжлэгч"] + "_vs_" + model_df["хариуцагч"]
model_df["гомдол_гаргагч"] = model_df["appealing_party"].fillna("тодорхойгүй")

model_df["холбогдох_шийдвэрийн_тоо"] = (
    model_df["referenced_decisions"]
    .fillna("")
    .apply(lambda x: len(x.split("|")) if x else 0)
)

global_rate = model_df["шийдвэр_хэвээр"].mean()
SMOOTHING_M = 5 

def smoothed_rate(group_col, df=model_df):
    grouped = df.groupby(group_col)["шийдвэр_хэвээр"].agg(["sum", "count"])
    grouped.columns = ["хэвээр_үлдсэн", "нийт"]
    grouped["бодит_хувь"] = grouped["хэвээр_үлдсэн"] / grouped["нийт"]
    grouped["тэгшитгэсэн_хувь"] = (
        (grouped["нийт"] * grouped["бодит_хувь"] + SMOOTHING_M * global_rate)
        / (grouped["нийт"] + SMOOTHING_M)
    )
    return grouped.sort_values("тэгшитгэсэн_хувь")

print(f"Нийт хэргийн тоо: {len(model_df)}")
print(f"Шийдвэр хэвээр үлдсэн дундаж магадлал: {global_rate:.1%}\n")

# Гол хувьсагчдын нөлөө (Тэгшитгэсэн магадлал)
print("--- Шүүхийн шатаар ---")
print(smoothed_rate("шүүхийн_шат").to_string())

print("\n--- Гомдол гаргагчаар ---")
print(smoothed_rate("гомдол_гаргагч").to_string())

# 4. Ложистик Регресс (Логистик регресс)
feature_cols_numeric = ["нэхэмжлэлийн_дүн_log", "нэхэмжлэлийн_дүн_байгаа_эсэх", "холбогдох_шийдвэрийн_тоо"]
dummies = pd.get_dummies(
    model_df[["нэхэмжлэгч", "хариуцагч", "шүүхийн_шат", "гомдол_гаргагч"]],
    drop_first=True,
)

X = pd.concat([model_df[feature_cols_numeric].reset_index(drop=True),
               dummies.reset_index(drop=True)], axis=1)
y = model_df["шийдвэр_хэвээр"].reset_index(drop=True)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", random_state=42)),
])

# 5-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

print(f"\n--- Ложистик Регрессийн Үр дүн ---")
print(f"5-Fold CV Нарийвчлал (Accuracy): {scores.mean():.3f} ± {scores.std():.3f}")

pipe.fit(X, y)
lr_model = pipe.named_steps["lr"]

print("\n--- Хувьсагчдын Нөлөөлөл (Odds Ratios) ---")
coef_df = pd.DataFrame({
    "Хувьсагч": X.columns,
    "Коэффициент": lr_model.coef_[0],
    "Нөлөөлөл (Odds Ratio)": np.exp(lr_model.coef_[0]),
}).sort_values("Коэффициент", ascending=False)
print(coef_df.to_string(index=False))
