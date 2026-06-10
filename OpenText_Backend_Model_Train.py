import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# =========================
# LOAD DATA
# =========================

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")

# =========================
# CLEAN COLUMN NAMES
# =========================

train_df.columns = train_df.columns.str.strip()
val_df.columns = val_df.columns.str.strip()

# =========================
# PARSE TP CODE
# =========================

def parse_tp_code(tp):
    try:
        parts = str(tp).split("_")

        prefix = parts[0] if len(parts) > 0 else ""
        customer = parts[1] if len(parts) > 1 else ""
        message_type = parts[2] if len(parts) > 2 else ""

        business = prefix[0] if len(prefix) > 0 else ""
        direction_code = prefix[1] if len(prefix) > 1 else ""

        return pd.Series([business, direction_code, customer, message_type])

    except:
        return pd.Series(["", "", "", ""])

for df in [train_df, val_df]:
    df[["tp_business", "tp_direction_code", "tp_customer", "tp_message_type"]] = df["TP Code"].apply(parse_tp_code)
    df["customer_message_combo"] = df["tp_customer"] + "_" + df["tp_message_type"]

# =========================
# REMOVE LEAKAGE + WEAK FEATURES
# =========================

columns_to_exclude = [
    "TP Code",
    "billing_id",
    "Damco ID",
    "Month",
    "Destination Application from Axway Team",
    "TP NAME for REPORTS",
    "Flow ID Values from OpenText TPCodes",
    "Christian, Yuanhao, Confluence Source",
    "Christian, Yuanhao, Confluence Backend App",
    "Business Unit by Naming Convention"
]

train_df = train_df.drop(columns=[col for col in columns_to_exclude if col in train_df.columns])
val_df = val_df.drop(columns=[col for col in columns_to_exclude if col in val_df.columns])

# =========================
# DEFINE FEATURES / TARGET
# =========================

X_train = train_df.drop(columns=["Backend Application"])
y_train = train_df["Backend Application"]

X_val = val_df.drop(columns=["Backend Application"])
y_val = val_df["Backend Application"]

print("\nFeatures used:")
print(X_train.columns.tolist())

# =========================
# MODEL PIPELINE
# =========================

categorical_cols = X_train.columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

# =========================
# TRAIN
# =========================

pipeline.fit(X_train, y_train)

# =========================
# VALIDATION
# =========================

y_pred = pipeline.predict(X_val)

print("\nValidation Accuracy:")
print(accuracy_score(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# =========================
# SAVE VALIDATION OUTPUT
# =========================

probs = pipeline.predict_proba(X_val)
classes = pipeline.named_steps["model"].classes_

top_n = 3
top_preds = np.argsort(probs, axis=1)[:, -top_n:][:, ::-1]

results = []

for i, row in enumerate(top_preds):
    result_row = {
        "actual_backend": y_val.iloc[i],
        "tp_customer": val_df.iloc[i]["tp_customer"],
        "tp_message_type": val_df.iloc[i]["tp_message_type"]
    }

    for j, idx in enumerate(row):
        result_row[f"top_{j+1}_backend"] = classes[idx]
        result_row[f"top_{j+1}_prob"] = probs[i][idx]

    results.append(result_row)

pd.DataFrame(results).to_csv("validation_predictions_final.csv", index=False)

print("\n✅ Validation predictions saved")
