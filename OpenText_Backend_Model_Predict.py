import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATA
# =========================

train_df = pd.read_csv("train.csv")
predict_df = pd.read_csv("OpenText Backend Application Prediction.csv", encoding="latin-1")

# =========================
# CLEAN COLUMNS
# =========================

train_df.columns = train_df.columns.str.strip()
predict_df.columns = predict_df.columns.str.strip()

# =========================
# PRESERVE TP CODE (CRITICAL)
# =========================

predict_tp_codes = predict_df["TP Code"].copy()

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

for df in [train_df, predict_df]:
    df[["tp_business", "tp_direction_code", "tp_customer", "tp_message_type"]] = df["TP Code"].apply(parse_tp_code)
    df["customer_message_combo"] = df["tp_customer"] + "_" + df["tp_message_type"]

# =========================
# REMOVE SAME FEATURES AS TRAINING
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
predict_df = predict_df.drop(columns=[col for col in columns_to_exclude if col in predict_df.columns])

# =========================
# TRAIN MODEL
# =========================

X_train = train_df.drop(columns=["Backend Application"])
y_train = train_df["Backend Application"]

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

pipeline.fit(X_train, y_train)

# =========================
# PREDICT
# =========================

probs = pipeline.predict_proba(predict_df)
classes = pipeline.named_steps["model"].classes_

top_n = 3
top_preds = np.argsort(probs, axis=1)[:, -top_n:][:, ::-1]

# =========================
# BUILD OUTPUT
# =========================

results = []

for i, row in enumerate(top_preds):

    top1_idx = row[0]
    top1_prob = probs[i][top1_idx]

    result_row = {
        "tp_code": predict_tp_codes.iloc[i],
        "top_1_backend": classes[top1_idx],
        "top_1_prob": top1_prob,
        "confidence_flag": (
            "HIGH" if top1_prob > 0.8 else
            "MEDIUM" if top1_prob > 0.6 else
            "LOW"
        )
    }

    for j, idx in enumerate(row):
        result_row[f"top_{j+1}_backend"] = classes[idx]
        result_row[f"top_{j+1}_prob"] = probs[i][idx]

    results.append(result_row)

df_results = pd.DataFrame(results)

# =========================
# SAVE OUTPUT
# =========================

df_results.to_csv("backend_predictions_output.csv", index=False)

print("\n✅ Predictions saved to backend_predictions_output.csv")