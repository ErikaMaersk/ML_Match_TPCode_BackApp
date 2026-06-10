import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# LOAD FULL LABELLED DATASET
# =========================

# Replace with your updated file name
df = pd.read_csv("Raw_Train_Test_Validate_Dataset.csv")

# =========================
# CLEAN COLUMN NAMES
# =========================

df.columns = df.columns.str.strip()

# =========================
# OPTIONAL: REMOVE NULL TARGETS (SAFETY)
# =========================

df = df[df["Backend Application"].notna()]

print("\nTotal rows in dataset:", len(df))

# =========================
# HANDLE RARE CLASSES (IMPORTANT)
# =========================

counts = df["Backend Application"].value_counts()

# Group very small classes into OTHER
threshold = 20   # You can adjust if needed

df["Backend Application"] = df["Backend Application"].apply(
    lambda x: x if counts[x] >= threshold else "OTHER"
)

print("\nBackend Application distribution AFTER grouping:")
print(df["Backend Application"].value_counts())

print("\nNumber of 'OTHER' rows:",
      (df["Backend Application"] == "OTHER").sum())

# =========================
# STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# =========================

# First split: Train (70%) + Temp (30%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["Backend Application"],
    random_state=42
)

# Second split: Validation (15%) + Test (15%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["Backend Application"],
    random_state=42
)

# =========================
# SAVE FILES
# =========================

train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

# =========================
# VALIDATION OUTPUT
# =========================

print("\nSplit sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTrain distribution:")
print(train_df["Backend Application"].value_counts(normalize=True))

print("\nValidation distribution:")
print(val_df["Backend Application"].value_counts(normalize=True))

print("\nTest distribution:")
print(test_df["Backend Application"].value_counts(normalize=True))

print("\n✅ Data split complete")