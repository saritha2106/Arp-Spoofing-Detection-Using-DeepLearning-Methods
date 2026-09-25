import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

# ==============================
# STEP 1: Load Dataset
# ==============================
data = pd.read_csv("dataset/All_Labelled.csv")

print("Original Dataset Shape:", data.shape)

# ==============================
# STEP 2: Convert Label to Numeric
# ==============================
data["Label"] = data["Label"].map({
    "normal": 0,
    "arp_spoofing": 1
})

# ==============================
# STEP 3: Keep Only Numeric Columns
# ==============================
data = data.select_dtypes(include=[np.number])

print("After Removing Non-Numeric Columns:", data.shape)

print("\nLabel Counts:\n", data["Label"].value_counts())

# ==============================
# STEP 4: Separate Features & Label
# ==============================
X = data.drop("Label", axis=1)
y = data["Label"]

print("\nFeatures Shape:", X.shape)
print("Label Shape:", y.shape)

# ==============================
# STEP 5: Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ==============================
# STEP 6: Feature Scaling
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.save")

print("Scaling Completed.")

# ==============================
# STEP 7: XGBoost Feature Selection
# ==============================
print("\nStarting XGBoost Feature Selection...")

xgb = XGBClassifier(eval_metric='logloss')
xgb.fit(X_train_scaled, y_train)

importances = xgb.feature_importances_
feature_names = X.columns

# Select top 30 features
indices = np.argsort(importances)[-30:]

# Save selected feature indices
np.save("selected_indices.npy", indices)

selected_features = feature_names[indices]

print("\nSelected 30 Features:")
for f in selected_features:
    print(f)

# Select only important features
X_train_selected = X_train_scaled[:, indices]
X_test_selected = X_test_scaled[:, indices]

print("\nSelected Feature Count:", X_train_selected.shape[1])

# ==============================
# STEP 8: Save Processed Data
# ==============================
np.save("X_train.npy", X_train_selected)
np.save("X_test.npy", X_test_selected)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("\nPreprocessing Completed Successfully.")