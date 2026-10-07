"""
STEP 2: Preprocess the data, train the ML model, evaluate it and save it.
Run:  python train_model.py
Output: model/demand_model.pkl, model/metrics.json, model/test_results.csv
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("data/canteen_sales.csv")
print("Rows, columns:", df.shape)

# ---------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------
print("Missing values per column:\n", df.isnull().sum())
df = df.dropna()                 # remove rows with missing values (none here)
df = df.drop_duplicates()        # remove duplicate rows
df = df[df["quantity_sold"] >= 0]  # quantity can never be negative

# Inputs (X) and the value we want to predict (y)
feature_cols = ["food_item", "day_of_week", "month", "temperature",
                "is_rainy", "is_exam_period", "is_holiday"]
X = df[feature_cols]
y = df["quantity_sold"]

# ML models understand only numbers, so the text column "food_item"
# is converted to 0/1 columns using One-Hot Encoding.
preprocessor = ColumnTransformer(
    transformers=[("food", OneHotEncoder(handle_unknown="ignore"), ["food_item"])],
    remainder="passthrough",     # keep the other (already numeric) columns as they are
)

# ---------------------------------------------------------------
# 3. TRAIN / TEST SPLIT  (80% training, 20% testing)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")

# ---------------------------------------------------------------
# 4. TRAIN TWO MODELS AND COMPARE
# ---------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

results = {}
trained = {}
for name, algo in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", algo)])
    pipe.fit(X_train, y_train)               # learning happens here
    pred = pipe.predict(X_test)              # predict on unseen data
    results[name] = {
        "MAE": round(mean_absolute_error(y_test, pred), 2),
        "R2": round(r2_score(y_test, pred), 4),
    }
    trained[name] = pipe
    print(f"{name}: MAE = {results[name]['MAE']}, R2 = {results[name]['R2']}")

# ---------------------------------------------------------------
# 5. PICK THE BEST MODEL (lowest MAE) AND SAVE EVERYTHING
# ---------------------------------------------------------------
best_name = min(results, key=lambda n: results[n]["MAE"])
print("Best model:", best_name)

os.makedirs("model", exist_ok=True)
joblib.dump(trained[best_name], "model/demand_model.pkl")

with open("model/metrics.json", "w") as f:
    json.dump({"best_model": best_name, "results": results}, f, indent=2)

# Save test predictions so the app can draw "Actual vs Predicted" chart
test_out = X_test.copy()
test_out["actual"] = y_test.values
test_out["predicted"] = trained[best_name].predict(X_test).round(0)
test_out.to_csv("model/test_results.csv", index=False)

print("Model, metrics and test results saved in the 'model' folder.")
