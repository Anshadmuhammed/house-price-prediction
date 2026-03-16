"""
train.py  –  Train and save the House Price Prediction model.
Run once before launching app.py:
    python train.py
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ── Generate synthetic training data ─────────────────────────────────────────
# Replace with a real dataset like Boston Housing or Kaggle House Prices
np.random.seed(42)
N = 500

locations  = ["Urban", "Suburban", "Rural"]
conditions = ["Excellent", "Good", "Fair", "Poor"]

data = pd.DataFrame({
    "area_sqft":     np.random.randint(500, 5000, N),
    "bedrooms":      np.random.randint(1, 7, N),
    "bathrooms":     np.random.randint(1, 5, N),
    "age_years":     np.random.randint(0, 50, N),
    "floors":        np.random.randint(1, 4, N),
    "garage":        np.random.randint(0, 2, N),
    "garden":        np.random.randint(0, 2, N),
    "location":      np.random.choice(locations, N),
    "condition":     np.random.choice(conditions, N),
})

loc_map  = {"Urban": 1.3, "Suburban": 1.0, "Rural": 0.7}
cond_map = {"Excellent": 1.2, "Good": 1.0, "Fair": 0.85, "Poor": 0.7}

data["price"] = (
    data["area_sqft"]  * 120
    + data["bedrooms"] * 8000
    + data["bathrooms"]* 5000
    - data["age_years"]* 500
    + data["floors"]   * 3000
    + data["garage"]   * 10000
    + data["garden"]   * 7000
) * data["location"].map(loc_map) * data["condition"].map(cond_map)

data["price"] += np.random.normal(0, 15000, N)
data["price"]  = data["price"].clip(lower=30000).round(-2)

print("Dataset shape:", data.shape)
print(data.describe())

# ── Preprocessing ─────────────────────────────────────────────────────────────
NUM_FEATURES = ["area_sqft","bedrooms","bathrooms","age_years","floors","garage","garden"]
CAT_FEATURES = ["location","condition"]
TARGET       = "price"

X = data[NUM_FEATURES + CAT_FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])

# ── Train models & compare ────────────────────────────────────────────────────
models = {
    "Linear Regression":    LinearRegression(),
    "Ridge Regression":     Ridge(alpha=1.0),
    "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
}

print("\n=== Model Comparison ===")
best_name, best_pipe, best_r2 = None, None, -999

for name, model in models.items():
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse= np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name:25s}  R²={r2:.4f}  MAE={mae:,.0f}  RMSE={rmse:,.0f}")
    if r2 > best_r2:
        best_r2, best_name, best_pipe = r2, name, pipe

print(f"\nBest model: {best_name}  (R²={best_r2:.4f})")

# ── Save best model ───────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({"model": best_pipe, "features": NUM_FEATURES + CAT_FEATURES}, f)

print("Saved → model.pkl")
print("Run:  streamlit run app.py")
