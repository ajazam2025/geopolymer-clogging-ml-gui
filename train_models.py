import os
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor

os.makedirs("models", exist_ok=True)

df = pd.read_excel("data/input data.xlsx")

# Porosity model
X_poro = df.drop(columns=["Porosity_percent"])
y_poro = df["Porosity_percent"]

poro_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", BayesianRidge())
])
poro_model.fit(X_poro, y_poro)
joblib.dump(poro_model, "models/porosity_model.pkl")

# Permeability model
X_perm = df.drop(columns=["Permeability_mm_hr"])
y_perm = df["Permeability_mm_hr"]

perm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42))
])
perm_model.fit(X_perm, y_perm)
joblib.dump(perm_model, "models/permeability_model.pkl")

# Clogging model
X_clog = df.drop(columns=["Clogging_Rate_percent_per_year"])
y_clog = df["Clogging_Rate_percent_per_year"]

clog_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42))
])
clog_model.fit(X_clog, y_clog)
joblib.dump(clog_model, "models/clogging_model.pkl")

print("Models trained and saved successfully.")
