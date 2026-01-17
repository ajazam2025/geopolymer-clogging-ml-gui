import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor

st.set_page_config(page_title="Geopolymer Pervious Concrete Predictor")

st.title("🧱 Geopolymer Pervious Concrete – ML Prediction App")
st.markdown(
    "Predict **Porosity**, **Permeability**, and **Clogging Rate** using a unified ML model"
)

# --------------------------------------------------
# Load training data (NO USER UPLOAD)
# --------------------------------------------------
@st.cache_data
def load_data():
return pd.read_excel("input data.xlsx")

df = load_data()

# --------------------------------------------------
# Train models (cached)
# --------------------------------------------------
@st.cache_resource
def train_models(df):

    # Porosity
    X_poro = df.drop(columns=["Porosity_percent"])
    y_poro = df["Porosity_percent"]
    poro_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", BayesianRidge())
    ])
    poro_model.fit(X_poro, y_poro)

    # Permeability
    X_perm = df.drop(columns=["Permeability_mm_hr"])
    y_perm = df["Permeability_mm_hr"]
    perm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        ))
    ])
    perm_model.fit(X_perm, y_perm)

    # Clogging
    X_clog = df.drop(columns=["Clogging_Rate_percent_per_year"])
    y_clog = df["Clogging_Rate_percent_per_year"]
    clog_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        ))
    ])
    clog_model.fit(X_clog, y_clog)

    return poro_model, perm_model, clog_model


with st.spinner("🔄 Initializing ML models..."):
    poro_model, perm_model, clog_model = train_models(df)

st.success("✅ Models ready")

# --------------------------------------------------
# User inputs
# --------------------------------------------------
st.sidebar.header("🔧 Input Parameters")

Water_Binder_Ratio = st.sidebar.number_input("Water–Binder Ratio", 0.20, 0.60, 0.35)
NaOH_Molarity = st.sidebar.number_input("NaOH Molarity (M)", 6.0, 16.0, 10.0)
Ns_Nh_Ratio = st.sidebar.number_input("Ns/Nh Ratio", 0.5, 3.0, 1.5)
Fine_Aggregate_percent = st.sidebar.number_input("Fine Aggregate (%)", 0.0, 40.0, 15.0)
Compressive_Strength_MPa = st.sidebar.number_input("Compressive Strength (MPa)", 5.0, 60.0, 25.0)
Predicted_Lifespan_years = st.sidebar.number_input("Design Lifespan (years)", 1, 100, 25)

input_df = pd.DataFrame([{
    "Water_Binder_Ratio": Water_Binder_Ratio,
    "NaOH_Molarity": NaOH_Molarity,
    "Ns_Nh_Ratio": Ns_Nh_Ratio,
    "Fine_Aggregate_percent": Fine_Aggregate_percent,
    "Compressive_Strength_MPa": Compressive_Strength_MPa,
    "Predicted_Lifespan_years": Predicted_Lifespan_years
}])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🚀 Predict All Parameters"):

    porosity = poro_model.predict(input_df)[0]
    input_df["Porosity_percent"] = porosity

    permeability = perm_model.predict(input_df)[0]
    input_df["Permeability_mm_hr"] = permeability

    clogging = clog_model.predict(input_df)[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Porosity (%)", f"{porosity:.2f}")
    col2.metric("Permeability (mm/hr)", f"{permeability:.2f}")
    col3.metric("Clogging Rate (% / year)", f"{clogging:.2f}")

    st.success("✅ Prediction completed successfully")
