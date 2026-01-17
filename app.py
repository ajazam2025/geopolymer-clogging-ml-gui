
import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# ----------------------------------------
# Page setup
# ----------------------------------------
st.set_page_config(page_title="Clogging Prediction App")

st.title("🧱 Geopolymer Pervious Concrete – Clogging Prediction")
st.markdown(
    "Predict **Clogging Rate (% per year)** using a machine learning model"
)

# ----------------------------------------
# Load training data (from GitHub)
# ----------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("input data.xlsx")

df = load_data()

# ----------------------------------------
# Train clogging model (cached)
# ----------------------------------------
@st.cache_resource
def train_clogging_model(df):

    X = df.drop(columns=["Clogging_Rate_percent_per_year"])
    y = df["Clogging_Rate_percent_per_year"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ))
    ])

    model.fit(X, y)
    return model


with st.spinner("🔄 Initializing clogging prediction model..."):
    clog_model = train_clogging_model(df)

st.success("✅ Model ready")

# ----------------------------------------
# User inputs
# ----------------------------------------
st.sidebar.header("🔧 Input Parameters")

Water_Binder_Ratio = st.sidebar.number_input("Water–Binder Ratio", 0.20, 0.60, 0.35)
NaOH_Molarity = st.sidebar.number_input("NaOH Molarity (M)", 6.0, 16.0, 10.0)
Ns_Nh_Ratio = st.sidebar.number_input("Ns/Nh Ratio", 0.5, 3.0, 1.5)
Fine_Aggregate_percent = st.sidebar.number_input("Fine Aggregate (%)", 0.0, 40.0, 15.0)
Compressive_Strength_MPa = st.sidebar.number_input("Compressive Strength (MPa)", 5.0, 60.0, 25.0)
Permeability_mm_hr = st.sidebar.number_input("Permeability (mm/hr)", 10.0, 5000.0, 1000.0)
Porosity_percent = st.sidebar.number_input("Porosity (%)", 5.0, 35.0, 20.0)
Predicted_Lifespan_years = st.sidebar.number_input("Design Lifespan (years)", 1, 100, 25)

input_df = pd.DataFrame([{
    "Water_Binder_Ratio": Water_Binder_Ratio,
    "NaOH_Molarity": NaOH_Molarity,
    "Ns_Nh_Ratio": Ns_Nh_Ratio,
    "Fine_Aggregate_percent": Fine_Aggregate_percent,
    "Compressive_Strength_MPa": Compressive_Strength_MPa,
    "Permeability_mm_hr": Permeability_mm_hr,
    "Porosity_percent": Porosity_percent,
    "Predicted_Lifespan_years": Predicted_Lifespan_years
}])

# ----------------------------------------
# Prediction
# ----------------------------------------
if st.button("🚀 Predict Clogging Rate"):

    clogging = clog_model.predict(input_df)[0]

    st.metric(
        label="Clogging Rate (% per year)",
        value=f"{clogging:.2f}"
    )

    st.success("✅ Prediction completed successfully")
