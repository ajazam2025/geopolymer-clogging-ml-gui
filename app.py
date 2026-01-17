import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Geopolymer Pervious Concrete Predictor")

st.title("🧱 Geopolymer Pervious Concrete – ML Prediction App")
st.markdown(
    "Predict **Porosity**, **Permeability**, and **Clogging Rate** in a single application"
)

# Load trained models
poro_model = joblib.load("models/porosity_model.pkl")
perm_model = joblib.load("models/permeability_model.pkl")
clog_model = joblib.load("models/clogging_model.pkl")

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

if st.button("🚀 Predict All Parameters"):
    porosity = poro_model.predict(input_df)[0]
    input_df["Porosity_percent"] = porosity

    permeability = perm_model.predict(input_df)[0]
    input_df["Permeability_mm_hr"] = permeability

    clogging = clog_model.predict(input_df)[0]

    st.success("✅ Prediction Successful")

    col1, col2, col3 = st.columns(3)
    col1.metric("Porosity (%)", f"{porosity:.2f}")
    col2.metric("Permeability (mm/hr)", f"{permeability:.2f}")
    col3.metric("Clogging Rate (% / year)", f"{clogging:.2f}")
