import streamlit as st
import pandas as pd
import joblib
import os
import subprocess

st.set_page_config(page_title="Geopolymer Pervious Concrete Predictor")

st.title("🧱 Geopolymer Pervious Concrete – ML Prediction App")
st.markdown(
    "Predict **Porosity**, **Permeability**, and **Clogging Rate** using a single ML-based GUI"
)

# -----------------------------
# Upload data
# -----------------------------
st.sidebar.header("📂 Upload Input Data (Excel)")

uploaded_file = st.sidebar.file_uploader(
    "Upload input data.xlsx",
    type=["xlsx"]
)

# -----------------------------
# Train models if not present
# -----------------------------
if uploaded_file is not None:

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    data_path = "data/input data.xlsx"
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if not os.path.exists("models/porosity_model.pkl"):
        st.info("🔄 Training models (first-time setup)...")
        subprocess.run(["python", "train_models.py"])
        st.success("✅ Models trained successfully")

    # Load models
    poro_model = joblib.load("models/porosity_model.pkl")
    perm_model = joblib.load("models/permeability_model.pkl")
    clog_model = joblib.load("models/clogging_model.pkl")

    # -----------------------------
    # Input parameters
    # -----------------------------
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

    # -----------------------------
    # Prediction
    # -----------------------------
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

else:
    st.warning("⬅️ Please upload `input data.xlsx` to start")
