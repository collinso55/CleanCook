# ==============================================================
# 🔥 CleanCook Streamlit App — Firewood vs Clean Fuel Predictor
# ==============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

# --------------------------------------------------------------
# 1️⃣ Load model and scaler
# --------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("lstm_multioutput_model.h5", compile=False)
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# --------------------------------------------------------------
# 2️⃣ Page configuration
# --------------------------------------------------------------
st.set_page_config(page_title="CleanCook Predictor", page_icon="🔥", layout="centered")
st.title("🔥 CleanCook Predictor — Firewood vs Clean Fuel Access")

st.markdown("""
This app predicts **Firewood Demand** and **Clean Fuel Access** in Cameroon  
using environmental and historical parameters with our trained LSTM model.
""")

# --------------------------------------------------------------
# 3️⃣ Input parameters
# --------------------------------------------------------------
st.sidebar.header("🧭 Input Parameters")

# Date → cyclic encoding
date_input = st.sidebar.date_input("Select Date", datetime(2024, 1, 1))
day_of_year = date_input.timetuple().tm_yday
sin_doy = np.sin(2 * np.pi * day_of_year / 365.25)
cos_doy = np.cos(2 * np.pi * day_of_year / 365.25)
is_weekend = 1 if date_input.weekday() >= 5 else 0

# Weather inputs (these are scaled by the scaler)
temperature_avg = st.sidebar.slider("Avg Temperature (°C)", 10.0, 45.0, 25.0)
rainfall_mm = st.sidebar.slider("Rainfall (mm)", 0.0, 200.0, 10.0)
humidity_avg = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 60.0)

# Historical values (not scaled)
st.sidebar.markdown("### 🔁 Lag & Rolling Averages (approximated)")
firewood_lag_1 = st.sidebar.number_input("Firewood lag 1 day (m³)", 0.0, 10000.0, 3000.0)
firewood_lag_7 = st.sidebar.number_input("Firewood lag 7 days (m³)", 0.0, 10000.0, 2800.0)
firewood_lag_30 = st.sidebar.number_input("Firewood lag 30 days (m³)", 0.0, 10000.0, 2700.0)
cleanfuel_lag_1 = st.sidebar.slider("Clean fuel lag 1 day", 0.0, 1.0, 0.3)
cleanfuel_lag_7 = st.sidebar.slider("Clean fuel lag 7 days", 0.0, 1.0, 0.4)
cleanfuel_lag_30 = st.sidebar.slider("Clean fuel lag 30 days", 0.0, 1.0, 0.5)
fw_roll_mean_30 = st.sidebar.number_input("30-day rolling mean firewood (m³)", 0.0, 10000.0, 2500.0)
cf_roll_mean_30 = st.sidebar.slider("30-day rolling mean clean fuel", 0.0, 1.0, 0.45)

# --------------------------------------------------------------
# 4️⃣ Build input DataFrame (exact feature order for model)
# --------------------------------------------------------------
input_data = {
    'temperature_avg': temperature_avg,
    'rainfall_mm': rainfall_mm,
    'humidity_avg': humidity_avg,
    'sin_doy': sin_doy,
    'cos_doy': cos_doy,
    'is_weekend': is_weekend,
    'firewood_demand_m3_lag_1': firewood_lag_1,
    'firewood_demand_m3_lag_7': firewood_lag_7,
    'firewood_demand_m3_lag_30': firewood_lag_30,
    'clean_fuel_access_lag_1': cleanfuel_lag_1,
    'clean_fuel_access_lag_7': cleanfuel_lag_7,
    'clean_fuel_access_lag_30': cleanfuel_lag_30,
    'fw_roll_mean_30': fw_roll_mean_30,
    'cf_roll_mean_30': cf_roll_mean_30
}

input_df = pd.DataFrame([input_data])

# --------------------------------------------------------------
# 5️⃣ Scale only the features known by the scaler
# --------------------------------------------------------------
scaled_part = scaler.transform(
    [[temperature_avg, rainfall_mm, humidity_avg, 0, 0]]
)[:, :3]  # take only scaled weather features
scaled_df = pd.DataFrame(scaled_part, columns=['temperature_avg', 'rainfall_mm', 'humidity_avg'])

# Replace in input_df
input_df[['temperature_avg', 'rainfall_mm', 'humidity_avg']] = scaled_df

# --------------------------------------------------------------
# 6️⃣ Prepare for LSTM
# --------------------------------------------------------------
X_scaled = np.expand_dims(input_df.values, axis=1)

# --------------------------------------------------------------
# 7️⃣ Predict
# --------------------------------------------------------------
if st.button("🔮 Predict"):
    preds = model.predict(X_scaled)
    firewood_pred, cleanfuel_pred = preds[0]

    st.subheader("📊 Prediction Results")
    st.metric("Predicted Firewood Demand (m³)", f"{firewood_pred:.2f}")
    st.metric("Predicted Clean Fuel Access (0–1 index)", f"{cleanfuel_pred:.2f}")

    st.markdown("### 🧩 Interpretation")
    if cleanfuel_pred > firewood_pred:
        st.success("✅ **Higher clean fuel access** than firewood demand — positive energy transition.")
    else:
        st.warning("⚠️ **Firewood demand exceeds clean fuel access** — continued biomass dependence.")

st.caption("Built with ❤️ by Team CleanCook | Powered by LSTM Forecasting Engine")
