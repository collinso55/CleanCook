
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

# # Historical values (not scaled)
# st.sidebar.markdown("### 🔁 Lag & Rolling Averages (approximated)")
# firewood_lag_1 = st.sidebar.number_input("Firewood lag 1 day (m³)", 0.0, 10000.0, 3000.0)
# firewood_lag_7 = st.sidebar.number_input("Firewood lag 7 days (m³)", 0.0, 10000.0, 2800.0)
# firewood_lag_30 = st.sidebar.number_input("Firewood lag 30 days (m³)", 0.0, 10000.0, 2700.0)
# cleanfuel_lag_1 = st.sidebar.slider("Clean fuel lag 1 day", 0.0, 1.0, 0.3)
# cleanfuel_lag_7 = st.sidebar.slider("Clean fuel lag 7 days", 0.0, 1.0, 0.4)
# cleanfuel_lag_30 = st.sidebar.slider("Clean fuel lag 30 days", 0.0, 1.0, 0.5)
# fw_roll_mean_30 = st.sidebar.number_input("30-day rolling mean firewood (m³)", 0.0, 10000.0, 2500.0)
# cf_roll_mean_30 = st.sidebar.slider("30-day rolling mean clean fuel", 0.0, 1.0, 0.45)


# --------------------------------------------------------------
# 🔁 Past Energy Usage Patterns — Lags and Rolling Averages
# --------------------------------------------------------------
st.sidebar.markdown("### 🔁 Past Energy Usage Patterns")

st.sidebar.markdown("""
These represent **how energy demand and access behaved recently**.  
Adjust them to simulate how past trends affect today's prediction.
""")

# --- Firewood History ---
st.sidebar.divider()
st.sidebar.markdown("#### 🔥 Firewood History")

firewood_lag_1 = st.sidebar.number_input(
    "Yesterday’s Firewood Usage (m³)",
    0.0, 10000.0, 3000.0,
    help="Estimated firewood demand one day ago."
)
firewood_lag_7 = st.sidebar.number_input(
    "Firewood Usage 1 Week Ago (m³)",
    0.0, 10000.0, 2800.0,
    help="Approximate firewood demand from 7 days ago."
)
firewood_lag_30 = st.sidebar.number_input(
    "Firewood Usage 1 Month Ago (m³)",
    0.0, 10000.0, 2700.0,
    help="Average firewood demand around 30 days ago."
)
fw_roll_mean_30 = st.sidebar.number_input(
    "Monthly Firewood Usage Trend (30-Day Average, m³)",
    0.0, 10000.0, 2500.0,
    help="Average daily firewood usage across the last 30 days — smooths out daily variations."
)

# --- Clean Fuel History ---
st.sidebar.divider()
st.sidebar.markdown("#### ⛽ Clean Fuel History")

cleanfuel_lag_1 = st.sidebar.slider(
    "Yesterday’s Clean Fuel Access (0–1 scale)",
    0.0, 1.0, 0.3,
    help="Clean fuel access level one day ago. 0 = none, 1 = full access."
)
cleanfuel_lag_7 = st.sidebar.slider(
    "Clean Fuel Access 1 Week Ago (0–1 scale)",
    0.0, 1.0, 0.4,
    help="Clean fuel access level 7 days ago."
)
cleanfuel_lag_30 = st.sidebar.slider(
    "Clean Fuel Access 1 Month Ago (0–1 scale)",
    0.0, 1.0, 0.5,
    help="Average clean fuel access 30 days ago."
)
cf_roll_mean_30 = st.sidebar.slider(
    "Monthly Clean Fuel Access Trend (30-Day Average, 0–1 scale)",
    0.0, 1.0, 0.45,
    help="Average clean fuel access over the last 30 days — smooths out short-term changes."
)

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
    y_pred_scaled = model.predict(X_scaled)
    
    # inverse-transform (2 targets: firewood_demand_m3, clean_fuel_access)
    n_total_cols = scaler.n_features_in_
    tmp = np.zeros((1, n_total_cols))
    tmp[:, -2:] = y_pred_scaled
    y_intermediate=tmp[:, -2:]
    y_pred_actual = scaler.inverse_transform(tmp)[:, -2:]

    st.subheader("📊 Prediction Results")
    st.metric("Predicted Firewood Demand (m³)", f"{y_intermediate[0, 0]:.2f}")
    st.metric("Predicted Clean Fuel Access (0–1 index)", f"{y_intermediate[0, 1]:.2f}")
    firewood_pred_m3 = y_pred_actual[0, 0]
    gas_pred_tons = y_pred_actual[0, 1] * 100  # scaled 0–1 → convert to tons proxy

    st.success("✅ Prediction complete!")
    st.metric("🔥 Firewood Demand", f"{firewood_pred_m3:,.2f} m³")
    st.metric("⛽ Gas (Clean Fuel Access)", f"{gas_pred_tons:,.2f} tons (proxy)")

    st.markdown("### 🧩 Interpretation")
    if y_pred_actual[0, 1] > y_pred_actual[0, 0]:
        st.success("✅ **Higher clean fuel access** than firewood demand — positive energy transition.")
    else:
        st.warning("⚠️ **Firewood demand exceeds clean fuel access** — continued biomass dependence.")


    # st.markdown(f"""
    # **Interpretation:**
    # - Estimated daily **firewood demand**: `{firewood_pred_m3:,.2f} m³`
    # - Equivalent **clean fuel (gas)**: `{gas_pred_tons:,.2f} tons`
    # """)

    #---------------------------------------------------------------------------------
    # preds = model.predict(X_scaled)
    # firewood_pred, cleanfuel_pred = preds[0]

    # st.subheader("📊 Prediction Results")
    # st.metric("Predicted Firewood Demand (m³)", f"{firewood_pred:.2f}")
    # st.metric("Predicted Clean Fuel Access (0–1 index)", f"{cleanfuel_pred:.2f}")

    # st.markdown("### 🧩 Interpretation")
    # if cleanfuel_pred > firewood_pred:
    #     st.success("✅ **Higher clean fuel access** than firewood demand — positive energy transition.")
    # else:
    #     st.warning("⚠️ **Firewood demand exceeds clean fuel access** — continued biomass dependence.")
#------------------------------------------------------------------------------------------------------------
st.caption("Built  by Team CleanCook | Powered by LSTM Forecasting Engine")
st.caption('SEED.inc')

# streamlit_app.py
# ==============================================================
# 🔥 CleanCook: Firewood vs Gas Prediction Dashboard
# ==============================================================
# Uses the trained LSTM model + scaler to predict firewood demand (m³)
# and clean fuel access (gas usage proxy in tons)
# ==============================================================

# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib
# from tensorflow.keras.models import load_model
# from datetime import datetime

# # --------------------------------------------------------------
# # ⚙️ 1. Load model and scaler
# # --------------------------------------------------------------
# MODEL_PATH = "lstm_multioutput_model.h5"
# SCALER_PATH = "scaler.pkl"

# @st.cache_resource
# def load_resources():
#     #model = load_model(MODEL_PATH)
#     model = load_model(MODEL_PATH, custom_objects={'mse': 'mse'})
#     scaler = joblib.load(SCALER_PATH)
#     return model, scaler

# model, scaler = load_resources()

# st.title("🔥 CleanCook: Firewood vs Gas Demand Predictor")
# st.markdown("""
# Predict **future firewood demand (m³)** and **clean fuel access (gas, tons)**  
# based on weather conditions and recent energy usage trends.
# """)

# st.divider()

# # --------------------------------------------------------------
# # 🌤️ 2. Weather Inputs
# # --------------------------------------------------------------
# st.sidebar.header("🌦 Weather Conditions")
# temperature_avg = st.sidebar.number_input(
#     "Average Temperature (°C)",
#     min_value=-5.0, max_value=50.0, value=25.0,
#     help="Average daily temperature for the selected date and region."
# )
# rainfall_mm = st.sidebar.number_input(
#     "Rainfall (mm)",
#     min_value=0.0, max_value=500.0, value=20.0,
#     help="Daily rainfall amount in millimeters."
# )
# humidity_avg = st.sidebar.slider(
#     "Average Humidity (%)",
#     min_value=0, max_value=100, value=65,
#     help="Relative humidity percentage."
# )

# # --------------------------------------------------------------
# # 📅 3. Temporal Inputs
# # --------------------------------------------------------------
# st.sidebar.header("📆 Date Information")
# date_input = st.sidebar.date_input(
#     "Select Date",
#     value=datetime.today(),
#     help="Prediction date (affects cyclical day-of-year features)."
# )
# day_of_year = date_input.timetuple().tm_yday
# sin_doy = np.sin(2 * np.pi * day_of_year / 365.25)
# cos_doy = np.cos(2 * np.pi * day_of_year / 365.25)

# is_weekend = 1 if date_input.weekday() >= 5 else 0

# # --------------------------------------------------------------
# # 🔁 4. Past Energy Usage Patterns — Lag and Rolling Features
# # --------------------------------------------------------------
# st.sidebar.header("🔁 Past Energy Usage Patterns")

# st.sidebar.markdown("""
# These represent **how energy demand and access behaved recently**.  
# Adjust them to simulate how past patterns influence the next prediction.
# """)

# # --- Firewood History ---
# st.sidebar.subheader("🔥 Firewood History")
# firewood_lag_1 = st.sidebar.number_input(
#     "Yesterday’s Firewood Usage (m³)",
#     0.0, 10000.0, 3000.0,
#     help="Estimated firewood demand one day ago."
# )
# firewood_lag_7 = st.sidebar.number_input(
#     "Firewood Usage 1 Week Ago (m³)",
#     0.0, 10000.0, 2800.0,
#     help="Approximate firewood demand from 7 days ago."
# )
# firewood_lag_30 = st.sidebar.number_input(
#     "Firewood Usage 1 Month Ago (m³)",
#     0.0, 10000.0, 2700.0,
#     help="Average firewood demand around 30 days ago."
# )
# fw_roll_mean_30 = st.sidebar.number_input(
#     "Monthly Firewood Usage Trend (30-Day Average, m³)",
#     0.0, 10000.0, 2500.0,
#     help="Average daily firewood usage across the last 30 days — smooths out daily variations."
# )

# # --- Clean Fuel History ---
# st.sidebar.subheader("⛽ Clean Fuel History")
# cleanfuel_lag_1 = st.sidebar.slider(
#     "Yesterday’s Clean Fuel Access (0–1 scale)",
#     0.0, 1.0, 0.3,
#     help="Clean fuel access level one day ago. 0 = none, 1 = full access."
# )
# cleanfuel_lag_7 = st.sidebar.slider(
#     "Clean Fuel Access 1 Week Ago (0–1 scale)",
#     0.0, 1.0, 0.4,
#     help="Clean fuel access level 7 days ago."
# )
# cleanfuel_lag_30 = st.sidebar.slider(
#     "Clean Fuel Access 1 Month Ago (0–1 scale)",
#     0.0, 1.0, 0.5,
#     help="Average clean fuel access 30 days ago."
# )
# cf_roll_mean_30 = st.sidebar.slider(
#     "Monthly Clean Fuel Access Trend (30-Day Average, 0–1 scale)",
#     0.0, 1.0, 0.45,
#     help="Average clean fuel access over the last 30 days — smooths out short-term changes."
# )

# # --------------------------------------------------------------
# # 🧩 5. Combine All Inputs into Model Feature Order
# # --------------------------------------------------------------
# input_dict = {
#     'temperature_avg': temperature_avg,
#     'rainfall_mm': rainfall_mm,
#     'humidity_avg': humidity_avg,
#     'sin_doy': sin_doy,
#     'cos_doy': cos_doy,
#     'is_weekend': is_weekend,
#     'firewood_demand_m3_lag_1': firewood_lag_1,
#     'firewood_demand_m3_lag_7': firewood_lag_7,
#     'firewood_demand_m3_lag_30': firewood_lag_30,
#     'clean_fuel_access_lag_1': cleanfuel_lag_1,
#     'clean_fuel_access_lag_7': cleanfuel_lag_7,
#     'clean_fuel_access_lag_30': cleanfuel_lag_30,
#     'fw_roll_mean_30': fw_roll_mean_30,
#     'cf_roll_mean_30': cf_roll_mean_30,
# }

# input_df = pd.DataFrame([input_dict])

# # --------------------------------------------------------------
# # 🔢 6. Scale features using the fitted scaler
# # --------------------------------------------------------------
# try:
#     all_features = list(scaler.feature_names_in_)
#     X_scaled = scaler.transform(input_df[all_features])
# except Exception as e:
#     st.error(f"⚠️ Scaling failed: {e}")
#     st.stop()

# # Reshape for LSTM: (1 sample, timesteps=1, features=n)
# X_scaled = np.expand_dims(X_scaled, axis=1)

# # --------------------------------------------------------------
# # 🔮 7. Make Prediction
# # --------------------------------------------------------------
# if st.button("🔮 Predict Energy Demand"):
#     y_pred_scaled = model.predict(X_scaled)
    
#     # inverse-transform (2 targets: firewood_demand_m3, clean_fuel_access)
#     n_total_cols = scaler.n_features_in_
#     tmp = np.zeros((1, n_total_cols))
#     tmp[:, -2:] = y_pred_scaled
#     y_pred_actual = scaler.inverse_transform(tmp)[:, -2:]

#     firewood_pred_m3 = y_pred_actual[0, 0]
#     gas_pred_tons = y_pred_actual[0, 1] * 100  # scaled 0–1 → convert to tons proxy

#     st.success("✅ Prediction complete!")
#     st.metric("🔥 Firewood Demand", f"{firewood_pred_m3:,.2f} m³")
#     st.metric("⛽ Gas (Clean Fuel Access)", f"{gas_pred_tons:,.2f} tons (proxy)")

#     st.markdown(f"""
#     **Interpretation:**
#     - Estimated daily **firewood demand**: `{firewood_pred_m3:,.2f} m³`
#     - Equivalent **clean fuel (gas)**: `{gas_pred_tons:,.2f} tons`
#     """)
