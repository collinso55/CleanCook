import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as KerasMSE
import os
from datetime import datetime, timedelta

# --- 1. CONFIGURATION AND FILE PATHS ---
MODEL_PATH = './data/processed/lstm_fuel_model.h5'
SCALER_PATH = './data/processed/minmax_scaler.pkl'
MASTER_DATA_PATH = './data/fuel_consumption_master_dataset_2010_2024.csv' 

# Constants based on the training data
TARGET_FEATURE_INDEX = 0 # Index of 'woodfuel_production_m3' in feature_cols
SEQUENCE_LENGTH = 7      # The lookback window (7 days)

# --- 2. MODEL AND ARTIFACT LOADING (Cached for performance) ---

@st.cache_resource
def load_artifacts():
    """Loads the model, scaler, and master dataset."""
    
    # Custom objects resolve the 'ValueError: Could not deserialize...' issue
    CUSTOM_OBJECTS = {'loss': MeanSquaredError, 'mse': KerasMSE}
    
    try:
        model = load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
    except Exception as e:
        st.error(f"CRITICAL ERROR: Failed to load LSTM model from {MODEL_PATH}.")
        st.exception(e)
        return None, None, None, None

    try:
        scaler = joblib.load(SCALER_PATH)
        feature_cols = ['woodfuel_production_m3', 'temperature_avg', 'rainfall_mm', 
                        'humidity_avg', 'woodfuel_lag7', 'temp_7d_avg', 'is_dry_season']
    except Exception as e:
        st.error("CRITICAL ERROR: Failed to load scaler or determine feature columns.")
        st.exception(e)
        return model, None, None, None

    if not os.path.exists(MASTER_DATA_PATH):
        st.error(f"CRITICAL ERROR: Master dataset not found at: {MASTER_DATA_PATH}")
        st.error(f"Please ensure 'fuel_consumption_master_dataset_2010_2024.csv' is inside the 'data' folder.")
        return model, scaler, None, feature_cols
        
    try:
        master_df = pd.read_csv(MASTER_DATA_PATH, index_col='date', parse_dates=True)
        # Recreate required features and clean up NaN rows created by lags/rolling for a clean start
        master_df['woodfuel_lag7'] = master_df['woodfuel_production_m3'].shift(7)
        master_df['temp_7d_avg'] = master_df['temperature_avg'].rolling(window=7).mean()
        master_df.dropna(subset=feature_cols, inplace=True) 
        master_df = master_df[feature_cols].copy()
    except Exception as e:
        st.error(f"CRITICAL ERROR: Master dataset loaded but columns are incorrect.")
        st.exception(e)
        return model, scaler, None, feature_cols

    return model, scaler, master_df, feature_cols

# Load all resources at startup
model, scaler, master_df, feature_cols = load_artifacts()

if model is None or scaler is None or master_df is None:
    st.stop()


# --- 3. ITERATIVE FORECASTING LOGIC ---

def inverse_transform_prediction(scaled_pred_value, scaler, num_features):
    """Helper function to unscale the single prediction value."""
    prediction_unscaled_matrix = np.zeros((1, num_features))
    prediction_unscaled_matrix[:, TARGET_FEATURE_INDEX] = scaled_pred_value
    actual_prediction_value = scaler.inverse_transform(prediction_unscaled_matrix)[0, TARGET_FEATURE_INDEX]
    return actual_prediction_value

def simulate_and_predict_series(model, scaler, master_df, target_date):
    """
    Performs multi-step forecasting from the last known date to the target date.
    """
    last_known_date = master_df.index.max()
    
    # 3.1 Initial Setup: Copy the master DF to use for iterative predictions
    forecast_df = master_df.copy()
    num_features = len(master_df.columns)
    
    # The first day we need to predict
    current_date = last_known_date + timedelta(days=1)
    
    # Store the full prediction series for plotting later
    prediction_series = []
    
    # Loop from the first day after training data up to the target date
    while current_date <= target_date:
        
        # 3.2 Prepare Input Sequence (7 days ending the day before current_date)
        end_date = current_date - timedelta(days=1)
        start_date = end_date - timedelta(days=SEQUENCE_LENGTH - 1)
        
        # Get the 7 days (mix of historical and previous predictions)
        input_sequence = forecast_df.loc[start_date:end_date]
        
        if len(input_sequence) != SEQUENCE_LENGTH:
            # This should only happen at the beginning if master_df was too small,
            # but is a safety check.
            return None, "Input sequence is incomplete for the first prediction step."

        # 3.3 Prepare Data: Scale and Reshape for LSTM
        scaled_sequence = scaler.transform(input_sequence.values)
        X_predict = scaled_sequence.reshape(1, SEQUENCE_LENGTH, num_features)
        
        # 3.4 Predict & Unscale
        scaled_prediction = model.predict(X_predict, verbose=0)[0, 0]
        unscaled_prediction = inverse_transform_prediction(scaled_prediction, scaler, num_features)

        # 3.5 Prepare Next Input Row (Crucial for iterative forecasting)
        
        # Use simple persistence/average for future weather features
        last_input_row = input_sequence.iloc[-1].copy()
        
        # 1. Primary Prediction: 'woodfuel_production_m3'
        new_row_values = [unscaled_prediction] 
        
        # 2. Weather/Static Features: Assume persistence (use the last known values)
        new_row_values.extend([
            last_input_row['temperature_avg'], 
            last_input_row['rainfall_mm'], 
            last_input_row['humidity_avg']
        ])
        
        # 3. Lag Features: Calculate new lag features based on the current state of forecast_df
        # woodfuel_lag7: The 'woodfuel_production_m3' value 7 days prior to current_date
        woodfuel_7_days_ago = forecast_df.loc[current_date - timedelta(days=7), 'woodfuel_production_m3']
        
        # temp_7d_avg: The average temperature over the 7 days ending the day before current_date
        # (Since we assume temperature is persistent/fixed, this is simpler)
        temp_7d_avg = input_sequence['temperature_avg'].mean()
        
        new_row_values.extend([
            woodfuel_7_days_ago, 
            temp_7d_avg
        ])

        # 4. Binary Feature: is_dry_season (Assume persistence or simple calendar logic)
        # We'll use a simple calendar approximation for the dry season (Dec-Mar)
        is_dry = 1 if current_date.month in [12, 1, 2, 3] else 0
        new_row_values.append(is_dry)
        
        # Create the new row for the forecast_df
        new_row_df = pd.DataFrame(
            [new_row_values], 
            columns=feature_cols, 
            index=[current_date]
        )
        
        # 3.6 Append the newly predicted row to the series for the next prediction
        forecast_df = pd.concat([forecast_df, new_row_df])
        prediction_series.append((current_date, unscaled_prediction))
        
        current_date += timedelta(days=1) # Move to the next day

    # Extract the final result and the input sequence used to make the final prediction
    final_input_sequence = forecast_df.loc[start_date:end_date]
    final_prediction = prediction_series[-1][1]
    full_prediction_df = pd.DataFrame(
        prediction_series, 
        columns=['date', 'woodfuel_production_m3']
    ).set_index('date')

    return final_prediction, final_input_sequence, full_prediction_df


# --- 4. STREAMLIT UI SETUP (Professional Dark Mode for Readability) ---

st.set_page_config(
    page_title="CleanCook: Wood Fuel Demand Predictor",
    layout="wide"
)

# Custom CSS for a professional, high-contrast dark theme with date input fix
st.markdown(
    """
    <style>
    /* Global Dark Mode Settings: Deep contrast for readability */
    .stApp {
        background-color: #121212; /* Deep Black background */
        color: #FAFAFA; /* Bright White text for main content */
        font-family: 'Inter', sans-serif;
    }
    
    /* Ensure all general text (p, headers, etc.) is readable on the dark background */
    p, label, span, div.stMarkdown, h1, h2, h3, h4, .stText {
        color: #FAFAFA !important;
    }

    /* Sidebar specific styling (Left) */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E; /* Slightly lighter dark gray for separation */
        color: #FAFAFA !important;
    }
    [data-testid="stSidebar"] * {
        color: #FAFAFA !important; 
    }
    
    /* FIX: Date Input Text Color */
    /* Target the input field inside stDateInput and force black text on white background */
    .stDateInput div input[type="text"] {
        color: #121212 !important; /* Black text */
        background-color: #FFFFFF !important; /* White background for the input box */
        border: 1px solid #00BFFF; /* Accent border */
        border-radius: 6px;
    }
    /* Ensure the label text above the date input remains white */
    .stDateInput label {
        color: #FAFAFA !important;
    }


    /* Title and Header styling - Use a vibrant accent color */
    .stTitle {
        color: #00BFFF; /* Electric Blue Accent */
        font-weight: 900;
        text-align: center;
        margin-top: 1rem;
    }
    h2, h3 {
        color: #90CAF9 !important; /* Light blue for secondary headings */
    }

    /* Center the main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Customize Metric boxes (The result cards) */
    [data-testid="stMetric"] {
        background-color: #282828; /* Darker card background */
        border-radius: 12px;
        border: 2px solid #00BFFF; /* Blue accent border */
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5); 
    }
    /* Ensure metric label and value are white/light */
    [data-testid="stMetricLabel"] > div, [data-testid="stMetricValue"] {
        color: #FAFAFA !important;
    }

    /* Better error display (Highly visible, contrasting red) */
    .stAlert {
        border-radius: 8px;
        border-left: 8px solid #FF4D4D; /* Vibrant Red line */
        background-color: #331A1A; /* Dark Red background */
        color: #FFDADA !important; /* Light text for readability */
        font-weight: 700;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    /* Info box in sidebar */
    .stAlert.info {
        background-color: #1A2E33; /* Darker blue-gray info background */
        border-left-color: #00BFFF;
        color: #B3E5FC !important;
    }
    
    /* Styling for primary button (High contrast green accent) */
    .stButton>button {
        background-color: #009933; /* Green accent */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #007329; 
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🔥 CleanCook: Wood Fuel Demand Forecasting")
st.markdown('<p style="text-align: center; color: #90CAF9; font-size: 1.1rem;">LSTM-Based Model for Sustainable Energy Interventions in Cameroon</p>', unsafe_allow_html=True)

st.markdown("---")

# --- 5. SIDEBAR INPUTS ---
st.sidebar.header("📅 Forecast Configuration")

last_known_date = master_df.index.max()
# Limit forecasting to the end of 2025, as requested
max_predict_date = datetime(2025, 12, 31).date() 
default_predict_date = datetime(2025, 1, 30).date() 

# This info box now uses the custom .stAlert.info style
st.sidebar.info(f"Historical data used up to: **{last_known_date.strftime('%Y-%m-%d')}**")

target_date = st.sidebar.date_input(
    "Select Target Date for Prediction",
    value=default_predict_date,
    min_value=last_known_date + timedelta(days=1),
    max_value=max_predict_date,
    help=f"Select a date in 2025. The model will forecast iteratively from {last_known_date.strftime('%Y-%m-%d')}."
)
target_datetime = datetime.combine(target_date, datetime.min.time())


if st.sidebar.button("Generate Multi-Step Forecast", type="primary"):
    with st.spinner(f"Running multi-step forecast from {last_known_date.strftime('%Y-%m-%d')} to {target_date.strftime('%Y-%m-%d')}..."):
        
        # 6. RUN ITERATIVE PREDICTION
        final_prediction, final_input_sequence, full_prediction_df = simulate_and_predict_series(
            model, scaler, master_df, target_datetime
        )
        
        if final_prediction is not None:
            # Store results in session state to persist after the button click
            st.session_state['predicted_value'] = final_prediction
            st.session_state['input_sequence'] = final_input_sequence # Stored as 'input_sequence'
            st.session_state['target_date'] = target_date
            st.session_state['full_prediction_df'] = full_prediction_df
        else:
            # This uses the highly visible custom .stAlert error style
            st.error("Forecast failed due to incomplete data.")

# --- 7. MAIN DASHBOARD OUTPUTS ---

if 'predicted_value' in st.session_state:
    
    predicted_value = st.session_state['predicted_value']
    input_sequence = st.session_state['input_sequence'] # Correctly retrieved as 'input_sequence'
    target_date = st.session_state['target_date']
    full_prediction_df = st.session_state['full_prediction_df']
    
    st.header(f"Final Forecast for {target_date.strftime('%B %d, %Y')}")

    # A. Display Key Result Metrics
    col1, col2 = st.columns([1, 1])
    
    # Contextual Metric: Compare to the average of the last 7 days of the forecast series
    avg_demand_7d = full_prediction_df.iloc[-SEQUENCE_LENGTH-1:-1]['woodfuel_production_m3'].mean()
    delta = predicted_value - avg_demand_7d
    delta_percent = (delta / avg_demand_7d) * 100 if avg_demand_7d else 0
    
    col1.metric(
        label="Predicted Wood Fuel (m³)",
        value=f"{predicted_value:,.0f}",
        delta="Final Estimate",
        delta_color="off"
    )
    
    col2.metric(
        label=f"Trend Change from Prior 7 Days",
        value=f"{delta:+.0f} m³",
        delta=f"{delta_percent:+.1f}%",
        delta_color="inverse" if delta > 0 else "normal"
    )

    # B. Policy Insights/Recommendations
    st.markdown("---")
    st.subheader("💡 Intervention Recommendation")
    
    # Using specific colors for alert boxes that are readable on the dark background
    if predicted_value > 40000:
        st.error(
            "🚨 **HIGH DEMAND WARNING:** Wood fuel demand is critically high. "
            "Immediate interventions such as **LPG/biogas distribution** and **public awareness campaigns** "
            "are recommended to mitigate environmental and health impact."
        )
    elif predicted_value < 30000:
        st.success(
            "✅ **LOW DEMAND PERIOD:** Optimal time for **monitoring, infrastructure maintenance,** "
            "and **scaling up long-term clean fuel access programs**."
        )
    else:
        st.info(
            "🟡 **MODERATE DEMAND:** Focus on **sustainable sourcing education** and **incentivizing solar/electric cooking solutions**."
        )


    # C. Full Multi-Step Forecast Visualization
    st.markdown("---")
    st.subheader(f"Forecast Series ({len(full_prediction_df)} Days)")
    
    # Create combined historical + forecast dataframe for a clean plot
    historical_for_plot = master_df[['woodfuel_production_m3']].copy()
    
    # Merge the last 7 days of historical data and the forecast series
    full_plot_df = pd.concat([
        historical_for_plot.iloc[-SEQUENCE_LENGTH:], 
        full_prediction_df
    ])
    
    # Create series to plot the forecast visually distinct
    full_plot_df['Historical'] = full_plot_df['woodfuel_production_m3'].where(full_plot_df.index <= last_known_date)
    full_plot_df['Forecast'] = full_plot_df['woodfuel_production_m3'].where(full_plot_df.index > last_known_date)
    
    # Chart colors are chosen to contrast well with the dark background
    st.line_chart(
        full_plot_df, 
        y=['Historical', 'Forecast'], 
        color=['#00BFFF', '#FFD700'] # Electric Blue for historical, Gold/Yellow for forecast
    )
    
    st.caption(f"Historical data (Blue) used to generate the iterative forecast series (Gold).")
    
    with st.expander("Details on Final Input Sequence"):
        st.markdown(f"**7-Day Sequence used to predict {target_date.strftime('%Y-%m-%d')}:**")
        # FIX: Changed final_input_sequence to input_sequence
        st.dataframe(input_sequence.style.format("{:.2f}"))
