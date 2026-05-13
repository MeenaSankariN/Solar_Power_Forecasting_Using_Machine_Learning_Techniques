import streamlit as st
import pandas as pd
import numpy as np
import os

from src.model import run_models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Solar Power Prediction",
    layout="wide"
)

st.title("🌞 Solar Power Prediction App")

st.markdown(
    """
    Simple interactive dashboard for solar power forecasting
    using machine learning and time-series models.
    """
)

# =====================================
# LOAD DATA
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "solar_data.csv")

df = pd.read_csv(data_path)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# =====================================
# TRAIN / TEST SPLIT
# =====================================

train = df[df['Date'] < '2025-01-01']
test = df[df['Date'] >= '2025-01-01']

features = [
    'Solar_Irradiance',
    'Temperature',
    'Specific_Humidity',
    'Wind_Speed'
]

# =====================================
# TRAIN ML MODELS
# =====================================

results, models = run_models(features, train, test)

# =====================================
# LAYOUT
# =====================================

col1, col2 = st.columns([1, 1])

# =====================================
# LEFT SIDE - CONTROLS
# =====================================

with col1:

    st.subheader("⚙️ Controls")

    model_name = st.selectbox(
        "Choose Model",
        list(models.keys()) + ["SARIMAX"]
    )

    use_real = st.checkbox("Use random real dataset sample")

    # ---------------------------------
    # REAL DATA SAMPLE
    # ---------------------------------

    if use_real:

        sample = test.sample(1)

        st.markdown("### 📌 Sample Input")

        st.dataframe(
            sample[
                [
                    'Solar_Irradiance',
                    'Temperature',
                    'Specific_Humidity',
                    'Wind_Speed'
                ]
            ]
        )

        input_data = sample[features].values

    # ---------------------------------
    # MANUAL INPUT
    # ---------------------------------

    else:

        st.markdown("### 🔧 Manual Input")

        irr_data = df['Solar_Irradiance'].dropna()
        temp_data = df['Temperature'].dropna()
        hum_data = df['Specific_Humidity'].dropna()
        wind_data = df['Wind_Speed'].dropna()

        irradiance = st.slider(
            "Solar Irradiance",
            float(irr_data.min()),
            float(irr_data.max()),
            float(irr_data.mean())
        )

        temperature = st.slider(
            "Temperature",
            float(temp_data.min()),
            float(temp_data.max()),
            float(temp_data.mean())
        )

        humidity = st.slider(
            "Specific Humidity",
            float(hum_data.min()),
            float(hum_data.max()),
            float(hum_data.mean())
        )

        wind_speed = st.slider(
            "Wind Speed",
            float(wind_data.min()),
            float(wind_data.max()),
            float(wind_data.mean())
        )

        input_data = np.array([
            [
                irradiance,
                temperature,
                humidity,
                wind_speed
            ]
        ])

# =====================================
# PREDICTION
# =====================================

y_true = test['Solar_Power']

# ---------------------------------
# ML MODELS
# ---------------------------------

if model_name in models:

    model = models[model_name]

    y_pred = model.predict(test[features])

    prediction = model.predict(input_data)[0]

# ---------------------------------
# SARIMAX
# ---------------------------------

else:

    sarimax_model = SARIMAX(
        train['Solar_Power'],
        exog=train[features],
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 365)
    ).fit(disp=False)

    y_pred = sarimax_model.forecast(
        len(test),
        exog=test[features]
    )

    prediction = sarimax_model.forecast(
        1,
        exog=input_data
    )[0]

# =====================================
# METRICS
# =====================================

mae = mean_absolute_error(y_true, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_true, y_pred)
)

# =====================================
# RIGHT SIDE - RESULTS
# =====================================

with col2:

    st.subheader("📊 Prediction Results")

    st.success(
        f"Predicted Solar Power: {prediction:.4f}"
    )

    st.metric(
        "MAE",
        f"{mae:.5f}"
    )

    st.metric(
        "RMSE",
        f"{rmse:.5f}"
    )

    st.markdown("### 📋 Model Comparison")

    st.dataframe(results)

# =====================================
# FOOTER
# =====================================

st.info(
    """
    This application demonstrates machine learning and
    time-series forecasting models for solar power prediction.
    Detailed analysis, EDA, decomposition, and sensitivity
    studies are available in the Jupyter notebook.
    """
)