# app.py

import pandas as pd
import numpy as np
import warnings
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# --- Constants ---
FEATURES = [
    "TimeOfDay",
    "Season",
    "Temperature",
    "Humidity",
    "WindSpeed",
    "GeneralDiffuseFlows",
    "DiffuseFlows",
]
TARGETS = [
    "PowerConsumption_Zone1",
    "PowerConsumption_Zone2",
    "PowerConsumption_Zone3",
]

# --- Load and prepare data ---
@st.cache_resource
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["Datetime"],
        dayfirst=True
    )
    df.set_index("Datetime", inplace=True)
    df = df.asfreq("H")
    # ensure numeric
    for col in FEATURES + TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + TARGETS)
    return df

df = load_data("preprocessed_powerconsumption_hourly.csv")

# --- Train models ---
@st.cache_resource
def train_models(df: pd.DataFrame):
    orders = {}
    results = {}
    for zone in TARGETS:
        best_aic = np.inf
        best_order = None
        for p in range(4):
            for d in range(3):
                for q in range(4):
                    try:
                        m = SARIMAX(
                            df[zone],
                            exog=df[FEATURES],
                            order=(p, d, q)
                        )
                        r = m.fit(disp=False)
                        if r.aic < best_aic:
                            best_aic = r.aic
                            best_order = (p, d, q)
                    except:
                        continue
        orders[zone] = best_order
        # fit final model
        model = SARIMAX(
            df[zone],
            exog=df[FEATURES],
            order=best_order
        )
        results[zone] = model.fit(disp=False)
    return results

with st.spinner("Training models... this may take a minute"):
    models = train_models(df)

# --- Streamlit UI ---
st.title("Hourly Energy Consumption Predictor")
st.write("Enter weather and time values in the sidebar and click Predict.")

# Sidebar inputs
st.sidebar.header("Exogenous Variables")
input_vals = {}
input_vals["TimeOfDay"] = st.sidebar.number_input(
    "Hour of day (0–23)",
    min_value=0, max_value=23,
    value=int(df.index[-1].hour)
)
input_vals["Season"] = st.sidebar.number_input(
    "Season (1–4)",
    min_value=1, max_value=4,
    value=int(((df.index[-1].month % 4) + 1))
)
input_vals["Temperature"] = st.sidebar.number_input(
    "Temperature (°C)",
    value=float(df["Temperature"].iloc[-1])
)
input_vals["Humidity"] = st.sidebar.number_input(
    "Humidity (%)",
    value=float(df["Humidity"].iloc[-1])
)
input_vals["WindSpeed"] = st.sidebar.number_input(
    "Wind Speed (m/s)",
    value=float(df["WindSpeed"].iloc[-1])
)
input_vals["GeneralDiffuseFlows"] = st.sidebar.number_input(
    "General Diffuse Flows",
    value=float(df["GeneralDiffuseFlows"].iloc[-1])
)
input_vals["DiffuseFlows"] = st.sidebar.number_input(
    "Diffuse Flows",
    value=float(df["DiffuseFlows"].iloc[-1])
)

# Predict button
if st.sidebar.button("Predict Consumption"):
    exog_df = pd.DataFrame([input_vals])
    preds = {}
    for zone, res in models.items():
        f = res.get_forecast(steps=1, exog=exog_df)
        preds[zone] = float(f.predicted_mean.iloc[0])
    st.subheader("Predicted Power Consumption (next hour)")
    st.table(pd.DataFrame.from_dict(
        preds, orient="index", columns=["kW"]
    ))
else:
    st.write("Awaiting input. Click Predict in the sidebar.")
