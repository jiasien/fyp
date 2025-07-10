# Streamlit App for ARIMA Energy Forecast
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# --- Cell 1: Load and prepare data ---
df = pd.read_csv(
    'preprocessed_powerconsumption_hourly.csv',
    parse_dates=['Datetime'],
    dayfirst=True
)
df.set_index('Datetime', inplace=True)
df = df.asfreq('H')

# --- Cell 2: Define features and targets ---
features = ['TimeOfDay','Season','Temperature','Humidity','WindSpeed','GeneralDiffuseFlows','DiffuseFlows']
targets  = ['PowerConsumption_Zone1','PowerConsumption_Zone2','PowerConsumption_Zone3']

# Clean and drop missing
for col in features + targets:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=features + targets, inplace=True)

# --- Cell 3: Auto-select best ARIMA orders by AIC ---
orders = {}
for zone in targets:
    best_aic = np.inf
    best_order = None
    for p in range(4):
        for d in range(3):
            for q in range(4):
                try:
                    model = SARIMAX(df[zone], exog=df[features], order=(p,d,q))
                    res = model.fit(disp=False)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p,d,q)
                except:
                    continue
    orders[zone] = best_order

# --- Cell 4: Fit final SARIMAX models ---
results = {}
for zone, order in orders.items():
    model = SARIMAX(df[zone], exog=df[features], order=order)
    results[zone] = model.fit(disp=False)

# --- Streamlit UI ---
st.title("Hourly Energy Consumption Predictor")
st.write("Use the sidebar to input values for the exogenous variables." )

# Sidebar inputs
st.sidebar.header("Input Exogenous Variables")
input_vals = {}
input_vals['TimeOfDay']         = st.sidebar.number_input("TimeOfDay (0-23)", min_value=0, max_value=23, value=int(df.index[-1].hour))
input_vals['Season']            = st.sidebar.number_input("Season (1-4)",    min_value=1, max_value=4, value=int(((df.index[-1].month % 4) + 1)))
input_vals['Temperature']       = st.sidebar.number_input("Temperature",     value=float(df['Temperature'].iloc[-1]))
input_vals['Humidity']          = st.sidebar.number_input("Humidity",        value=float(df['Humidity'].iloc[-1]))
input_vals['WindSpeed']         = st.sidebar.number_input("WindSpeed",       value=float(df['WindSpeed'].iloc[-1]))
input_vals['GeneralDiffuseFlows'] = st.sidebar.number_input("GeneralDiffuseFlows", value=float(df['GeneralDiffuseFlows'].iloc[-1]))
input_vals['DiffuseFlows']      = st.sidebar.number_input("DiffuseFlows",    value=float(df['DiffuseFlows'].iloc[-1]))

# Predict button
if st.sidebar.button("Predict Consumption"):
    # Create single-row exog DataFrame
    exog_df = pd.DataFrame([input_vals])

    # Forecast one step ahead for each zone
    preds = {}
    for zone, res in results.items():
        f = res.get_forecast(steps=1, exog=exog_df)
        preds[zone] = float(f.predicted_mean.iloc[0])
    
    # Display results
    st.subheader("Predicted Power Consumption")
    st.write(pd.DataFrame.from_dict(preds, orient='index', columns=['Consumption (kW)']))
else:
    st.write("Awaiting input and click on 'Predict Consumption' in sidebar.")
