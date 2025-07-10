# Cell 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

# Cell 2: Load and prepare data
df = pd.read_csv(
    'preprocessed_powerconsumption_hourly.csv',
    parse_dates=['Datetime'],
    dayfirst=True
)
df.set_index('Datetime', inplace=True)
df = df.asfreq('H')

# Cell 3: define features and targets
features = [
    'TimeOfDay',
    'Season',
    'Temperature',
    'Humidity',
    'WindSpeed',
    'GeneralDiffuseFlows',
    'DiffuseFlows'
]
targets = [
    'PowerConsumption_Zone1',
    'PowerConsumption_Zone2',
    'PowerConsumption_Zone3'
]

# Cell 4: compute correlation matrix
corr_matrix = df[features + targets].corr()

# Cell 5: extract and sort correlations for each zone
for zone in targets:
    corr_with_zone = corr_matrix[zone][features]
    corr_sorted = corr_with_zone.abs().sort_values(ascending=False)
    print(f'\nTop correlations for {zone}:')
    print(corr_sorted)

# Cell 6: clean non-numeric and drop NaNs
for col in features + targets:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=features + targets)

# Cell 7: find best (p,d,q) by AIC for each zone
orders = {}
for zone in targets:
    best_aic = np.inf
    best_order = None

    for p in range(4):
        for d in range(3):
            for q in range(4):
                try:
                    m = SARIMAX(
                        df[zone],
                        exog=df[features],
                        order=(p, d, q)
                    )
                    r = m.fit(disp=False)
                    if r.aic < best_aic:
                        best_aic = r.aic
                        best_order = (p, d, q)
                except:
                    pass

    orders[zone] = best_order
    print(f"{zone} best order: {best_order} (AIC={best_aic:.0f})")

# Cell 8: fit SARIMAX for each zone
models = {}
results = {}

for zone, order in orders.items():
    m = SARIMAX(df[zone], exog=df[features], order=order)
    r = m.fit(disp=False)
    models[zone] = m
    results[zone] = r
    print(f"{zone} fitted with order {order}")

# Cell 9: (optional) inspect coefficient p-values
for zone, res in results.items():
    print(f"\n=== {zone} exog p-values ===")
    print(res.summary().tables[1])

# Cell 10: forecast next 24 hours with exogenous inputs
n_steps = 24

# 1. Build a future datetime index
last_ts = df.index[-1]
future_index = pd.date_range(
    start=last_ts + pd.Timedelta(hours=1),
    periods=n_steps,
    freq='H'
)

# 2. Create future exog DataFrame
future_exog = pd.DataFrame(index=future_index)
# cyclical features
future_exog['TimeOfDay'] = future_index.hour
# example mapping for season
future_exog['Season']    = ((future_index.month % 4) + 1)
# repeat last observed values
for col in ['Temperature','Humidity','WindSpeed','GeneralDiffuseFlows','DiffuseFlows']:
    future_exog[col] = df[col].iloc[-n_steps:].values

# 3. Forecast each zone using its fitted result
for zone, res in results.items():
    f    = res.get_forecast(steps=n_steps, exog=future_exog)
    mean = f.predicted_mean
    ci   = f.conf_int()

    plt.figure(figsize=(10,3))
    plt.plot(df[zone].iloc[-168:], label='history')
    plt.plot(future_index, mean, label='forecast')
    plt.fill_between(
        future_index,
        ci.iloc[:,0],
        ci.iloc[:,1],
        alpha=0.3
    )
    plt.title(f"{zone} 24-h Forecast")
    plt.xlabel("Datetime")
    plt.ylabel("Power Consumption (kW)")
    plt.legend()
    plt.show()

# Cell 11: Residual diagnostics
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

for zone, res in results.items():
    resid = res.resid
    resid.plot(figsize=(10,3), title=f'{zone} Residuals')
    plt.show()
    plot_acf(resid, lags=24)
    plt.show()
    plot_pacf(resid, lags=24)
    plt.show()
    lb = acorr_ljungbox(resid, lags=[24], return_df=True)
    print(f'{zone} Ljung–Box p-value:', lb["lb_pvalue"].values[0])

# Cell 12: Hold-out validation and accuracy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

test_size = 168  # last week
train = df.iloc[:-test_size]
test  = df.iloc[-test_size:]

# refit on train only
trained = {}
for zone, order in orders.items():
    m = SARIMAX(train[zone], exog=train[features], order=order)
    trained[zone] = m.fit(disp=False)

# forecast and score
for zone, res in trained.items():
    f = res.get_forecast(steps=test_size, exog=test[features])
    pred = f.predicted_mean
    rmse = np.sqrt(mean_squared_error(test[zone], pred))
    mae  = mean_absolute_error(test[zone], pred)
    mape = np.mean(np.abs((test[zone] - pred) / test[zone])) * 100
    print(f"{zone}  RMSE:{rmse:.2f}, MAE:{mae:.2f}, MAPE:{mape:.2f}%")

# Cell 13: Baseline comparisons
for zone in targets:
    last = train[zone].iloc[-1]
    base = pd.Series(last, index=test.index)
    rmse = np.sqrt(mean_squared_error(test[zone], base))
    print(f"{zone} persistence RMSE: {rmse:.2f}")

for zone in targets:
    pred_sn = test.index.map(lambda t: df[zone].loc[t - pd.Timedelta(hours=24)])
    pred_sn = pd.Series(pred_sn, index=test.index)
    rmse = np.sqrt(mean_squared_error(test[zone], pred_sn))
    print(f"{zone} seasonal-naive RMSE: {rmse:.2f}")

# Cell 14: Drop exogs with p>0.05 and refit
for zone, res in results.items():
    pvals = res.pvalues[features]
    drop  = pvals[pvals > .05].index.tolist()
    keep  = [f for f in features if f not in drop]
    print(f"{zone} drop: {drop}")
    m2 = SARIMAX(df[zone], exog=df[keep], order=orders[zone])
    r2 = m2.fit(disp=False)
    print(f"{zone} new AIC: {r2.aic:.0f}")

# Cell 15: Add a seasonal term (daily) and compare AIC
for zone in targets:
    p,d,q = orders[zone]
    m_seas = SARIMAX(
        df[zone],
        exog=df[features],
        order=(p,d,q),
        seasonal_order=(1,0,1,24)
    )
    r_seas = m_seas.fit(disp=False)
    print(f"{zone} seasonal AIC: {r_seas.aic:.0f}")
