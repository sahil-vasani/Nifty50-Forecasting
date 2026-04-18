import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("nifty_features.csv")

# use target (returns)
series = df["target"]

# =========================
# FUNCTION: CHECK STATIONARITY
# =========================
def check_stationarity(series):

    result = adfuller(series)

    print("\n===== ADF TEST =====")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")

    if result[1] < 0.05:
        print("✅ Data is STATIONARY")
        return True
    else:
        print("❌ Data is NOT stationary")
        return False

# =========================
# STEP 1: CHECK STATIONARITY
# =========================
is_stationary = check_stationarity(series)

# =========================
# STEP 2: DIFFERENCING IF NEEDED
# =========================
d = 0

if not is_stationary:
    print("\nApplying differencing...")

    series = series.diff().dropna()
    d = 1

    # check again
    check_stationarity(series)

# =========================
# STEP 3: TRAIN ARIMA
# =========================
# (p,d,q) → simple start
p = 2
q = 2

model = ARIMA(series, order=(p, d, q))
model_fit = model.fit()

print("\n===== MODEL SUMMARY =====")
print(model_fit.summary())

# =========================
# STEP 4: PREDICTION
# =========================
forecast = model_fit.forecast(steps=1)

print("\n===== NEXT DAY FORECAST =====")
print(f"Predicted Return: {forecast.iloc[0]}")

# =========================
# STEP 5: PLOT
# =========================
plt.figure(figsize=(10,5))
plt.plot(series[-100:], label="Actual")
plt.title("Last 100 Points (Stationary Series)")
plt.legend()
plt.show()