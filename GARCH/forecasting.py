import pandas as pd
import numpy as np
import pickle
from scipy.stats import norm

# =========================
# LOAD DATA + MODEL
# =========================
df = pd.read_csv("../nifty_final_dataset.csv")
df = df.dropna().reset_index(drop=True)

returns = df["target"].values * 100

with open("garch_scratch.pkl", "rb") as f:
    model = pickle.load(f)

omega = model["omega"]
alpha = model["alpha"]
beta  = model["beta"]

# =========================
# COMPUTE LAST VARIANCE
# =========================
T = len(returns)
var = np.zeros(T)
var[0] = np.var(returns)

for t in range(1, T):
    var[t] = omega + alpha * returns[t-1]**2 + beta * var[t-1]

last_var = var[-1]
last_return = returns[-1]

# =========================
# NEXT DAY VARIANCE
# =========================
next_var = omega + alpha * last_return**2 + beta * last_var
next_vol = np.sqrt(next_var)

print("\n===== NEXT DAY FORECAST =====")
print(f"Volatility (σ): {next_vol:.4f}%")

# =========================
# POINT FORECAST (mean ≈ 0)
# =========================
point_forecast = 0

# =========================
# CONFIDENCE INTERVALS
# =========================
conf_levels = [0.90, 0.95, 0.99]

print("\n===== CONFIDENCE INTERVALS =====")

for conf in conf_levels:
    z = norm.ppf((1 + conf) / 2)
    
    lower = point_forecast - z * next_vol
    upper = point_forecast + z * next_vol
    
    print(f"{int(conf*100)}% CI: [{lower:.4f}%, {upper:.4f}%]")

# =========================
# OPTIONAL PRICE FORECAST
# =========================
last_price = df["close"].iloc[-1]

lower_price = last_price * (1 + lower / 100)
upper_price = last_price * (1 + upper / 100)

print("\n===== PRICE RANGE =====")
print(f"Expected Range: {lower_price:.2f} - {upper_price:.2f}")