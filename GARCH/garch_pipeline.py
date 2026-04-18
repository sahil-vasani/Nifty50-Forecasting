import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("nifty_features.csv")

returns = df["target"] * 100   # IMPORTANT: scale (GARCH needs %)

# =========================
# CHECK BASIC STATS
# =========================
print("Mean:", returns.mean())
print("Std:", returns.std())

# =========================
# BUILD GARCH(1,1)
# =========================
model = arch_model(
    returns,
    vol='Garch',
    p=1,
    q=1,
    mean='Zero'   # returns ≈ 0 mean
)

res = model.fit(disp='off')
import pickle

with open("garch_model.pkl", "wb") as f:
    pickle.dump(res, f)
print("\n===== GARCH SUMMARY =====")
print(res.summary())

# =========================
# FORECAST VOLATILITY
# =========================
forecast = res.forecast(horizon=1)

# variance → std
volatility = np.sqrt(forecast.variance.values[-1][0])

print("\n===== NEXT DAY VOLATILITY =====")
print(f"Predicted Volatility: {volatility:.4f}%")

# =========================
# PLOT VOLATILITY
# =========================
cond_vol = res.conditional_volatility

plt.figure(figsize=(10,5))
plt.plot(cond_vol, label="Conditional Volatility")
plt.title("Volatility Over Time (GARCH)")
plt.legend()
plt.show()