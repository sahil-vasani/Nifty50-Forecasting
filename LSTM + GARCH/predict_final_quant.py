import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

SEQ_LEN = 20
CONF_LIST = [0.90, 0.95, 0.99]

# =========================
# LOAD FILES
# =========================
scaler = pickle.load(open("lstm_scaler.pkl","rb"))
features = pickle.load(open("lstm_features.pkl","rb"))
errors = pickle.load(open("lstm_errors.pkl","rb"))
garch_model = pickle.load(open("garch_model.pkl","rb"))

# =========================
# LSTM MODEL
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)

        lower = out[:, 0]
        upper = lower + torch.abs(out[:, 1])

        return lower, upper

model = LSTMModel(len(features))
model.load_state_dict(torch.load("lstm_model.pth", weights_only=True))
model.eval()

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("nifty_features.csv")

X = df[features].values
X_latest = scaler.transform(X[-SEQ_LEN:])
X_latest = torch.tensor(X_latest, dtype=torch.float32).unsqueeze(0)

# =========================
# LSTM PREDICTION (Tube Loss)
# =========================
with torch.no_grad():
    lower, upper = model(X_latest)

lower = lower.item()
upper = upper.item()

# =========================
# GARCH VOLATILITY 🔥
# =========================
forecast = garch_model.forecast(horizon=1)
vol = np.sqrt(forecast.variance.values[-1][0]) / 100  # convert back

print(f"GARCH Volatility: {vol:.5f}")

# =========================
# COMBINE LSTM + GARCH
# =========================
# shrink interval using realistic volatility
center = (lower + upper) / 2

lower = center - vol
upper = center + vol

print("\nAfter GARCH Adjustment:")
print(lower, upper)

# =========================
# CONFORMAL (FINAL STEP)
# =========================
print("\n===== FINAL PREDICTION =====")

for conf in CONF_LIST:

    q_hat = np.quantile(errors, conf)

    l_adj = lower - q_hat
    u_adj = upper + q_hat

    pred = (l_adj + u_adj) / 2

    print(f"\nConfidence: {conf}")
    print(f"Lower: {l_adj:.5f}")
    print(f"Upper: {u_adj:.5f}")

    if pred > 0:
        print("Signal: BUY 📈")
    else:
        print("Signal: SELL 📉")