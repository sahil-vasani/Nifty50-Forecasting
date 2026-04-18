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
scaler = pickle.load(open("scaler.pkl","rb"))
features = pickle.load(open("features.pkl","rb"))
errors = pickle.load(open("calibration_errors.pkl","rb"))

# =========================
# MODEL
# =========================
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, 2)

    def forward(self,x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:,-1,:]

        out = self.fc(x)

        lower = out[:,0]
        upper = lower + torch.abs(out[:,1])

        return lower, upper

model = TransformerModel(len(features))
model.load_state_dict(torch.load("transformer_model.pth", weights_only=True))
model.eval()

# =========================
# DATA
# =========================
df = pd.read_csv("nifty_features.csv")

X = df[features].values
X_latest = scaler.transform(X[-SEQ_LEN:])

X_latest = torch.tensor(X_latest, dtype=torch.float32).unsqueeze(0)

# =========================
# PREDICT
# =========================
with torch.no_grad():
    lower, upper = model(X_latest)

lower = lower.item()
upper = upper.item()

print("\n===== TRANSFORMER PREDICTION =====")

for conf in CONF_LIST:
    q_hat = np.quantile(errors, conf)

    l_adj = lower - q_hat
    u_adj = upper + q_hat

    pred = (l_adj + u_adj)/2

    print(f"\nConfidence: {conf}")
    print(f"Lower: {l_adj:.5f}")
    print(f"Upper: {u_adj:.5f}")

    if pred > 0:
        print("Signal: BUY 📈")
    else:
        print("Signal: SELL 📉")