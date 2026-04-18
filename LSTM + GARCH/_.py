"""
Nifty50 Multi-Horizon Forecasting System
=========================================
Predicts Nifty50 returns for 1 to N days ahead (configurable via DAYS).
Uses LSTM (tube loss) + GARCH volatility + Conformal prediction intervals.
Also identifies strongest and weakest sectors for each forecast horizon.

Usage:
    Set DAYS = 1  → predict next 1 day
    Set DAYS = 3  → predict next 1, 2, 3 days
    Set DAYS = 5  → predict next 1 to 5 days  (max recommended)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG  ← change only this
# ─────────────────────────────────────────────
DAYS       = 5          # forecast horizon: 1 to 5
SEQ_LEN    = 20         # lookback window (must match training)
CONF_LIST  = [0.90, 0.95, 0.99]

# file paths (keep all files in the same folder as this script)
DATA_FILE   = "../nifty_final_dataset.csv"
SCALER_FILE = "lstm_scaler.pkl"
FEAT_FILE   = "lstm_features.pkl"
ERR_FILE    = "lstm_errors.pkl"
GARCH_FILE  = "garch_model.pkl"
MODEL_FILE  = "lstm_model.pth"

SECTORS = ["bank", "it", "pharma", "auto", "fmcg", "metal", "energy"]

# ─────────────────────────────────────────────
# LSTM MODEL  (matches saved weights: fc → [2, 64])
# ─────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc   = nn.Linear(64, 2)           # outputs: [lower_base, width]

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = out[:, -1, :]
        out    = self.fc(out)
        lower  = out[:, 0]
        upper  = lower + torch.abs(out[:, 1])  # tube loss: width always ≥ 0
        return lower, upper


# ─────────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────────
def load_artefacts():
    scaler      = pickle.load(open(SCALER_FILE, "rb"))
    features    = pickle.load(open(FEAT_FILE,   "rb"))
    errors      = np.array(pickle.load(open(ERR_FILE, "rb")), dtype=float)
    garch_model = pickle.load(open(GARCH_FILE,  "rb"))

    model = LSTMModel(len(features))
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    model.eval()

    return scaler, features, errors, garch_model, model


# ─────────────────────────────────────────────
# LSTM INFERENCE  (single-horizon base signal)
# ─────────────────────────────────────────────
def lstm_predict(model, scaler, features, df):
    X        = df[features].values
    X_scaled = scaler.transform(X[-SEQ_LEN:])
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        lower, upper = model(X_tensor)

    return lower.item(), upper.item()


# ─────────────────────────────────────────────
# GARCH VOLATILITY SCALING
# Horizon h volatility ≈ vol_1day × sqrt(h)
# (standard square-root-of-time rule for i.i.d. returns)
# ─────────────────────────────────────────────
def garch_vol_per_horizon(garch_model, max_days):
    forecast  = garch_model.forecast(horizon=max_days)
    var_row   = forecast.variance.values[-1]         # shape: (max_days,)
    vols      = np.sqrt(var_row) / 100               # back to return scale
    return vols                                      # index 0 = day 1


# ─────────────────────────────────────────────
# SECTOR ANALYSIS
# For each horizon, extrapolate sector returns
# using their recent momentum relative to Nifty
# ─────────────────────────────────────────────
def sector_outlook(df, horizon):
    """
    Estimate cumulative sector return over `horizon` days
    using recent average daily returns scaled by horizon.
    """
    recent = df.tail(SEQ_LEN)
    outlook = {}
    for s in SECTORS:
        col = f"{s}_ret"
        if col in df.columns:
            avg_daily = recent[col].mean()
            outlook[s] = avg_daily * horizon   # simple linear projection
    return outlook


# ─────────────────────────────────────────────
# CONFORMAL PREDICTION INTERVALS
# ─────────────────────────────────────────────
def conformal_interval(lower, upper, errors, conf):
    q_hat = np.quantile(errors, conf)
    return lower - q_hat, upper + q_hat


# ─────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────
def separator(char="─", n=55):
    print(char * n)

def signal_label(pred_return):
    return "BUY  📈" if pred_return > 0 else "SELL 📉"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    assert 1 <= DAYS <= 5, "DAYS must be between 1 and 5."

    # ── Load ──────────────────────────────────
    scaler, features, errors, garch_model, model = load_artefacts()
    df = pd.read_csv(DATA_FILE)

    last_close = df["close"].iloc[-1]
    last_date  = df["date"].iloc[-1]

    print()
    separator("═")
    print("  Nifty50 Multi-Horizon Forecasting System")
    separator("═")
    print(f"  Last date   : {last_date}")
    print(f"  Last close  : {last_close:,.2f}")
    print(f"  Horizon     : 1 to {DAYS} day(s)")
    separator("═")

    # ── Base LSTM signal (day-1 interval) ─────
    base_lower, base_upper = lstm_predict(model, scaler, features, df)
    base_center = (base_lower + base_upper) / 2

    # ── GARCH vols for each horizon ───────────
    garch_vols = garch_vol_per_horizon(garch_model, DAYS)

    # ── Forecast loop ─────────────────────────
    for h in range(1, DAYS + 1):
        vol_h = garch_vols[h - 1]

        # Scale LSTM center linearly with horizon (compounding approximation)
        center_h = base_center * h

        # Widen the tube proportionally (sqrt-of-time volatility scaling)
        half_width_h = vol_h  # already horizon-specific from GARCH

        lower_h = center_h - half_width_h
        upper_h = center_h + half_width_h

        # Price bounds
        price_lower_h = last_close * (1 + lower_h)
        price_upper_h = last_close * (1 + upper_h)
        price_center_h = last_close * (1 + center_h)

        print()
        separator()
        print(f"  DAY +{h} FORECAST")
        separator()
        print(f"  GARCH Vol (σ)  : {vol_h:.5f}")
        print(f"  Return (center): {center_h:+.5f}  |  "
              f"Price ≈ {price_center_h:,.2f}")
        print(f"  Raw Tube       : [{lower_h:+.5f}, {upper_h:+.5f}]")
        print(f"  Price Tube     : [{price_lower_h:,.2f}, {price_upper_h:,.2f}]")

        # Conformal intervals
        print()
        print(f"  {'Conf':>6}  {'Lower Ret':>10}  {'Upper Ret':>10}  "
              f"{'Lower Price':>12}  {'Upper Price':>12}  Signal")
        print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*7}")

        for conf in CONF_LIST:
            l_adj, u_adj = conformal_interval(lower_h, upper_h, errors, conf)
            pred_return  = (l_adj + u_adj) / 2
            lp = last_close * (1 + l_adj)
            up = last_close * (1 + u_adj)
            sig = signal_label(pred_return)
            print(f"  {conf:>6.0%}  {l_adj:>+10.5f}  {u_adj:>+10.5f}  "
                  f"  {lp:>10,.2f}    {up:>10,.2f}  {sig}")

        # Sector outlook
        outlook = sector_outlook(df, h)
        if outlook:
            sorted_sectors = sorted(outlook.items(), key=lambda x: x[1], reverse=True)
            top    = sorted_sectors[0]
            bottom = sorted_sectors[-1]
            print()
            print(f"  Sector Outlook (Day +{h}):")
            print(f"    🚀 Strongest : {top[0].upper():8s}  est. {top[1]:+.4f}")
            print(f"    ⬇  Weakest   : {bottom[0].upper():8s}  est. {bottom[1]:+.4f}")
            print("    Full ranking:")
            for name, val in sorted_sectors:
                bar = "▓" * int(abs(val) * 1000)
                direction = "+" if val >= 0 else "-"
                print(f"      {name.upper():8s} {direction}  {abs(val):.5f}  {bar}")

    # ── Summary table ─────────────────────────
    print()
    separator("═")
    print("  SUMMARY  –  Signal at 95% Confidence")
    separator("═")
    print(f"  {'Day':>4}  {'Return':>10}  {'Price ≈':>10}  Signal")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*7}")

    for h in range(1, DAYS + 1):
        vol_h    = garch_vols[h - 1]
        center_h = base_center * h
        lower_h  = center_h - vol_h
        upper_h  = center_h + vol_h
        l_adj, u_adj = conformal_interval(lower_h, upper_h, errors, 0.95)
        pred_r   = (l_adj + u_adj) / 2
        price_p  = last_close * (1 + pred_r)
        sig      = signal_label(pred_r)
        print(f"  {h:>4}  {pred_r:>+10.5f}  {price_p:>10,.2f}  {sig}")

    separator("═")
    print()


if __name__ == "__main__":
    main()