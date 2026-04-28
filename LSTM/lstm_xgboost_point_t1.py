import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv('../nifty_final_dataset.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# ==============================
# TARGET (SMOOTHED)
# ==============================
df['target'] = df['log_ret'].rolling(3).mean().shift(-1)

# ==============================
# FEATURES
# ==============================
df['lag1'] = df['log_ret'].shift(1)
df['lag2'] = df['log_ret'].shift(2)
df['lag3'] = df['log_ret'].shift(3)
df['rolling_mean_5'] = df['log_ret'].rolling(5).mean()
df['rolling_std_5']  = df['log_ret'].rolling(5).std()

features = [
    'log_ret', 'rsi', 'momentum', 'vol_5',
    'dist_ma_5', 'lag1', 'lag2', 'lag3',
    'rolling_mean_5', 'rolling_std_5'
]

df = df.dropna().reset_index(drop=True)

print("=" * 60)
print("DATA SUMMARY")
print("=" * 60)
print(f"Total rows after dropna  : {len(df)}")
print(f"Date range               : {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"Features used ({len(features)})      : {features}")
print()

# ==============================
# SPLIT (LAST 3 MONTHS)
# ==============================
test_days = 63
train_df = df.iloc[:-test_days].reset_index(drop=True)
test_df  = df.iloc[-test_days:].reset_index(drop=True)

print("=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)
print(f"Train rows : {len(train_df)}")
print(f"Test rows  : {len(test_df)}")
print()

X_train_raw = train_df[features].values
y_train_raw = train_df['target'].values.reshape(-1, 1)
X_test_raw  = test_df[features].values
y_test_raw  = test_df['target'].values.reshape(-1, 1)

# ==============================
# SCALING
# ==============================
sx = RobustScaler()
sy = RobustScaler()

X_train = sx.fit_transform(X_train_raw)
X_test  = sx.transform(X_test_raw)
y_train = sy.fit_transform(y_train_raw)
# y_test is left unscaled for metric computation

print("=" * 60)
print("SCALER INFO (RobustScaler — center | scale)")
print("=" * 60)
for i, f in enumerate(features):
    print(f"  {f:20s} | center={sx.center_[i]:.6f} | scale={sx.scale_[i]:.6f}")
print()

# ==============================
# LSTM MODEL (Bidirectional)
# ==============================
class BiLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, 64, batch_first=True, bidirectional=True)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(128, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        o = self.dropout(self.relu(o))
        return self.fc(o[:, -1, :])

# ==============================
# HELPERS
# ==============================
def create_seq(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback: i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def mape(y_true, y_pred):
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

# ==============================
# GRID SEARCH
# ==============================
LOOKBACKS = [20, 25, 30, 40, 50 , 70 , 90]

results = []
graphs  = []

print("=" * 60)
print("GRID SEARCH — LOOKBACK WINDOW SIZES")
print("=" * 60)

for lb in LOOKBACKS:

    print()
    print(f"{'─'*60}")
    print(f"  LOOKBACK = {lb} days")
    print(f"{'─'*60}")

    Xtr, ytr = create_seq(X_train, y_train, lb)
    Xte, yte = create_seq(X_test,  y_test_raw, lb)

    print(f"  Train sequences : {Xtr.shape}  (samples, timesteps, features)")
    print(f"  Test sequences  : {Xte.shape}")

    if len(Xtr) == 0 or len(Xte) == 0:
        print("  >> Skipped — insufficient data")
        continue

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).to(device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32).to(device)

    model   = BiLSTM(Xtr.shape[2]).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.HuberLoss()

    epochs = 100
    print(f"  Training BiLSTM for {epochs} epochs ...")
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xtr_t)
        loss = loss_fn(pred, ytr_t)
        loss.backward()
        opt.step()
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:>3}/{epochs} | Train Loss: {loss.item():.6f}")

    print(f"  Final training loss : {loss.item():.6f}")

    # LSTM predictions (scaled space)
    model.eval()
    with torch.no_grad():
        lstm_pred_scaled = model(Xte_t).cpu().numpy()

    # XGBoost (tabular features aligned to lookback offset)
    X_train_xgb = X_train[lb:]
    X_test_xgb  = X_test[lb:]
    y_train_xgb = y_train[lb:]

    print(f"  XGBoost rows -> train: {X_train_xgb.shape[0]} | test: {X_test_xgb.shape[0]}")

    xgb = XGBRegressor(
        n_estimators  = 400,
        max_depth      = 4,
        learning_rate  = 0.03,
        verbosity      = 0,
        random_state   = 42
    )
    xgb.fit(X_train_xgb, y_train_xgb.ravel())
    xgb_pred_scaled = xgb.predict(X_test_xgb).reshape(-1, 1)

    # Ensemble (weighted average)
    combined_scaled = 0.7 * xgb_pred_scaled + 0.3 * lstm_pred_scaled
    combined        = sy.inverse_transform(combined_scaled)

    y_true_flat = yte.ravel()
    y_pred_flat = combined.ravel()

    r2     = r2_score(y_true_flat, y_pred_flat)
    rmse   = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae    = mean_absolute_error(y_true_flat, y_pred_flat)
    mape_v = mape(y_true_flat, y_pred_flat)

    print(f"  >> Lookback={lb:>2} | R2={r2:.6f} | RMSE={rmse:.6f} | MAE={mae:.6f} | MAPE={mape_v:.3f}%")

    results.append({
        "Lookback" : lb,
        "TrainRows": Xtr.shape[0],
        "TestRows" : Xte.shape[0],
        "R2"       : r2,
        "RMSE"     : rmse,
        "MAE"      : mae,
        "MAPE(%)"  : mape_v,
    })

    graphs.append({
        "lookback"          : lb,
        "dates"             : test_df['date'].values[lb:],
        "y_true"            : y_true_flat,
        "y_pred"            : y_pred_flat,
        "residuals"         : y_true_flat - y_pred_flat,
        "feature_importance": xgb.feature_importances_,
    })

print()
print("=" * 60)
print("GRID SEARCH COMPLETED")
print("=" * 60)

# ==============================
# BIG RESULTS TABLE
# ==============================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='R2', ascending=False).reset_index(drop=True)
results_df.index += 1

pd.set_option('display.max_rows',    None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width',       200)

print()
print("=" * 60)
print("FULL METRICS TABLE  (sorted by R2 descending)")
print("=" * 60)
print(results_df.to_string())
print()

# Highlight best window
best_by_r2   = results_df.iloc[0]
best_by_rmse = results_df.sort_values('RMSE').iloc[0]

print("─" * 60)
print(f"  ★  BEST by R2   : Lookback = {int(best_by_r2['Lookback'])} days"
      f"  |  R2={best_by_r2['R2']:.6f}  RMSE={best_by_r2['RMSE']:.6f}")
print(f"  ★  BEST by RMSE : Lookback = {int(best_by_rmse['Lookback'])} days"
      f"  |  R2={best_by_rmse['R2']:.6f}  RMSE={best_by_rmse['RMSE']:.6f}")
print("─" * 60)

print()
print("SUMMARY  (Lookback | R2 | RMSE | MAE | MAPE%)")
print(results_df[['Lookback', 'R2', 'RMSE', 'MAE', 'MAPE(%)']].to_string())
print()

# ==============================
# PLOT: TOP 3 LOOKBACKS ONLY
# (1) Actual vs Predicted  — most important
# (2) RMSE bar comparison  — overview
# (3) Feature importance   — for best window
# ==============================

top3_lookbacks = results_df.head(3)['Lookback'].tolist()
print(f"Plotting top 3 lookbacks: {top3_lookbacks}")

# --- Plot 1 : Actual vs Predicted for Top-3 in one figure ---
fig, axes = plt.subplots(len(top3_lookbacks), 1,
                         figsize=(14, 4 * len(top3_lookbacks)),
                         sharex=False)
if len(top3_lookbacks) == 1:
    axes = [axes]

for ax, lb in zip(axes, top3_lookbacks):
    g      = next(x for x in graphs if x['lookback'] == lb)
    dates  = pd.to_datetime(g['dates'])
    r2_val = results_df.loc[results_df['Lookback'] == lb, 'R2'].values[0]
    ax.plot(dates, g['y_true'], label='Actual',    linewidth=1.5, color='steelblue')
    ax.plot(dates, g['y_pred'], label='Predicted', linewidth=1.2, color='darkorange', alpha=0.85)
    ax.set_title(f'Lookback={lb} | R²={r2_val:.4f}')
    ax.set_ylabel('Target (log ret smoothed)')
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', rotation=30)

axes[-1].set_xlabel('Date')
plt.suptitle('Actual vs Predicted — Top 3 Lookbacks', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('top3_actual_vs_predicted.png', dpi=120)
plt.show()
print("  Saved: top3_actual_vs_predicted.png")

# --- Plot 2 : RMSE Comparison Bar Chart (all windows) ---
plt.figure(figsize=(10, 4))
bar_colors = ['gold' if lb in top3_lookbacks else 'steelblue'
              for lb in results_df['Lookback']]
plt.bar(results_df['Lookback'].astype(str), results_df['RMSE'], color=bar_colors)
plt.xlabel('Lookback (days)')
plt.ylabel('RMSE')
plt.title('RMSE Comparison Across Lookback Sizes  (★ = Top 3 highlighted)')
plt.tight_layout()
plt.savefig('lookback_rmse_comparison.png', dpi=120)
plt.show()
print("  Saved: lookback_rmse_comparison.png")

# --- Plot 3 : XGBoost Feature Importance for BEST window only ---
best_lb = int(best_by_r2['Lookback'])
best_g  = next(x for x in graphs if x['lookback'] == best_lb)
fi      = best_g['feature_importance']
idx     = np.argsort(fi)[::-1]

plt.figure(figsize=(9, 4))
plt.bar([features[i] for i in idx], fi[idx], color='teal')
plt.title(f'XGBoost Feature Importance — Best Lookback={best_lb}')
plt.xticks(rotation=40, ha='right')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('best_window_feature_importance.png', dpi=120)
plt.show()
print("  Saved: best_window_feature_importance.png")

print()
print("=" * 60)
print("DONE")
print("=" * 60)