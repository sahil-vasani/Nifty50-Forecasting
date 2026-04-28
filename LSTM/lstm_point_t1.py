import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. Load Data & Select Features
# ==========================================

df = pd.read_csv('../nifty_final_dataset.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
df['target'] = df['log_ret'].rolling(1).mean().shift(-1)
features = ['log_ret', 'rsi', 'momentum', 'vol_5', 'dist_ma_5']
target_col = 'target'

data = df[features + [target_col]].dropna().values

print("=" * 50)
print("DATA SUMMARY")
print("=" * 50)
print(f"Total rows after dropna : {len(data)}")
print(f"Features used           : {features}")
print(f"Target column           : {target_col}")
print()

# ==========================================
# 2. Chronological Train/Test Split (80/20)
# ==========================================

split_ratio = 0.8
split_index = int(len(data) * split_ratio)

train_data = data[:split_index]
test_data  = data[split_index:]

print("=" * 50)
print("TRAIN / TEST SPLIT")
print("=" * 50)
print(f"Train rows : {len(train_data)}")
print(f"Test rows  : {len(test_data)}")
print()

# ==========================================
# 3. Scaling (fit on train only)
# ==========================================

scaler_features = StandardScaler()
scaler_target   = StandardScaler()

train_features_scaled = scaler_features.fit_transform(train_data[:, :-1])
train_target_scaled   = scaler_target.fit_transform(train_data[:, -1].reshape(-1, 1))

test_features_scaled  = scaler_features.transform(test_data[:, :-1])
test_target_scaled    = scaler_target.transform(test_data[:, -1].reshape(-1, 1))

train_scaled = np.hstack((train_features_scaled, train_target_scaled))
test_scaled  = np.hstack((test_features_scaled, test_target_scaled))

# ==========================================
# 4. Sequence Generator
# ==========================================

def create_sequences(dataset, lookback, forecast):
    X, y = [], []
    for i in range(lookback, len(dataset) - forecast + 1):
        X.append(dataset[i - lookback: i, :-1])
        y.append(dataset[i: i + forecast, -1])
    return np.array(X), np.array(y)

# ==========================================
# 5. LSTM Model Definition
# ==========================================

class NiftyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(NiftyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ==========================================
# 6. Grid Search Over Window Sizes
# ==========================================

WINDOW_SIZES  = [10, 20, 30, 45, 60, 90]   # ← change / expand as needed
forecast_days = 5
batch_size    = 32
epochs        = 150
patience      = 20

all_results = []

print("=" * 50)
print("GRID SEARCH OVER WINDOW SIZES")
print("=" * 50)

for window_size in WINDOW_SIZES:

    print()
    print(f"{'─'*50}")
    print(f"  WINDOW SIZE = {window_size} days")
    print(f"{'─'*50}")

    X_train_np, y_train_np = create_sequences(train_scaled, window_size, forecast_days)
    X_test_np,  y_test_np  = create_sequences(test_scaled,  window_size, forecast_days)

    print(f"  Train sequences : {X_train_np.shape}  (samples, timesteps, features)")
    print(f"  Test sequences  : {X_test_np.shape}")

    if len(X_train_np) == 0 or len(X_test_np) == 0:
        print("  >> Skipped — insufficient data for this window size")
        continue

    X_train_full = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_full = torch.tensor(y_train_np, dtype=torch.float32)
    X_test       = torch.tensor(X_test_np,  dtype=torch.float32)
    y_test       = torch.tensor(y_test_np,  dtype=torch.float32)

    val_size   = int(0.1 * len(X_train_full))
    train_size = len(X_train_full) - val_size

    X_train, y_train = X_train_full[:train_size], y_train_full[:train_size]
    X_val,   y_val   = X_train_full[train_size:], y_train_full[train_size:]

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),
                              batch_size=batch_size, shuffle=False)

    model = NiftyLSTM(
        input_size  = X_train.shape[2],
        hidden_size = 64,
        output_size = forecast_days,
        num_layers  = 2,
        dropout     = 0.2
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    best_val_loss    = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    print(f"  Training (max {epochs} epochs, patience={patience}) ...")

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                val_loss += criterion(model(batch_X), batch_y).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:>3}/{epochs} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}  (best val loss: {best_val_loss:.6f})")
                break

    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test).numpy()

    real_predictions = scaler_target.inverse_transform(test_predictions)
    real_y_test      = scaler_target.inverse_transform(y_test_np)

    rmse = np.sqrt(mean_squared_error(real_y_test[:, 0], real_predictions[:, 0]))
    mae  = mean_absolute_error(real_y_test[:, 0], real_predictions[:, 0])
    r2   = r2_score(real_y_test[:, 0], real_predictions[:, 0])

    print(f"  >> Window={window_size:>3} | RMSE={rmse:.5f} | MAE={mae:.5f} | R2={r2:.5f}")

    all_results.append({
        "Window Size": window_size,
        "RMSE"       : rmse,
        "MAE"        : mae,
        "R2"         : r2,
        "real_y_test": real_y_test,
        "real_preds" : real_predictions,
    })

# ==========================================
# 7. Summary Table
# ==========================================

print()
print("=" * 60)
print("FINAL RESULTS TABLE  (sorted by RMSE ascending)")
print("=" * 60)

results_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ("real_y_test", "real_preds")}
    for r in all_results
])
results_df = results_df.sort_values("RMSE").reset_index(drop=True)
results_df.index += 1

pd.set_option('display.float_format', '{:.5f}'.format)
print(results_df.to_string())
print()

best_row = results_df.iloc[0]
print(f"  ★  BEST WINDOW SIZE by RMSE  : {int(best_row['Window Size'])} days")
print(f"     RMSE = {best_row['RMSE']:.5f}  |  MAE = {best_row['MAE']:.5f}  |  R2 = {best_row['R2']:.5f}")
print("=" * 60)

# ==========================================
# 8. Plot: Best Window — Actual vs Predicted
# ==========================================

best_window = int(best_row['Window Size'])
best_result = next(r for r in all_results if r["Window Size"] == best_window)

plt.figure(figsize=(14, 5))
plt.plot(best_result["real_y_test"][:, 0], label='Actual',    color='steelblue', alpha=0.7)
plt.plot(best_result["real_preds"][:, 0],  label='Predicted', color='darkorange', alpha=0.85)
plt.title(f'Best Window={best_window} | Actual vs Predicted Next-Day Returns')
plt.xlabel('Test Set Timeline (Days)')
plt.ylabel('Percentage Change')
plt.legend()
plt.tight_layout()
plt.savefig('best_window_forecast.png')
plt.show()

# ==========================================
# 9. Comparison Bar Chart: RMSE by Window
# ==========================================

plt.figure(figsize=(10, 4))
colors = ['gold' if int(r['Window Size']) == best_window else 'steelblue'
          for _, r in results_df.iterrows()]
plt.bar(results_df['Window Size'].astype(str), results_df['RMSE'], color=colors)
plt.xlabel('Window Size (days)')
plt.ylabel('RMSE')
plt.title('RMSE Comparison Across Window Sizes  (★ = Best)')
plt.tight_layout()
plt.savefig('window_rmse_comparison.png')
plt.show()

print("\nDone. Plots saved: best_window_forecast.png | window_rmse_comparison.png")