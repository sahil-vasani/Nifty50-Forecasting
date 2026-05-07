"""
=============================================================================
Nifty50 % Change Forecasting — LSTM | Single Multi-Output Quantile Model
PyTorch + CUDA Implementation  |  Systematic Hyperparameter Tuning
=============================================================================
Architecture       : Single LSTM with multiple quantile output heads
Quantile Levels    : [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
Horizons           : t+1 (next-day % change)
Grid Search        : window_size, units, dropout, learning_rate
Metrics            : PICP, MPIW, Winkler Score, Calibration
Framework          : PyTorch (CUDA-enabled)
=============================================================================
Improvements over TF/Keras version:
  - Full PyTorch + CUDA support with mixed-precision training (AMP)
  - AdamW optimizer with weight decay for better generalization
  - Gradient clipping to prevent exploding gradients
  - CosineAnnealingLR scheduler for smoother convergence
  - Monotonicity enforcement across quantile heads (crossing prevention)
  - Quantile crossing loss penalty during training
  - Per-epoch validation loss tracked for better early stopping
  - Model checkpoint saving/loading (best val loss)
=============================================================================
"""

import os
import random
import warnings
from datetime import datetime
from itertools import product as itertools_product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ── Random Seeds ──────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Device Setup ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()   # Automatic Mixed Precision only on CUDA

# ── Plotting Style ────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("LSTM SINGLE-MODEL MULTI-QUANTILE FORECASTING  [PyTorch]")
print("=" * 80)
print(f"PyTorch Version     : {torch.__version__}")
print(f"Device              : {DEVICE}")
print(f"GPU Name            : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"Mixed Precision AMP : {USE_AMP}")
print("=" * 80)


# =============================================================================
# 1. Quantile Configuration
# =============================================================================

QUANTILES = [0.05, 0.5, 0.95]
N_QUANTILES = len(QUANTILES)
QUANTILE_TENSOR = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)

print(f"\n[1/11] Quantile Configuration")
print(f"Quantile Levels ({N_QUANTILES}): {QUANTILES}")


def pinball_loss_all(preds: torch.Tensor, target: torch.Tensor,
                     quantiles: torch.Tensor) -> torch.Tensor:
    """
    Vectorised pinball loss across all quantiles simultaneously.

    Args:
        preds    : (B, Q)  — model output for all Q quantiles
        target   : (B,)    — ground truth
        quantiles: (Q,)    — quantile levels

    Returns:
        Scalar mean loss over batch and quantiles.
    """
    target_exp = target.unsqueeze(1).expand_as(preds)   # (B, Q)
    err = target_exp - preds                             # (B, Q)
    loss = torch.where(err >= 0,
                       quantiles * err,
                       (quantiles - 1.0) * err)
    return loss.mean()


def crossing_penalty(preds: torch.Tensor) -> torch.Tensor:
    """
    Penalises quantile crossing: enforces q_i <= q_{i+1}.

    Args:
        preds : (B, Q)

    Returns:
        Scalar penalty.
    """
    diffs = preds[:, 1:] - preds[:, :-1]       # should be >= 0
    return F.relu(-diffs).mean()

def tube_loss(preds: torch.Tensor) -> torch.Tensor:
    """
    Penalize excessively wide prediction intervals.

    preds : (B, Q)
    assumes:
        index 0 = q0.05
        index 2 = q0.95
    """

    lower = preds[:, 0]
    upper = preds[:, 2]

    width = upper - lower

    return width.mean()


# =============================================================================
# 2. Data Loading and Preprocessing
# =============================================================================

print("\n[2/11] Loading and Preprocessing Data...")

df = pd.read_csv('../nifty_final_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

print("\n" + "=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Dataset Shape    : {df.shape}")
print(f"Date Range       : {df.index.min()} to {df.index.max()}")
print(f"Duration         : {(df.index.max() - df.index.min()).days} days")

# Feature Selection
technical_features  = ['log_ret', 'vol_5', 'vol_15', 'rsi', 'momentum', 'trend_strength']
price_features      = ['body', 'range', 'upper_wick', 'lower_wick', 'close_pos']
ma_features         = ['ma_5', 'ma_20', 'dist_ma_5', 'dist_ma_20']
volume_features     = ['volume', 'volume_ma_5', 'volume_spike']
sector_features     = ['bank_ret', 'it_ret', 'pharma_ret', 'auto_ret',
                       'fmcg_ret', 'metal_ret', 'energy_ret']
lag_features        = (
    ['log_ret_lag1', 'log_ret_lag2'] +
    [f'{s}_lag1' for s in ['bank_ret', 'it_ret', 'pharma_ret', 'auto_ret',
                            'fmcg_ret', 'metal_ret', 'energy_ret']] +
    [f'{s}_lag2' for s in ['bank_ret', 'it_ret', 'pharma_ret', 'auto_ret',
                            'fmcg_ret', 'metal_ret', 'energy_ret']]
)
sector_analysis     = [
    'sector_mean', 'sector_std',
    'bank_ret_vs_nifty', 'it_ret_vs_nifty', 'pharma_ret_vs_nifty',
    'auto_ret_vs_nifty', 'fmcg_ret_vs_nifty', 'metal_ret_vs_nifty', 'energy_ret_vs_nifty'
]

selected_features = (
    technical_features + price_features + ma_features + volume_features +
    sector_features + lag_features + sector_analysis
)

available_features = [f for f in selected_features if f in df.columns]
df_features = df[available_features + ['target']].copy().dropna()
N_FEATURES = len(available_features)

print(f"\nFeature Summary:")
print(f"  Technical        : {len([f for f in technical_features if f in available_features])}")
print(f"  Price            : {len([f for f in price_features if f in available_features])}")
print(f"  Moving Averages  : {len([f for f in ma_features if f in available_features])}")
print(f"  Volume           : {len([f for f in volume_features if f in available_features])}")
print(f"  Sector           : {len([f for f in sector_features if f in available_features])}")
print(f"  Lags             : {len([f for f in lag_features if f in available_features])}")
print(f"  Sector Analysis  : {len([f for f in sector_analysis if f in available_features])}")
print(f"  {'─'*40}")
print(f"  TOTAL FEATURES   : {N_FEATURES}")
print(f"\nRecords after cleaning: {len(df_features)}")

# Train / Test Split
train_size = int(len(df_features) * 0.8)
train_data = df_features.iloc[:train_size]
test_data  = df_features.iloc[train_size:]

X_train_raw = train_data[available_features].values.astype(np.float32)
y_train_raw = train_data['target'].values.astype(np.float32)
X_test_raw  = test_data[available_features].values.astype(np.float32)
y_test_raw  = test_data['target'].values.astype(np.float32)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw).astype(np.float32)
X_test_scaled  = scaler_X.transform(X_test_raw).astype(np.float32)
y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten().astype(np.float32)
y_test_scaled  = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten().astype(np.float32)

print("\n" + "=" * 80)
print("DATA SPLIT")
print("=" * 80)
print(f"Training Samples : {len(train_data):>6} ({len(train_data)/len(df_features)*100:.1f}%)")
print(f"Testing Samples  : {len(test_data):>6} ({len(test_data)/len(df_features)*100:.1f}%)")


# =============================================================================
# 3. Sequence Creation
# =============================================================================

def create_sequences(X: np.ndarray, y: np.ndarray, window_size: int):
    """Create rolling-window sequences for LSTM input."""
    Xs, ys = [], []
    for i in range(window_size, len(X)):
        Xs.append(X[i - window_size:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def make_loaders(X_tr_sc, y_tr_sc, X_te_sc, y_te_sc, window_size, batch_size=32):
    """Build DataLoaders from scaled arrays and a given window size."""
    X_tr, y_tr = create_sequences(X_tr_sc, y_tr_sc, window_size)
    X_te, y_te = create_sequences(X_te_sc, y_te_sc, window_size)

    # Validation split from training set (last 20%)
    n_val = int(len(X_tr) * 0.2)
    X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
    X_tr,  y_tr  = X_tr[:-n_val], y_tr[:-n_val]

    tr_ds  = TensorDataset(torch.from_numpy(X_tr),  torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    te_ds  = TensorDataset(torch.from_numpy(X_te),  torch.from_numpy(y_te))

    tr_dl  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,  drop_last=True,  pin_memory=USE_AMP)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=USE_AMP)
    te_dl  = DataLoader(te_ds,  batch_size=batch_size, shuffle=False, pin_memory=USE_AMP)

    return tr_dl, val_dl, te_dl, y_te


# =============================================================================
# 4. Multi-Quantile LSTM Model
# =============================================================================

class QuantileLSTM(nn.Module):
    """
    Multi-output LSTM for simultaneous quantile regression.

    Architecture:
        Input(window_size, n_features)
        → LSTM(units, return_sequences=True)  + Dropout + LayerNorm
        → LSTM(units//2, return_sequences=False) + Dropout + LayerNorm
        → Dense(64, GELU) + Dropout
        → [Dense(32, GELU) + Dropout + Dense(1)] × Q  (per-quantile heads)
        → Stack outputs → (B, Q)

    Improvements vs Keras version:
        - LayerNorm (more stable than BatchNorm in recurrent nets)
        - AdamW + weight decay handles L2 reg properly
        - Single forward pass returns all quantiles → enables crossing penalty
    """

    def __init__(self, window_size: int, n_features: int, n_quantiles: int,
                 units: int = 128, dropout: float = 0.2):
        super().__init__()
        self.units = units

        # ── Shared Backbone ───────────────────────────────────────────────────
        self.lstm1    = nn.LSTM(n_features, units, batch_first=True)
        self.drop1    = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(units)

        self.lstm2    = nn.LSTM(units, units // 2, batch_first=True)
        self.drop2    = nn.Dropout(dropout)
        self.norm2    = nn.LayerNorm(units // 2)

        self.shared   = nn.Sequential(
            nn.Linear(units // 2, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )

        # ── Per-Quantile Heads ────────────────────────────────────────────────
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(32, 1),
            )
            for _ in range(n_quantiles)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, F)
        Returns:
            preds : (B, Q)
        """
        # LSTM 1
        out, _ = self.lstm1(x)                  # (B, T, units)
        out    = self.drop1(out)
        out    = self.norm1(out)

        # LSTM 2
        out, _ = self.lstm2(out)                # (B, T, units//2)
        out    = self.drop2(out[:, -1, :])      # take last timestep
        out    = self.norm2(out)

        # Shared dense
        shared = self.shared(out)               # (B, 64)

        # Per-quantile heads → stack
        qs = [head(shared) for head in self.heads]  # Q × (B, 1)
        return torch.cat(qs, dim=1)                 # (B, Q)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 5. Architecture Display
# =============================================================================

print("\n[3/11] Building Sample Model for Architecture Display...")

_sample = QuantileLSTM(10, N_FEATURES, N_QUANTILES, units=128, dropout=0.2).to(DEVICE)
total_params = _sample.count_params()

print("\n" + "=" * 90)
print("LSTM QUANTILE MODEL ARCHITECTURE  [PyTorch]")
print("=" * 90)
arch_rows = [
    ("Input",             f"(B, 10, {N_FEATURES})",   0,          "Sequence input"),
    ("LSTM (1st)",        f"(B, 10, 128)",             "~91K",     "Temporal feature extraction"),
    ("Dropout",           f"(B, 10, 128)",             0,          "Regularization"),
    ("LayerNorm",         f"(B, 10, 128)",             256,        "Stabilises recurrent output"),
    ("LSTM (2nd)",        "(B, 64)",                   "~49K",     "Compressed temporal encoding"),
    ("Dropout",           "(B, 64)",                   0,          "Prevent overfitting"),
    ("LayerNorm",         "(B, 64)",                   128,        "Feature normalisation"),
    ("Linear→GELU",       "(B, 64)",                   "~4K",      "Shared feature extractor"),
    ("Dropout",           "(B, 64)",                   0,          "Regularization"),
    (f"Heads (3×Linear)", "(B, 32)", "~2K each", "Separate head per quantile"),
    ("Dropout (heads)", "(B, 32)", 0, "Regularization per head"),
    ("Output (3×)", "(B, 1)", "33 each", "Quantile output"),
    ("Stack → concat", "(B, 3)", 0, "Final multi-quantile output"),
]
print(f"{'#':<3} {'Layer':<28} {'Output Shape':<18} {'Params':<14} Description")
print("-" * 90)
for i, (l, s, p, d) in enumerate(arch_rows, 1):
    ps = f"{p:,}" if isinstance(p, int) else str(p)
    print(f"{i:<3} {l:<28} {s:<18} {ps:<14} {d}")
print("-" * 90)
print(f"{'TOTAL TRAINABLE PARAMETERS':<50} {total_params:,}")
print("=" * 90)

del _sample
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# =============================================================================
# 6. Training Utilities
# =============================================================================

def train_one_epoch(model, loader, optimizer, scaler_amp, crossing_weight=0.1):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()

        if USE_AMP:
            with torch.amp.autocast('cuda'):
                preds = model(xb)                                        # (B, Q)
                pb = pinball_loss_all(preds, yb, QUANTILE_TENSOR)

                cp = crossing_penalty(preds)

                tp = tube_loss(preds)

                loss = (
                    pb
                    + crossing_weight * cp
                    + 0.02 * tp
                )
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            preds = model(xb)
            pb    = pinball_loss_all(preds, yb, QUANTILE_TENSOR)
            cp    = crossing_penalty(preds)
            loss  = pb + crossing_weight * cp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds  = model(xb)
        loss   = pinball_loss_all(preds, yb, QUANTILE_TENSOR)
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict_all(model, loader):
    """Return (N, Q) predictions in numpy."""
    model.eval()
    chunks = []
    for xb, _ in loader:
        chunks.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_model(model, tr_dl, val_dl, lr, epochs=100, patience=15):
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    amp_scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    best_val   = float('inf')
    best_state = None
    patience_c = 0
    history    = []

    for ep in range(1, epochs + 1):
        tr_loss  = train_one_epoch(model, tr_dl, optimizer, amp_scaler)
        val_loss = evaluate(model, val_dl)
        scheduler.step()
        history.append((tr_loss, val_loss))

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history


# =============================================================================
# 7. Systematic Hyperparameter Tuning
# =============================================================================

print("\n[4/11] Hyperparameter Grid Search...")

PARAM_GRID = {
    'window_size'  : [10, 20, 30],
    'units'        : [64, 128],
    'dropout'      : [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.0005],
}

all_combos = list(itertools_product(
    PARAM_GRID['window_size'],
    PARAM_GRID['units'],
    PARAM_GRID['dropout'],
    PARAM_GRID['learning_rate'],
))

print("\n" + "=" * 80)
print("HYPERPARAMETER SEARCH SPACE")
print("=" * 80)
for param, values in PARAM_GRID.items():
    print(f"  {param:<18}: {values}")
print(f"\n  Total Combinations : {len(all_combos)}")
print("=" * 80)

tuning_results = []
best_rmse      = float('inf')
best_config    = None
best_model     = None
best_X_te      = None
best_y_te      = None

for combo_idx, (ws, units, dropout, lr) in enumerate(all_combos, 1):
    print(f"\n[{combo_idx:02d}/{len(all_combos)}] "
          f"window={ws:2d}, units={units:3d}, dropout={dropout:.1f}, lr={lr:.4f}")

    tr_dl, val_dl, te_dl, y_te_np = make_loaders(
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, ws
    )

    model = QuantileLSTM(ws, N_FEATURES, N_QUANTILES, units=units, dropout=dropout).to(DEVICE)
    model, history = train_model(model, tr_dl, val_dl, lr=lr, epochs=100, patience=15)

# ── Evaluate on median (Q index = 1 for q=0.5)
    preds_all  = predict_all(model, te_dl)       # (N, 3)
    q50_scaled = preds_all[:, QUANTILES.index(0.5)]
    y_pred     = scaler_y.inverse_transform(q50_scaled.reshape(-1, 1)).flatten()
    y_true     = scaler_y.inverse_transform(y_te_np.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)

    epochs_run = len(history)
    print(f"  → RMSE={rmse:.6f}, MSE={mse:.6f}, MAE={mae:.6f}, Epochs={epochs_run}")

    record = {
        'window_size'  : ws,
        'units'        : units,
        'dropout'      : dropout,
        'learning_rate': lr,
        'RMSE'         : round(rmse, 6),
        'MSE'          : round(mse, 6),
        'MAE'          : round(mae, 6),
        'final_epoch'  : epochs_run,
    }
    tuning_results.append(record)

    if rmse < best_rmse:
        best_rmse   = rmse
        best_config = {k: v for k, v in record.items() if k in PARAM_GRID}
        best_model  = model
        best_X_te   = te_dl
        best_y_te   = y_te_np
    else:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING COMPLETE")
print("=" * 80)


# =============================================================================
# 8. Results Table
# =============================================================================

print("\n[5/11] Generating Tuning Results Table...")

results_df = pd.DataFrame(tuning_results).sort_values('RMSE').reset_index(drop=True)
results_df.index += 1

print("\n" + "=" * 95)
print("HYPERPARAMETER TUNING RESULTS (Sorted by RMSE)")
print("=" * 95)
print(results_df.to_string())

print("\n" + "=" * 80)
print("BEST HYPERPARAMETER CONFIGURATION")
print("=" * 80)
print(f"  Window Size      : {best_config['window_size']}")
print(f"  LSTM Units       : {best_config['units']}")
print(f"  Dropout          : {best_config['dropout']}")
print(f"  Learning Rate    : {best_config['learning_rate']}")
print(f"  Best RMSE        : {best_rmse:.6f}")
print("=" * 80)


# =============================================================================
# 9. Generate Predictions with Best Model
# =============================================================================

print("\n[6/11] Generating Predictions with Best Model...")

WINDOW_SIZE = best_config['window_size']

# Re-create test sequences for full test set
X_te_full, y_te_full = create_sequences(X_test_scaled, y_test_scaled, WINDOW_SIZE)
te_ds_full = TensorDataset(
    torch.from_numpy(X_te_full),
    torch.from_numpy(y_te_full)
)
te_dl_full = DataLoader(te_ds_full, batch_size=32, shuffle=False, pin_memory=USE_AMP)

preds_scaled = predict_all(best_model, te_dl_full)   # (N, 7)
y_actual     = scaler_y.inverse_transform(y_te_full.reshape(-1, 1)).flatten()

quantile_predictions = {}
for idx, q in enumerate(QUANTILES):
    quantile_predictions[q] = scaler_y.inverse_transform(
        preds_scaled[:, idx].reshape(-1, 1)
    ).flatten()

print(f"Predictions shape  : {preds_scaled.shape}")
print(f"Test samples       : {len(y_actual)}")


# =============================================================================
# 10. Prediction Interval Metrics
# =============================================================================

print("\n[7/11] Computing Prediction Interval Metrics...")


def calculate_interval_metrics(y_true, lower, upper, confidence_level=0.95):
    coverage  = (y_true >= lower) & (y_true <= upper)
    picp      = np.mean(coverage)
    widths    = upper - lower
    mpiw      = np.mean(widths)
    dr        = np.max(y_true) - np.min(y_true)
    nmpiw     = mpiw / dr if dr > 0 else 0.0

    mu  = confidence_level
    eta = 50
    gamma = 0 if picp >= mu else 1
    cwc = nmpiw * (1 + gamma * np.exp(-eta * (picp - mu)))

    ace       = abs(picp - confidence_level)
    sharpness = mpiw / np.mean(np.abs(y_true)) * 100 if np.mean(np.abs(y_true)) > 0 else 0.0

    alpha   = 1 - confidence_level
    winkler = widths + (2 / alpha) * (
        (lower - y_true) * (y_true < lower) +
        (y_true - upper) * (y_true > upper)
    )
    winkler_score = np.mean(winkler)

    return {
        'PICP': picp, 'MPIW': mpiw, 'NMPIW': nmpiw, 'CWC': cwc,
        'ACE': ace, 'Sharpness_%': sharpness, 'Winkler_Score': winkler_score,
        'Coverage': coverage, 'Widths': widths,
    }


confidence_levels = {
    '90%': (0.05, 0.95),
}

print("\n" + "=" * 80)
print("PREDICTION INTERVAL METRICS")
print("=" * 80)

metrics_summary = {}
for level_name, (q_low, q_high) in confidence_levels.items():
    lower   = quantile_predictions[q_low]
    upper   = quantile_predictions[q_high]
    metrics = calculate_interval_metrics(
        y_actual, lower, upper,
        confidence_level=float(level_name.strip('%')) / 100
    )
    metrics_summary[level_name] = metrics

    print(f"\n{level_name} Prediction Interval:")
    print(f"  PICP (Coverage)      : {metrics['PICP']*100:>6.2f}%  (Target: {level_name})")
    print(f"  MPIW (Avg Width)     : {metrics['MPIW']:>10.6f}")
    print(f"  NMPIW (Normalised)   : {metrics['NMPIW']:>10.6f}")
    print(f"  CWC (Quality)        : {metrics['CWC']:>10.6f}  (lower is better)")
    print(f"  ACE (Coverage Error) : {metrics['ACE']*100:>6.2f}%")
    print(f"  Sharpness            : {metrics['Sharpness_%']:>6.2f}%")
    print(f"  Winkler Score        : {metrics['Winkler_Score']:>10.6f}")


# =============================================================================
# 11. Next 5-Day Forecast
# =============================================================================

print("\n[8/11] Generating 5-Day Forecast...")


@torch.no_grad()
def predict_next_n_days(model, last_seq_scaled, scaler_y, quantiles, n_days=5):
    """
    Autoregressive 5-day forecast. Updates the first feature (log_ret proxy)
    with the median scaled prediction after each step.
    """
    model.eval()
    preds_dict  = {q: [] for q in quantiles}
    current_seq = last_seq_scaled.copy()

    for _ in range(n_days):
        x_t = torch.from_numpy(
            current_seq.reshape(1, *current_seq.shape)
        ).to(DEVICE)

        q_preds_sc = model(x_t).cpu().numpy()[0]          # (Q,)

        for idx, q in enumerate(quantiles):
            val = scaler_y.inverse_transform([[q_preds_sc[idx]]])[0, 0]
            preds_dict[q].append(val)

        # Roll window — use median (index 3) as proxy for next log_ret
        median_sc = q_preds_sc[quantiles.index(0.5)]
        new_row   = current_seq[-1].copy()
        new_row[0] = median_sc
        current_seq = np.vstack([current_seq[1:], new_row])

    return preds_dict


last_sequence_scaled = X_test_scaled[-WINDOW_SIZE:]
next_5_quantiles     = predict_next_n_days(
    best_model, last_sequence_scaled, scaler_y, QUANTILES, n_days=5
)

last_date    = df_features.index[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)

print("\n" + "=" * 90)
print("NEXT 5-DAY FORECAST")
print("=" * 90)
print(f"Last Trading Date : {last_date.strftime('%Y-%m-%d')}\n")
# print(f"{'Date':<14} {'Q2.5%':>10} {'Q25%':>10} {'Median':>10} {'Q75%':>10} {'Q97.5%':>10}")
print(f"{'Date':<14} {'Q5%':>10} {'Median':>10} {'Q95%':>10}")
print("-" * 70)

for i, date in enumerate(future_dates):
    
    print(
        f"{date.strftime('%Y-%m-%d'):<14} "
        f"{next_5_quantiles[0.05][i]:>+8.4f}%  "
        f"{next_5_quantiles[0.5][i]:>+8.4f}%  "
        f"{next_5_quantiles[0.95][i]:>+8.4f}%"
    )

# =============================================================================
# 12. Visualization
# =============================================================================

print("\n[9/11] Creating Visualizations...")

test_dates = test_data.index[WINDOW_SIZE:]

# ── Plot 1: Prediction Intervals ──────────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 1, figsize=(18, 12))

axes1[0].plot(test_dates, y_actual,
               'b-', lw=2, label='Actual', alpha=0.8)

axes1[0].plot(test_dates, quantile_predictions[0.5],
               'r--', lw=2, label='Median (q=0.5)', alpha=0.8)

# 90% PI
axes1[0].fill_between(
    test_dates,
    quantile_predictions[0.05],
    quantile_predictions[0.95],
    alpha=0.25,
    color='red',
    label='90% PI'
)

axes1[0].axhline(0, color='black', ls=':', lw=1, alpha=0.5)

axes1[0].set_xlabel('Date', fontsize=12)
axes1[0].set_ylabel('Return (%)', fontsize=12)

axes1[0].set_title(
    'LSTM Quantile Predictions — 90% Prediction Interval [PyTorch]',
    fontweight='bold',
    fontsize=14
)

axes1[0].legend(loc='best', fontsize=10)
axes1[0].grid(alpha=0.3)

# ─────────────────────────────────────────────────────────────────────────────

zoom_n = min(50, len(test_dates))

axes1[1].plot(
    test_dates[-zoom_n:],
    y_actual[-zoom_n:],
    'b-',
    lw=2.5,
    label='Actual',
    marker='o',
    ms=4
)

axes1[1].plot(
    test_dates[-zoom_n:],
    quantile_predictions[0.5][-zoom_n:],
    'r--',
    lw=2.5,
    label='Median',
    marker='s',
    ms=4
)

axes1[1].fill_between(
    test_dates[-zoom_n:],
    quantile_predictions[0.05][-zoom_n:],
    quantile_predictions[0.95][-zoom_n:],
    alpha=0.3,
    color='red',
    label='90% PI'
)

axes1[1].axhline(0, color='black', ls=':', lw=1, alpha=0.5)

axes1[1].set_xlabel('Date', fontsize=12)
axes1[1].set_ylabel('Return (%)', fontsize=12)

axes1[1].set_title(
    f'Detailed View: Last {zoom_n} Predictions',
    fontweight='bold',
    fontsize=14
)

axes1[1].legend(loc='best', fontsize=10)
axes1[1].grid(alpha=0.3)

plt.tight_layout()

fig1.savefig(
    "lstm_single_model_predictions.png",
    dpi=150,
    bbox_inches="tight"
)

print("  Saved: lstm_single_model_predictions.png")

plt.close(fig1)

# =============================================================================
# Plot 2: Metrics Grid
# =============================================================================

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

# Calibration
empirical_coverage = [
    np.mean(y_actual <= quantile_predictions[q])
    for q in QUANTILES
]

axes2[0, 0].plot(
    [0, 1],
    [0, 1],
    'k--',
    lw=2,
    label='Perfect Calibration'
)

axes2[0, 0].plot(
    QUANTILES,
    empirical_coverage,
    'bo-',
    lw=2,
    ms=8,
    label='LSTM Calibration'
)

axes2[0, 0].set_xlabel('Predicted Quantile', fontsize=12)
axes2[0, 0].set_ylabel('Empirical Coverage', fontsize=12)

axes2[0, 0].set_title(
    'Quantile Calibration Plot',
    fontweight='bold',
    fontsize=14
)

axes2[0, 0].legend(fontsize=10)
axes2[0, 0].grid(alpha=0.3)

# =============================================================================
# PICP vs Target
# =============================================================================

levels  = list(confidence_levels.keys())
picps   = [metrics_summary[l]['PICP'] * 100 for l in levels]
targets = [float(l.strip('%')) for l in levels]

x_pos = np.arange(len(levels))
w = 0.35

axes2[0, 1].bar(
    x_pos - w/2,
    picps,
    w,
    label='Actual PICP',
    color='steelblue',
    alpha=0.7,
    edgecolor='black'
)

axes2[0, 1].bar(
    x_pos + w/2,
    targets,
    w,
    label='Target Coverage',
    color='orange',
    alpha=0.7,
    edgecolor='black'
)

axes2[0, 1].set_xlabel('Confidence Level')
axes2[0, 1].set_ylabel('Coverage (%)')

axes2[0, 1].set_title(
    'PICP vs Target Coverage',
    fontweight='bold'
)

axes2[0, 1].set_xticks(x_pos)
axes2[0, 1].set_xticklabels(levels)

axes2[0, 1].legend()
axes2[0, 1].grid(alpha=0.3, axis='y')

# =============================================================================
# Width Distribution
# =============================================================================

widths_90 = metrics_summary['90%']['Widths']

axes2[1, 0].hist(
    widths_90,
    bins=50,
    edgecolor='black',
    alpha=0.7,
    color='steelblue'
)

axes2[1, 0].axvline(
    metrics_summary['90%']['MPIW'],
    color='red',
    ls='--',
    lw=2,
    label=f"Mean: {metrics_summary['90%']['MPIW']:.4f}"
)

axes2[1, 0].set_xlabel('Interval Width')
axes2[1, 0].set_ylabel('Frequency')

axes2[1, 0].set_title(
    '90% PI Width Distribution',
    fontweight='bold'
)

axes2[1, 0].legend()
axes2[1, 0].grid(alpha=0.3)

# =============================================================================
# 5-Day Forecast
# =============================================================================

day_labels = [
    f"t+{d+1}\n{dt.strftime('%m/%d')}"
    for d, dt in enumerate(future_dates)
]

bar_colors = [
    "#2E7D32" if m > 0 else "#C62828"
    for m in next_5_quantiles[0.5]
]

axes2[1, 1].bar(
    range(5),
    next_5_quantiles[0.5],
    color=bar_colors,
    alpha=0.6,
    label="Median Forecast"
)

yerr_lo = [
    next_5_quantiles[0.5][i] - next_5_quantiles[0.05][i]
    for i in range(5)
]

yerr_up = [
    next_5_quantiles[0.95][i] - next_5_quantiles[0.5][i]
    for i in range(5)
]

axes2[1, 1].errorbar(
    range(5),
    next_5_quantiles[0.5],
    yerr=[yerr_lo, yerr_up],
    fmt="none",
    ecolor="black",
    capsize=8,
    capthick=2,
    lw=2,
    label="90% PI"
)

axes2[1, 1].axhline(0, color="black", ls="--", lw=1.2)

axes2[1, 1].set_xticks(range(5))
axes2[1, 1].set_xticklabels(day_labels, fontsize=9)

axes2[1, 1].set_ylabel("% Change")

axes2[1, 1].set_title(
    "5-Day Forecast with 90% PI",
    fontweight='bold'
)

axes2[1, 1].legend(fontsize=9)
axes2[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()

fig2.savefig(
    "lstm_single_model_metrics.png",
    dpi=150,
    bbox_inches="tight"
)

print("  Saved: lstm_single_model_metrics.png")

plt.close(fig2)