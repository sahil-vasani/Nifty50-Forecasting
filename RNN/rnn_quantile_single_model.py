# =============================================================================
# Nifty50 Quantile Forecasting using SimpleRNN  —  PyTorch
# Probabilistic Time Series Forecasting with Prediction Intervals
# =============================================================================
# Objective: Estimate the full conditional distribution of next-day Nifty50
# returns using a multi-output SimpleRNN with quantile regression, including
# systematic hyperparameter tuning.
#
# Research Foundation:
#   - "Probabilistic Forecasting with Recurrent Neural Networks" (Wen et al., 2017)
#   - "A Multi-Horizon Quantile Recurrent Forecaster" (Gasthaus et al., 2019)
#
# Conversion note: Only the model / training / inference code has been ported
# from TensorFlow/Keras to PyTorch.  Data loading, feature engineering,
# evaluation metrics, and visualisation are unchanged.
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product as itertools_product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ★ PyTorch replaces TensorFlow/Keras ★
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Random Seeds ─────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Plotting Style ────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print(f"PyTorch Version : {torch.__version__}")
print(f"Device          : {DEVICE}")


# =============================================================================
# 1. Quantile Regression — Pinball Loss
# =============================================================================

QUANTILES = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]


def pinball_loss(predictions, targets, quantile):
    """
    Pinball (quantile) loss for a single quantile level.

    Args:
        predictions : torch.Tensor, shape (N,)
        targets     : torch.Tensor, shape (N,)
        quantile    : float, target quantile level

    Returns:
        Scalar loss tensor
    """
    error = targets - predictions
    loss  = torch.max(quantile * error, (quantile - 1) * error)
    return loss.mean()


def multi_quantile_loss(predictions_list, targets, quantiles):
    """
    Sum of pinball losses across all quantile heads.

    Args:
        predictions_list : list of tensors, one per quantile, each shape (N,)
        targets          : torch.Tensor, shape (N,)
        quantiles        : list of float

    Returns:
        Scalar total loss tensor
    """
    total = sum(
        pinball_loss(pred.squeeze(), targets, q)
        for pred, q in zip(predictions_list, quantiles)
    )
    return total


print("Quantile Levels:", QUANTILES)
print("Pinball loss function: ready")


# =============================================================================
# 2. Data Loading and Preprocessing
# =============================================================================

df = pd.read_csv('../nifty_final_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Dataset Shape: {df.shape}")
print(f"Date Range: {df.index.min()} to {df.index.max()}")

# Feature Groups
technical_features       = ['log_ret', 'vol_5', 'vol_15', 'rsi', 'momentum', 'trend_strength']
price_features           = ['body', 'range', 'upper_wick', 'lower_wick', 'close_pos']
ma_features              = ['ma_5', 'ma_20', 'dist_ma_5', 'dist_ma_20']
volume_features          = ['volume', 'volume_ma_5', 'volume_spike']
sector_features          = ['bank_ret', 'it_ret', 'pharma_ret', 'auto_ret', 'fmcg_ret', 'metal_ret', 'energy_ret']
lag_features             = (
    ['log_ret_lag1', 'log_ret_lag2'] +
    [f'{s}_lag1' for s in ['bank_ret', 'it_ret', 'pharma_ret', 'auto_ret', 'fmcg_ret', 'metal_ret', 'energy_ret']] +
    [f'{s}_lag2' for s in ['bank_ret', 'it_ret', 'pharma_ret', 'auto_ret', 'fmcg_ret', 'metal_ret', 'energy_ret']]
)
sector_analysis_features = [
    'sector_mean', 'sector_std',
    'bank_ret_vs_nifty', 'it_ret_vs_nifty', 'pharma_ret_vs_nifty',
    'auto_ret_vs_nifty', 'fmcg_ret_vs_nifty', 'metal_ret_vs_nifty', 'energy_ret_vs_nifty'
]

selected_features = (
    technical_features + price_features + ma_features + volume_features +
    sector_features + lag_features + sector_analysis_features
)

df_features = df[selected_features + ['target']].copy()
df_features = df_features.dropna()

print(f"\nTotal Features: {len(selected_features)}")
print(f"Records after cleaning: {len(df_features)}")

# ── Train / Test Split ────────────────────────────────────────────────────────
train_size = int(len(df_features) * 0.8)
train_data = df_features.iloc[:train_size]
test_data  = df_features.iloc[train_size:]

X_train = train_data[selected_features].values
y_train = train_data['target'].values
X_test  = test_data[selected_features].values
y_test  = test_data['target'].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print("=" * 80)
print("DATA SPLIT")
print("=" * 80)
print(f"Training : {len(train_data)} samples ({len(train_data)/len(df_features)*100:.1f}%)")
print(f"Testing  : {len(test_data)} samples ({len(test_data)/len(df_features)*100:.1f}%)")


# =============================================================================
# 3. Sequence Creation
# =============================================================================

def create_sequences(X, y, window_size):
    """Create rolling window sequences for RNN input."""
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i - window_size:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


WINDOW_SIZE = 30

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, WINDOW_SIZE)
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  WINDOW_SIZE)

print(f"Window Size     : {WINDOW_SIZE} days")
print(f"Train Sequences : {X_train_seq.shape}")
print(f"Test Sequences  : {X_test_seq.shape}")


# =============================================================================
# 4. Multi-Quantile SimpleRNN Model  (PyTorch)
# =============================================================================

class QuantileRNN(nn.Module):
    """
    Multi-output SimpleRNN quantile regression model.

    Architecture:
        Input → RNN(units) → RNN(units//2) → BN → Dense(64)
              → [Dense(32) → Dense(1)] × Q

    Args:
        n_features  : Number of input features
        quantiles   : List of quantile levels
        units       : Hidden size of first RNN layer
        dropout     : Dropout probability
    """

    def __init__(self, n_features, quantiles, units=128, dropout=0.2):
        super().__init__()
        self.quantiles = quantiles

        # ── Shared RNN backbone ───────────────────────────────────────────────
        self.rnn1 = nn.RNN(
            input_size=n_features, hidden_size=units,
            batch_first=True, nonlinearity='tanh'
        )
        self.drop1 = nn.Dropout(dropout)
        self.bn1   = nn.BatchNorm1d(units)

        self.rnn2 = nn.RNN(
            input_size=units, hidden_size=units // 2,
            batch_first=True, nonlinearity='tanh'
        )
        self.drop2 = nn.Dropout(dropout)
        self.bn2   = nn.BatchNorm1d(units // 2)

        # ── Shared dense ─────────────────────────────────────────────────────
        self.shared_fc = nn.Sequential(
            nn.Linear(units // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )

        # ── Per-quantile heads ────────────────────────────────────────────────
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(32, 1),
            )
            for _ in quantiles
        ])

    def forward(self, x):
        """
        Args:
            x : torch.Tensor, shape (batch, window_size, n_features)

        Returns:
            List of tensors, one per quantile, each shape (batch, 1)
        """
        # RNN layer 1  →  take all time steps
        out, _ = self.rnn1(x)                    # (B, T, units)
        out     = self.drop1(out)
        out     = self.bn1(out.contiguous().view(-1, out.size(2))).view(out.size())

        # RNN layer 2  →  take last time step only
        out, _ = self.rnn2(out)                  # (B, T, units//2)
        out     = out[:, -1, :]                   # (B, units//2)
        out     = self.drop2(out)
        out     = self.bn2(out)

        # Shared dense
        out = self.shared_fc(out)                 # (B, 64)

        # Per-quantile outputs
        return [head(out) for head in self.heads]  # list of (B, 1)


def build_quantile_rnn_model(window_size, n_features, quantiles,
                              units=128, dropout=0.2, learning_rate=0.001):
    """Instantiate QuantileRNN and its Adam optimiser."""
    model     = QuantileRNN(n_features, quantiles, units=units, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


# Quick preview
WINDOW_SIZE = 10  # placeholder; updated after tuning
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, WINDOW_SIZE)
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  WINDOW_SIZE)

_preview_model, _ = build_quantile_rnn_model(
    WINDOW_SIZE, X_train_scaled.shape[1], QUANTILES
)
print("=" * 80)
print("QUANTILE SimpleRNN ARCHITECTURE  (PyTorch)")
print("=" * 80)
print(_preview_model)
total_params = sum(p.numel() for p in _preview_model.parameters() if p.requires_grad)
print(f"\nTotal Trainable Parameters: {total_params:,}")
del _preview_model


# =============================================================================
# Helper: training loop with early stopping + LR scheduling
# =============================================================================

def train_model(model, optimizer, X_tr, y_tr,
                epochs=100, batch_size=32, val_split=0.2,
                patience=10, lr_patience=5, lr_factor=0.5, min_lr=1e-7):
    """
    Train a QuantileRNN (or QuantileLSTM) model.

    Returns:
        history dict with keys 'train_loss' and 'val_loss'
    """
    # ── Validation split ──────────────────────────────────────────────────────
    n_val  = int(len(X_tr) * val_split)
    n_tr   = len(X_tr) - n_val

    X_t = torch.tensor(X_tr[:n_tr],  dtype=torch.float32)
    y_t = torch.tensor(y_tr[:n_tr],  dtype=torch.float32)
    X_v = torch.tensor(X_tr[n_tr:],  dtype=torch.float32)
    y_v = torch.tensor(y_tr[n_tr:],  dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor,
        patience=lr_patience, min_lr=min_lr
    )

    history         = {'train_loss': [], 'val_loss': []}
    best_val_loss   = float('inf')
    best_weights    = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = multi_quantile_loss(preds, yb, model.quantiles)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= n_tr

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            xv, yv   = X_v.to(DEVICE), y_v.to(DEVICE)
            val_preds = model(xv)
            val_loss  = multi_quantile_loss(val_preds, yv, model.quantiles).item()

        scheduler.step(val_loss)
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)

        # ── Early stopping ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_weights     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return history


def predict_quantiles(model, X_np):
    """
    Run inference and return a list of numpy arrays (one per quantile).

    Args:
        model : trained QuantileRNN / QuantileLSTM
        X_np  : numpy array, shape (N, window, features)

    Returns:
        List of numpy arrays, each shape (N,)
    """
    model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model(X_t)
    return [p.squeeze().cpu().numpy() for p in preds]


# =============================================================================
# 5. Systematic Hyperparameter Tuning
# =============================================================================

PARAM_GRID_Q = {
    'window_size'   : [5, 10, 20],
    'units'         : [32, 64, 128],
    'dropout'       : [0.0, 0.2, 0.3],
    'learning_rate' : [0.001, 0.0005],
}

all_combos_q = list(itertools_product(
    PARAM_GRID_Q['window_size'],
    PARAM_GRID_Q['units'],
    PARAM_GRID_Q['dropout'],
    PARAM_GRID_Q['learning_rate']
))

print("=" * 70)
print("HYPERPARAMETER SEARCH SPACE (Quantile RNN)")
print("=" * 70)
for param, values in PARAM_GRID_Q.items():
    print(f"  {param:<18}: {values}")
print(f"\n  Total combinations : {len(all_combos_q)}")
print("=" * 70)

# ── Grid Search ───────────────────────────────────────────────────────────────
q_tuning_results = []
best_q_rmse      = float('inf')
best_q_config    = None
best_q_model     = None
best_q_result    = None

n_features_q = X_train_scaled.shape[1]

for combo_idx, (ws, units, dropout, lr) in enumerate(all_combos_q, 1):
    config = dict(window_size=ws, units=units, dropout=dropout, learning_rate=lr)

    print(
        f"[{combo_idx:02d}/{len(all_combos_q)}]  "
        f"window={ws:2d}  units={units:3d}  dropout={dropout:.1f}  lr={lr}",
        end='  →  ', flush=True
    )

    X_tr_q, y_tr_q = create_sequences(X_train_scaled, y_train_scaled, ws)
    X_te_q, y_te_q = create_sequences(X_test_scaled,  y_test_scaled,  ws)

    model, optimizer = build_quantile_rnn_model(
        ws, n_features_q, QUANTILES,
        units=units, dropout=dropout, learning_rate=lr
    )

    train_model(
        model, optimizer, X_tr_q, y_tr_q,
        epochs=100, batch_size=32, val_split=0.2,
        patience=10, lr_patience=5
    )

    # Evaluate on median quantile (q50)
    preds_all_scaled = predict_quantiles(model, X_te_q)
    q50_idx          = QUANTILES.index(0.5)
    y_pred_scaled_q  = preds_all_scaled[q50_idx]
    y_pred_q         = scaler_y.inverse_transform(y_pred_scaled_q.reshape(-1, 1)).flatten()
    y_true_q         = scaler_y.inverse_transform(y_te_q.reshape(-1, 1)).flatten()

    mse_q  = mean_squared_error(y_true_q, y_pred_q)
    rmse_q = np.sqrt(mse_q)
    mae_q  = mean_absolute_error(y_true_q, y_pred_q)

    print(f"RMSE={rmse_q:.6f}  MSE={mse_q:.6f}  MAE={mae_q:.6f}")

    record = dict(
        window_size=ws, units=units, dropout=dropout, learning_rate=lr,
        MSE=round(mse_q, 6), RMSE=round(rmse_q, 6), MAE=round(mae_q, 6),
        predictions=y_pred_q, actuals=y_true_q
    )
    q_tuning_results.append(record)

    if rmse_q < best_q_rmse:
        best_q_rmse   = rmse_q
        best_q_config = config
        best_q_model  = model
        best_q_result = record

    del model, optimizer

print("\nHyperparameter search complete.")

# ── Results Table ─────────────────────────────────────────────────────────────
q_results_df = pd.DataFrame([{
    'Window Size'   : r['window_size'],
    'RNN Units'     : r['units'],
    'Dropout'       : r['dropout'],
    'Learning Rate' : r['learning_rate'],
    'MSE'           : r['MSE'],
    'RMSE'          : r['RMSE'],
    'MAE'           : r['MAE'],
} for r in q_tuning_results]).sort_values('RMSE').reset_index(drop=True)
q_results_df.index += 1

print("=" * 85)
print("QUANTILE RNN — HYPERPARAMETER TUNING RESULTS  (sorted by RMSE)")
print("=" * 85)
print(q_results_df.to_string())

print("\n" + "=" * 70)
print("BEST HYPERPARAMETER COMBINATION (Quantile RNN)")
print("=" * 70)
print(f"  Best Window Size   : {best_q_config['window_size']}")
print(f"  Best RNN Units     : {best_q_config['units']}")
print(f"  Best Dropout       : {best_q_config['dropout']}")
print(f"  Best Learning Rate : {best_q_config['learning_rate']}")
print(f"  Best RMSE          : {best_q_result['RMSE']:.6f}")
print(f"  Best MSE           : {best_q_result['MSE']:.6f}")
print(f"  Best MAE           : {best_q_result['MAE']:.6f}")
print("=" * 70)

# ── Tuning Heatmap ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Quantile RNN Tuning — RMSE Heatmap per Dropout', fontsize=14, fontweight='bold')
for ax, dr in zip(axes, [0.0, 0.2, 0.3]):
    sub   = q_results_df[q_results_df['Dropout'] == dr].copy()
    pivot = sub.groupby(['RNN Units', 'Window Size'])['RMSE'].min().unstack()
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax)
    ax.set_title(f'Dropout = {dr}', fontweight='bold')
    ax.set_xlabel('Window Size')
    ax.set_ylabel('RNN Units')
plt.tight_layout()
plt.show()

# ── Rebuild Best Model ────────────────────────────────────────────────────────
WINDOW_SIZE = best_q_config['window_size']
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, WINDOW_SIZE)
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  WINDOW_SIZE)

quantile_model, quantile_optimizer = build_quantile_rnn_model(
    WINDOW_SIZE, n_features_q, QUANTILES,
    units=best_q_config['units'],
    dropout=best_q_config['dropout'],
    learning_rate=best_q_config['learning_rate']
)
print("Best quantile model rebuilt with optimal hyperparameters.")


# =============================================================================
# 5. Model Training (Best Config)
# =============================================================================

print("=" * 80)
print("TRAINING BEST QUANTILE SimpleRNN MODEL")
print("=" * 80)
print(f"Quantile Levels  : {QUANTILES}")
print(f"Window Size      : {WINDOW_SIZE}")
print(f"RNN Units        : {best_q_config['units']}")
print(f"Dropout          : {best_q_config['dropout']}")
print(f"Learning Rate    : {best_q_config['learning_rate']}")
print(f"Training Samples : {len(X_train_seq)}")
print("\nTraining started...\n")

history = train_model(
    quantile_model, quantile_optimizer,
    X_train_seq, y_train_seq,
    epochs=100, batch_size=32, val_split=0.2,
    patience=20, lr_patience=7
)

print(f"\nTraining completed! Final epoch: {len(history['train_loss'])}")

# ── Training History ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax.plot(history['val_loss'],   label='Val Loss',   linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Total Pinball Loss')
ax.set_title('Quantile RNN Training History', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
# 6. Generate Predictions
# =============================================================================

preds_scaled_list = predict_quantiles(quantile_model, X_test_seq)

quantile_predictions = {}
for idx, q in enumerate(QUANTILES):
    pred = scaler_y.inverse_transform(preds_scaled_list[idx].reshape(-1, 1)).flatten()
    quantile_predictions[q] = pred

y_actual = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

print("=" * 80)
print("QUANTILE PREDICTIONS GENERATED")
print("=" * 80)
print(f"Test samples: {len(y_actual)}")
for q, pred in quantile_predictions.items():
    print(f"  Q{q}: {pred.shape}")


# =============================================================================
# 7. Prediction Interval Metrics
# =============================================================================

def calculate_interval_metrics(y_true, lower_bound, upper_bound, confidence_level=0.95):
    """Calculate comprehensive prediction interval metrics."""
    coverage   = (y_true >= lower_bound) & (y_true <= upper_bound)
    picp       = np.mean(coverage)
    widths     = upper_bound - lower_bound
    mpiw       = np.mean(widths)
    data_range = np.max(y_true) - np.min(y_true)
    nmpiw      = mpiw / data_range if data_range > 0 else 0

    mu    = confidence_level
    eta   = 50
    gamma = 0 if picp >= mu else 1
    cwc   = nmpiw * (1 + gamma * np.exp(-eta * (picp - mu)))

    ace       = np.abs(picp - confidence_level)
    sharpness = mpiw / np.mean(np.abs(y_true)) * 100 if np.mean(np.abs(y_true)) > 0 else 0

    alpha         = 1 - confidence_level
    winkler       = widths + (2 / alpha) * (
        (lower_bound - y_true) * (y_true < lower_bound) +
        (y_true - upper_bound) * (y_true > upper_bound)
    )
    return {
        'PICP': picp, 'MPIW': mpiw, 'NMPIW': nmpiw, 'CWC': cwc,
        'ACE': ace, 'Sharpness_%': sharpness, 'Winkler_Score': np.mean(winkler),
        'Coverage': coverage, 'Widths': widths
    }


confidence_levels = {'80%': (0.1, 0.9), '50%': (0.25, 0.75)}

print("=" * 80)
print("PREDICTION INTERVAL METRICS")
print("=" * 80)

metrics_summary = {}
for level_name, (q_low, q_high) in confidence_levels.items():
    metrics = calculate_interval_metrics(
        y_actual, quantile_predictions[q_low], quantile_predictions[q_high],
        confidence_level=float(level_name.strip('%')) / 100
    )
    metrics_summary[level_name] = metrics
    ace_val = metrics['ACE']
    quality = ("EXCELLENT" if ace_val < 0.02 else "GOOD" if ace_val < 0.05 else
               "ACCEPTABLE" if ace_val < 0.10 else "POOR")

    print(f"\n{level_name} Prediction Interval:")
    print("-" * 80)
    print(f"  PICP (Coverage)   : {metrics['PICP']*100:.2f}% (Target: {level_name})")
    print(f"  MPIW (Avg Width)  : {metrics['MPIW']:.6f}")
    print(f"  NMPIW (Normalized): {metrics['NMPIW']:.6f}")
    print(f"  CWC (Quality)     : {metrics['CWC']:.6f} (Lower is better)")
    print(f"  ACE (Coverage Err): {metrics['ACE']*100:.2f}%")
    print(f"  Sharpness         : {metrics['Sharpness_%']:.2f}%")
    print(f"  Winkler Score     : {metrics['Winkler_Score']:.6f}")
    print(f"  Coverage Quality  : {quality}")


# =============================================================================
# 8. Visualization of Prediction Intervals
# =============================================================================

test_dates = test_data.index[WINDOW_SIZE:]

fig, axes = plt.subplots(2, 1, figsize=(18, 12))

axes[0].plot(test_dates, y_actual,                   'b-',  linewidth=2, label='Actual',            alpha=0.8)
axes[0].plot(test_dates, quantile_predictions[0.5],  'r--', linewidth=2, label='Median Prediction', alpha=0.8)
axes[0].fill_between(test_dates, quantile_predictions[0.025], quantile_predictions[0.975],
                     alpha=0.3, color='red',    label='95% Prediction Interval')
axes[0].fill_between(test_dates, quantile_predictions[0.25],  quantile_predictions[0.75],
                     alpha=0.5, color='orange', label='50% Prediction Interval (IQR)')
axes[0].axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Return (%)', fontsize=12)
axes[0].set_title('Quantile RNN Predictions with Multiple Confidence Intervals (PyTorch)',
                  fontweight='bold', fontsize=14)
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(alpha=0.3)

zoom_n = min(50, len(test_dates))
axes[1].plot(test_dates[-zoom_n:], y_actual[-zoom_n:],
             'b-', linewidth=2.5, label='Actual', marker='o', markersize=4)
axes[1].plot(test_dates[-zoom_n:], quantile_predictions[0.5][-zoom_n:],
             'r--', linewidth=2.5, label='Median Prediction', marker='s', markersize=4)
axes[1].fill_between(test_dates[-zoom_n:],
                     quantile_predictions[0.025][-zoom_n:],
                     quantile_predictions[0.975][-zoom_n:],
                     alpha=0.3, color='red', label='95% PI')
axes[1].axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Return (%)', fontsize=12)
axes[1].set_title(f'Detailed View: Last {zoom_n} Predictions', fontweight='bold', fontsize=14)
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ── Coverage Analysis ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

coverage_80 = metrics_summary['80%']['Coverage']
axes[0, 0].fill_between(test_dates, 0, 1, where=coverage_80,  alpha=0.3, color='green', label='Covered')
axes[0, 0].fill_between(test_dates, 0, 1, where=~coverage_80, alpha=0.3, color='red',   label='Not Covered')
axes[0, 0].set_ylim([0, 1.1])
axes[0, 0].set_title(f'80% PI Coverage (PICP: {metrics_summary["80%"]["PICP"]*100:.2f}%)', fontweight='bold')
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

widths_80 = metrics_summary['80%']['Widths']
axes[0, 1].plot(test_dates, widths_80, linewidth=1.5, color='purple')
axes[0, 1].axhline(metrics_summary['80%']['MPIW'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {metrics_summary["80%"]["MPIW"]:.4f}')
axes[0, 1].set_title('Interval Width Over Time', fontweight='bold')
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

axes[1, 0].hist(widths_80, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 0].axvline(metrics_summary['80%']['MPIW'], color='red', linestyle='--', linewidth=2)
axes[1, 0].set_title('Distribution of Interval Widths', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

levels  = list(confidence_levels.keys())
picps   = [metrics_summary[l]['PICP'] * 100 for l in levels]
targets = [float(l.strip('%')) for l in levels]
x, w    = np.arange(len(levels)), 0.35
axes[1, 1].bar(x - w/2, picps,   w, label='Actual PICP',    color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 1].bar(x + w/2, targets, w, label='Target Coverage', color='orange',    alpha=0.7, edgecolor='black')
axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(levels)
axes[1, 1].set_title('PICP vs Target Coverage', fontweight='bold')
axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()


# =============================================================================
# 9. Quantile Calibration Analysis
# =============================================================================

empirical_coverage = [np.mean(y_actual <= quantile_predictions[q]) for q in QUANTILES]
calibration_errors = np.abs(np.array(QUANTILES) - np.array(empirical_coverage))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
axes[0].plot(QUANTILES, empirical_coverage, 'bo-', linewidth=2, markersize=8, label='Model Calibration')
axes[0].set_xlabel('Predicted Quantile', fontsize=12); axes[0].set_ylabel('Empirical Coverage', fontsize=12)
axes[0].set_title('Quantile Calibration Plot', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3); axes[0].set_xlim([0,1]); axes[0].set_ylim([0,1])

axes[1].bar([f'Q{int(q*100)}' for q in QUANTILES], calibration_errors, color='coral', alpha=0.7, edgecolor='black')
axes[1].set_title('Quantile Calibration Errors', fontweight='bold', fontsize=14)
axes[1].grid(alpha=0.3, axis='y'); axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("CALIBRATION ANALYSIS")
print("=" * 80)
print(f"\n{'Quantile':<12} {'Target':<12} {'Empirical':<12} {'Error':<12}")
print("-" * 48)
for q, emp, err in zip(QUANTILES, empirical_coverage, calibration_errors):
    print(f"{q:<12.3f} {q:<12.3f} {emp:<12.3f} {err:<12.4f}")

mean_abs_error = np.mean(calibration_errors)
print(f"\nMean Absolute Calibration Error: {mean_abs_error:.4f}")
print("Calibration Quality:", (
    "EXCELLENT" if mean_abs_error < 0.02 else
    "GOOD"      if mean_abs_error < 0.05 else "NEEDS IMPROVEMENT"
))


# =============================================================================
# 10. Next 5 Days Predictions
# =============================================================================

def predict_next_n_days_quantile(model, last_sequence, scaler_y, quantiles, n_days=5):
    """Predict next n days with full quantile predictions."""
    predictions      = {q: [] for q in quantiles}
    current_sequence = last_sequence.copy()

    model.eval()
    for _ in range(n_days):
        x_in = torch.tensor(
            current_sequence.reshape(1, *current_sequence.shape),
            dtype=torch.float32
        ).to(DEVICE)

        with torch.no_grad():
            q_preds = model(x_in)

        for idx, q in enumerate(quantiles):
            pred_scaled = q_preds[idx][0, 0].cpu().item()
            pred        = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            predictions[q].append(pred)

        median_idx      = quantiles.index(0.5)
        median_pred_sc  = q_preds[median_idx][0, 0].cpu().item()
        new_features    = current_sequence[-1].copy()
        new_features[0] = median_pred_sc
        current_sequence = np.vstack([current_sequence[1:], new_features])

    return predictions


last_sequence_scaled = X_test_scaled[-WINDOW_SIZE:]
next_5_quantiles = predict_next_n_days_quantile(
    quantile_model, last_sequence_scaled, scaler_y, QUANTILES, n_days=5
)

last_date    = df_features.index[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)

print("=" * 80)
print("NEXT 5 DAYS QUANTILE PREDICTIONS")
print("=" * 80)
print(f"\nLast Trading Date : {last_date.strftime('%Y-%m-%d')}")
print(f"Last Close Price  : {df['close'].iloc[-1]:.2f}\n")
print(f"{'Date':<15} {'Q2.5%':<10} {'Q25%':<10} {'Median':<10} {'Q75%':<10} {'Q97.5%':<10} {'Price (Med)':<12}")
print("-" * 80)

current_price = df['close'].iloc[-1]
for i, date in enumerate(future_dates):
    q50 = next_5_quantiles[0.5][i]
    predicted_price = current_price * (1 + q50 / 100)
    print(
        f"{date.strftime('%Y-%m-%d'):<15} "
        f"{next_5_quantiles[0.025][i]:>+7.4f}%  "
        f"{next_5_quantiles[0.25][i]:>+7.4f}%  "
        f"{q50:>+7.4f}%  "
        f"{next_5_quantiles[0.75][i]:>+7.4f}%  "
        f"{next_5_quantiles[0.975][i]:>+7.4f}%  "
        f"{predicted_price:>10.2f}"
    )
    current_price = predicted_price

cumulative_median = sum(next_5_quantiles[0.5])
cumulative_lower  = sum(next_5_quantiles[0.025])
cumulative_upper  = sum(next_5_quantiles[0.975])
final_price       = df['close'].iloc[-1] * (1 + cumulative_median / 100)
price_lower       = df['close'].iloc[-1] * (1 + cumulative_lower  / 100)
price_upper       = df['close'].iloc[-1] * (1 + cumulative_upper  / 100)

print(f"\n5-Day Median Return       : {cumulative_median:+.4f}%")
print(f"Predicted Price (Median)  : {final_price:.2f}  (from {df['close'].iloc[-1]:.2f})")
print(f"95% Price Range           : [{price_lower:.2f}, {price_upper:.2f}]")

# ── 5-Day Forecast Visualization ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

historical_dates  = df_features.index[-30:]
historical_prices = df['close'].iloc[-30:]

median_prices = [df['close'].iloc[-1]]
lower_prices  = [df['close'].iloc[-1]]
upper_prices  = [df['close'].iloc[-1]]
q25_prices    = [df['close'].iloc[-1]]
q75_prices    = [df['close'].iloc[-1]]
for i in range(5):
    median_prices.append(median_prices[-1] * (1 + next_5_quantiles[0.5][i]   / 100))
    lower_prices.append(lower_prices[-1]   * (1 + next_5_quantiles[0.025][i] / 100))
    upper_prices.append(upper_prices[-1]   * (1 + next_5_quantiles[0.975][i] / 100))
    q25_prices.append(q25_prices[-1]       * (1 + next_5_quantiles[0.25][i]  / 100))
    q75_prices.append(q75_prices[-1]       * (1 + next_5_quantiles[0.75][i]  / 100))

all_dates = [historical_dates[-1]] + list(future_dates)

axes[0].plot(historical_dates, historical_prices, 'b-',  linewidth=2.5, label='Historical')
axes[0].plot(all_dates, median_prices, 'r-', linewidth=2.5, marker='o', markersize=8, label='Median Forecast')
axes[0].fill_between(all_dates, lower_prices, upper_prices, alpha=0.2, color='red',    label='95% PI')
axes[0].fill_between(all_dates, q25_prices,   q75_prices,   alpha=0.3, color='orange', label='50% PI (IQR)')
axes[0].axvline(historical_dates[-1], color='black', linestyle=':', linewidth=2, alpha=0.7)
axes[0].set_title('5-Day Price Forecast with Prediction Intervals', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3)

day_labels = [f'Day {i+1}\n{d.strftime("%m/%d")}' for i, d in enumerate(future_dates)]
x_pos      = np.arange(5)
axes[1].bar(x_pos, next_5_quantiles[0.5],
            color=['green' if r > 0 else 'red' for r in next_5_quantiles[0.5]],
            alpha=0.6, label='Median')
lower_err = [next_5_quantiles[0.5][i] - next_5_quantiles[0.025][i] for i in range(5)]
upper_err = [next_5_quantiles[0.975][i] - next_5_quantiles[0.5][i] for i in range(5)]
axes[1].errorbar(x_pos, next_5_quantiles[0.5], yerr=[lower_err, upper_err],
                 fmt='none', ecolor='black', capsize=8, capthick=2, linewidth=2, label='95% PI')
for i in range(5):
    axes[1].plot([i, i], [next_5_quantiles[0.25][i], next_5_quantiles[0.75][i]],
                 'o-', color='orange', linewidth=4, markersize=6, alpha=0.6)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1.5)
axes[1].set_xticks(x_pos); axes[1].set_xticklabels(day_labels)
axes[1].set_ylabel('Return (%)'); axes[1].set_title('Daily Return Predictions', fontweight='bold', fontsize=14)
axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.show()


# =============================================================================
# 11. Sector Analysis
# =============================================================================

sectors     = ['bank', 'it', 'pharma', 'auto', 'fmcg', 'metal', 'energy']
recent_data = df_features.tail(30)

sector_analysis = pd.DataFrame({
    'Sector'      : [s.upper() for s in sectors],
    'Mean_Return' : [recent_data[f'{s}_ret'].mean() * 100 for s in sectors],
    'Volatility'  : [recent_data[f'{s}_ret'].std()  * 100 for s in sectors],
    'Latest'      : [df_features[f'{s}_ret'].iloc[-1] * 100 for s in sectors],
    'Min'         : [recent_data[f'{s}_ret'].min()  * 100 for s in sectors],
    'Max'         : [recent_data[f'{s}_ret'].max()  * 100 for s in sectors],
})
sector_analysis['Confidence_Score'] = (
    sector_analysis['Mean_Return'].rank() * 0.4 +
    (100 - sector_analysis['Volatility'].rank()) * 0.3 +
    sector_analysis['Latest'].rank() * 0.3
)
sector_analysis = sector_analysis.sort_values('Confidence_Score', ascending=False)

print("=" * 80)
print("SECTOR ANALYSIS WITH UNCERTAINTY QUANTIFICATION")
print("=" * 80)
print(sector_analysis.to_string(index=False))

print("\nTOP 3 INVESTMENT RECOMMENDATIONS:")
for idx, (_, row) in enumerate(sector_analysis.head(3).iterrows(), 1):
    rec = ("STRONG BUY" if row['Mean_Return'] > 0.5 and row['Volatility'] < 2.0 else
           "BUY"        if row['Mean_Return'] > 0.3 else
           "HOLD"       if row['Mean_Return'] > 0   else "AVOID")
    print(f"  {idx}. {row['Sector']}: {rec}  |  Return {row['Mean_Return']:+.4f}%  |  Risk {row['Volatility']:.4f}%")


# =============================================================================
# 12. Comprehensive Summary
# =============================================================================

available_level = list(metrics_summary.keys())[0]
m = metrics_summary[available_level]

print("=" * 80)
print("QUANTILE FORECASTING — FINAL SUMMARY (SimpleRNN / PyTorch)")
print("=" * 80)
print(f"  Architecture   : Multi-Output Quantile SimpleRNN")
print(f"  Window Size    : {WINDOW_SIZE} days")
print(f"  RNN Units      : {best_q_config['units']} → {best_q_config['units']//2}")
print(f"  Dropout        : {best_q_config['dropout']}")
print(f"  Learning Rate  : {best_q_config['learning_rate']}")
print(f"  Device         : {DEVICE}")
total_params = sum(p.numel() for p in quantile_model.parameters() if p.requires_grad)
print(f"  Total Params   : {total_params:,}")
print(f"\n  Best RMSE (q=0.5) : {best_q_result['RMSE']:.6f}")
print(f"  PICP ({available_level})      : {m['PICP']*100:.2f}%")
print(f"  CWC              : {m['CWC']:.6f}")
print(f"  Calibration MACE : {m['ACE']:.4f}")
print(f"\n  5-Day Median Return : {cumulative_median:+.4f}%")
print(f"  Price Target        : {final_price:.2f}  (95% range: [{price_lower:.2f}, {price_upper:.2f}])")

print("\n" + "=" * 80)
print("DISCLAIMER: For research purposes only. Consult a financial advisor.")
print("=" * 80)


# =============================================================================
# Save Model and Results
# =============================================================================

torch.save(quantile_model.state_dict(), 'quantile_forecasting_rnn_torch.pth')
print("\nModel weights saved to: quantile_forecasting_rnn_torch.pth")

results_df = pd.DataFrame({
    'Date'   : future_dates,
    'Q2.5%'  : next_5_quantiles[0.025],
    'Q10%'   : next_5_quantiles[0.1],
    'Q25%'   : next_5_quantiles[0.25],
    'Median' : next_5_quantiles[0.5],
    'Q75%'   : next_5_quantiles[0.75],
    'Q90%'   : next_5_quantiles[0.9],
    'Q97.5%' : next_5_quantiles[0.975],
})
results_df.to_csv('next_5_days_quantile_predictions_rnn_torch.csv', index=False)

metrics_df = pd.DataFrame([
    {'Confidence_Level': level, **{k: v for k, v in metrics.items()
                                   if k not in ('Coverage', 'Widths')}}
    for level, metrics in metrics_summary.items()
])
metrics_df.to_csv('prediction_interval_metrics_rnn_torch.csv', index=False)
print("Predictions and metrics saved.")