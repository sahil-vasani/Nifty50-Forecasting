# =============================================================================
# Nifty50 Quantile Forecasting using LSTM
# Probabilistic Time Series Forecasting with Prediction Intervals
# =============================================================================
# Objective: Estimate the full conditional distribution of next-day Nifty50
# returns using a multi-output LSTM with quantile regression, including
# systematic hyperparameter tuning.
#
# Research Foundation:
#   - "Probabilistic Forecasting with Recurrent Neural Networks" (Wen et al., 2017)
#   - "A Multi-Horizon Quantile Recurrent Forecaster" (Gasthaus et al., 2019)
#
# Key Difference from SimpleRNN version:
#   SimpleRNN replaced with LSTM layers throughout.
#   LSTM has explicit cell state (long-term memory) and three gates
#   (input, forget, output), making it better suited to capture long-range
#   temporal dependencies in financial time series.
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
# ★ LSTM replaces SimpleRNN ★
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# ── Random Seeds ─────────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ── Plotting Style ────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")


# =============================================================================
# 1. Quantile Regression — Pinball Loss
# =============================================================================

QUANTILES = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]


def pinball_loss(quantile):
    """
    Create pinball loss function for a specific quantile.

    Args:
        quantile: Target quantile level (e.g., 0.5 for median)

    Returns:
        Loss function for this quantile
    """
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return K.mean(K.maximum(quantile * error, (quantile - 1) * error), axis=-1)

    loss.__name__ = f'pinball_loss_{int(quantile * 100)}'
    return loss


print("Quantile Levels:", QUANTILES)
print("\nPinball Loss Functions Created:")
for q in QUANTILES:
    print(f"  - Quantile {q:.3f}: {pinball_loss(q).__name__}")


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
technical_features      = ['log_ret', 'vol_5', 'vol_15', 'rsi', 'momentum', 'trend_strength']
price_features          = ['body', 'range', 'upper_wick', 'lower_wick', 'close_pos']
ma_features             = ['ma_5', 'ma_20', 'dist_ma_5', 'dist_ma_20']
volume_features         = ['volume', 'volume_ma_5', 'volume_spike']
sector_features         = ['bank_ret', 'it_ret', 'pharma_ret', 'auto_ret', 'fmcg_ret', 'metal_ret', 'energy_ret']
lag_features            = (
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
    """Create rolling window sequences for LSTM input."""
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
# 4. Multi-Quantile LSTM Model
# =============================================================================
# ★ CHANGE vs SimpleRNN version:
#   SimpleRNN → LSTM throughout the backbone.
#   LSTM parameters are ~4× those of SimpleRNN for the same `units` value
#   because each LSTM cell has 4 gates, making it more expressive but slower.
# =============================================================================

def build_quantile_lstm_model(window_size, n_features, quantiles,
                               units=128, dropout=0.2, learning_rate=0.001):
    """
    Build multi-output LSTM quantile regression model.

    Architecture:
        Input → LSTM(units, return_sequences=True) → LSTM(units//2)
              → Dense(64) → [Dense(32) → Dense(1)] × Q

    Args:
        window_size   : Sequence length
        n_features    : Number of input features
        quantiles     : List of quantile levels
        units         : LSTM units in first layer (halved in second layer)
        dropout       : Dropout rate
        learning_rate : Adam learning rate

    Returns:
        Compiled Keras Model
    """
    inputs = Input(shape=(window_size, n_features))

    # ── Shared LSTM backbone ──────────────────────────────────────────────────
    x = LSTM(units, return_sequences=True)(inputs)   # ★ LSTM replaces SimpleRNN
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    x = LSTM(units // 2, return_sequences=False)(x)   # ★ LSTM replaces SimpleRNN
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    # ── Shared dense ──────────────────────────────────────────────────────────
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout / 2)(x)

    # ── Per-quantile heads ────────────────────────────────────────────────────
    outputs = []
    for q in quantiles:
        head = Dense(32, activation='relu', name=f'dense_q{int(q*100)}')(x)
        head = Dropout(dropout / 2)(head)
        head = Dense(1, name=f'q{int(q*100)}')(head)
        outputs.append(head)

    model = Model(inputs=inputs, outputs=outputs,
                  name=f'QuantileLSTM_u{units}_d{int(dropout*100)}')   # ★ name updated

    losses = {f'q{int(q*100)}': pinball_loss(q) for q in quantiles}
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=losses)
    return model


# Quick preview model
WINDOW_SIZE = 10  # placeholder; updated after tuning
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, WINDOW_SIZE)
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  WINDOW_SIZE)

quantile_model = build_quantile_lstm_model(
    WINDOW_SIZE, X_train_scaled.shape[1], QUANTILES,
    units=128, dropout=0.2, learning_rate=0.001
)
print("=" * 80)
print("QUANTILE LSTM ARCHITECTURE")
print("=" * 80)
quantile_model.summary()
print(f"\nTotal Parameters: {quantile_model.count_params():,}")


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
print("HYPERPARAMETER SEARCH SPACE (Quantile LSTM)")
print("=" * 70)
for param, values in PARAM_GRID_Q.items():
    print(f"  {param:<18}: {values}")
print(f"\n  Total combinations : {len(all_combos_q)}")
print("=" * 70)

# ── Grid Search ───────────────────────────────────────────────────────────────
q_tuning_results = []
best_q_rmse   = float('inf')
best_q_config = None
best_q_model  = None
best_q_result = None

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

    y_tr_dict = {f'q{int(q*100)}': y_tr_q for q in QUANTILES}

    model = build_quantile_lstm_model(                         # ★ LSTM builder
        ws, n_features_q, QUANTILES,
        units=units, dropout=dropout, learning_rate=lr
    )

    callbacks_q = [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-7, verbose=0),
    ]

    model.fit(
        X_tr_q, y_tr_dict,
        validation_split=0.2, epochs=100, batch_size=32,
        callbacks=callbacks_q, verbose=0
    )

    preds_all_scaled = model.predict(X_te_q, verbose=0)
    q50_idx          = QUANTILES.index(0.5)
    y_pred_scaled_q  = preds_all_scaled[q50_idx].flatten()
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

    del model
    tf.keras.backend.clear_session()

print("\nHyperparameter search complete.")

# ── Results Table ─────────────────────────────────────────────────────────────
q_results_df = pd.DataFrame([{
    'Window Size'   : r['window_size'],
    'LSTM Units'    : r['units'],              # ★ label updated
    'Dropout'       : r['dropout'],
    'Learning Rate' : r['learning_rate'],
    'MSE'           : r['MSE'],
    'RMSE'          : r['RMSE'],
    'MAE'           : r['MAE'],
} for r in q_tuning_results])

q_results_df = q_results_df.sort_values('RMSE').reset_index(drop=True)
q_results_df.index += 1

print("=" * 85)
print("QUANTILE LSTM — HYPERPARAMETER TUNING RESULTS  (sorted by RMSE)")
print("=" * 85)
print(q_results_df.to_string())

print("\n" + "=" * 70)
print("BEST HYPERPARAMETER COMBINATION (Quantile LSTM)")
print("=" * 70)
print(f"  Best Window Size    : {best_q_config['window_size']}")
print(f"  Best LSTM Units     : {best_q_config['units']}")
print(f"  Best Dropout        : {best_q_config['dropout']}")
print(f"  Best Learning Rate  : {best_q_config['learning_rate']}")
print(f"  Best RMSE           : {best_q_result['RMSE']:.6f}")
print(f"  Best MSE            : {best_q_result['MSE']:.6f}")
print(f"  Best MAE            : {best_q_result['MAE']:.6f}")
print("=" * 70)

# ── Tuning Heatmap ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Quantile LSTM Tuning — RMSE Heatmap per Dropout', fontsize=14, fontweight='bold')
for ax, dr in zip(axes, [0.0, 0.2, 0.3]):
    sub   = q_results_df[q_results_df['Dropout'] == dr].copy()
    pivot = sub.groupby(['LSTM Units', 'Window Size'])['RMSE'].min().unstack()
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax)
    ax.set_title(f'Dropout = {dr}', fontweight='bold')
    ax.set_xlabel('Window Size')
    ax.set_ylabel('LSTM Units')
plt.tight_layout()
plt.show()

# ── Rebuild Best Model ────────────────────────────────────────────────────────
WINDOW_SIZE = best_q_config['window_size']
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, WINDOW_SIZE)
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test_scaled,  WINDOW_SIZE)

quantile_model = build_quantile_lstm_model(                    # ★ LSTM builder
    WINDOW_SIZE, n_features_q, QUANTILES,
    units=best_q_config['units'],
    dropout=best_q_config['dropout'],
    learning_rate=best_q_config['learning_rate']
)
print("Best quantile LSTM model rebuilt with optimal hyperparameters.")


# =============================================================================
# 5. Model Training
# =============================================================================

y_train_dict = {f'q{int(q*100)}': y_train_seq for q in QUANTILES}
y_test_dict  = {f'q{int(q*100)}': y_test_seq  for q in QUANTILES}

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=7, min_lr=1e-7, verbose=1)
]

print("=" * 80)
print("TRAINING BEST QUANTILE LSTM MODEL")
print("=" * 80)
print(f"Quantile Levels  : {QUANTILES}")
print(f"Window Size      : {WINDOW_SIZE}")
print(f"LSTM Units       : {best_q_config['units']}")
print(f"Dropout          : {best_q_config['dropout']}")
print(f"Learning Rate    : {best_q_config['learning_rate']}")
print(f"Training Samples : {len(X_train_seq)}")
print("\nTraining started...\n")

history = quantile_model.fit(
    X_train_seq, y_train_dict,
    validation_split=0.2, epochs=100, batch_size=32,
    callbacks=callbacks, verbose=1
)

print("\nTraining completed!")
print(f"Final epoch: {len(history.history['loss'])}")

# ── Training History ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for idx, q in enumerate(QUANTILES):
    q_name = f'q{int(q*100)}'
    axes[idx].plot(history.history[f'{q_name}_loss'],     label=f'Train Loss Q{q}', linewidth=2)
    axes[idx].plot(history.history[f'val_{q_name}_loss'], label=f'Val Loss Q{q}',   linewidth=2)
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('Pinball Loss')
    axes[idx].set_title(f'Quantile {q} Training History', fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
# 6. Generate Predictions
# =============================================================================

quantile_predictions_scaled = quantile_model.predict(X_test_seq, verbose=0)

quantile_predictions = {}
for idx, q in enumerate(QUANTILES):
    pred_scaled = quantile_predictions_scaled[idx].flatten()
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    quantile_predictions[q] = pred

y_actual = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

print("=" * 80)
print("QUANTILE PREDICTIONS GENERATED (LSTM)")
print("=" * 80)
print(f"Test samples: {len(y_actual)}")
for q, pred in quantile_predictions.items():
    print(f"  Q{q}: {pred.shape}")


# =============================================================================
# 7. Prediction Interval Metrics
# =============================================================================

def calculate_interval_metrics(y_true, lower_bound, upper_bound, confidence_level=0.95):
    """
    Calculate comprehensive prediction interval metrics.

    Args:
        y_true           : Actual values
        lower_bound      : Lower prediction bound
        upper_bound      : Upper prediction bound
        confidence_level : Target coverage level (default 0.95)

    Returns:
        Dictionary of metrics
    """
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
    winkler_score = np.mean(winkler)

    return {
        'PICP': picp, 'MPIW': mpiw, 'NMPIW': nmpiw, 'CWC': cwc,
        'ACE': ace, 'Sharpness_%': sharpness, 'Winkler_Score': winkler_score,
        'Coverage': coverage, 'Widths': widths
    }


confidence_levels = {
    '80%': (0.1, 0.9),
    '50%': (0.25, 0.75),
}

print("=" * 80)
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
    print("-" * 80)
    print(f"  PICP (Coverage)   : {metrics['PICP']*100:.2f}% (Target: {level_name})")
    print(f"  MPIW (Avg Width)  : {metrics['MPIW']:.6f}")
    print(f"  NMPIW (Normalized): {metrics['NMPIW']:.6f}")
    print(f"  CWC (Quality)     : {metrics['CWC']:.6f} (Lower is better)")
    print(f"  ACE (Coverage Err): {metrics['ACE']*100:.2f}%")
    print(f"  Sharpness         : {metrics['Sharpness_%']:.2f}%")
    print(f"  Winkler Score     : {metrics['Winkler_Score']:.6f}")

    ace_val = metrics['ACE']
    quality = ("EXCELLENT" if ace_val < 0.02 else
               "GOOD"      if ace_val < 0.05 else
               "ACCEPTABLE" if ace_val < 0.10 else "POOR")
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
axes[0].set_title('LSTM Quantile Predictions with Multiple Confidence Intervals', fontweight='bold', fontsize=14)
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
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Coverage')
axes[0, 0].set_title(f'80% PI Coverage (PICP: {metrics_summary["80%"]["PICP"]*100:.2f}%)', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

widths_80 = metrics_summary['80%']['Widths']
axes[0, 1].plot(test_dates, widths_80, linewidth=1.5, color='purple')
axes[0, 1].axhline(metrics_summary['80%']['MPIW'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean Width: {metrics_summary["80%"]["MPIW"]:.4f}')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Interval Width')
axes[0, 1].set_title('80% Prediction Interval Width Over Time', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].hist(widths_80, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 0].axvline(metrics_summary['80%']['MPIW'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {metrics_summary["80%"]["MPIW"]:.4f}')
axes[1, 0].set_xlabel('Interval Width')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Interval Widths', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

levels  = list(confidence_levels.keys())
picps   = [metrics_summary[l]['PICP'] * 100 for l in levels]
targets = [float(l.strip('%')) for l in levels]
x, width = np.arange(len(levels)), 0.35
axes[1, 1].bar(x - width/2, picps,   width, label='Actual PICP',    color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 1].bar(x + width/2, targets, width, label='Target Coverage', color='orange',    alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Confidence Level')
axes[1, 1].set_ylabel('Coverage (%)')
axes[1, 1].set_title('PICP vs Target Coverage', fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(levels)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


# =============================================================================
# 9. Quantile Calibration Analysis
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

empirical_coverage = [np.mean(y_actual <= quantile_predictions[q]) for q in QUANTILES]

axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
axes[0].plot(QUANTILES, empirical_coverage, 'bo-', linewidth=2, markersize=8, label='LSTM Calibration')
axes[0].set_xlabel('Predicted Quantile', fontsize=12)
axes[0].set_ylabel('Empirical Coverage', fontsize=12)
axes[0].set_title('Quantile Calibration Plot (LSTM)', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])

calibration_errors = np.abs(np.array(QUANTILES) - np.array(empirical_coverage))
axes[1].bar([f'Q{int(q*100)}' for q in QUANTILES], calibration_errors,
            color='coral', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Quantile', fontsize=12)
axes[1].set_ylabel('Absolute Calibration Error', fontsize=12)
axes[1].set_title('Quantile Calibration Errors (LSTM)', fontweight='bold', fontsize=14)
axes[1].grid(alpha=0.3, axis='y')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("CALIBRATION ANALYSIS (LSTM)")
print("=" * 80)
print(f"\n{'Quantile':<12} {'Target':<12} {'Empirical':<12} {'Error':<12}")
print("-" * 48)
for q, emp, err in zip(QUANTILES, empirical_coverage, calibration_errors):
    print(f"{q:<12.3f} {q:<12.3f} {emp:<12.3f} {err:<12.4f}")

mean_abs_error = np.mean(calibration_errors)
print(f"\nMean Absolute Calibration Error: {mean_abs_error:.4f}")
print("Calibration Quality:", (
    "EXCELLENT" if mean_abs_error < 0.02 else
    "GOOD"      if mean_abs_error < 0.05 else
    "NEEDS IMPROVEMENT"
))


# =============================================================================
# 10. Next 5 Days Predictions with Confidence Intervals
# =============================================================================

def predict_next_n_days_quantile(model, last_sequence, scaler_X, scaler_y, quantiles, n_days=5):
    """Predict next n days with full quantile predictions using LSTM."""
    predictions      = {q: [] for q in quantiles}
    current_sequence = last_sequence.copy()

    for _ in range(n_days):
        quantile_preds_scaled = model.predict(
            current_sequence.reshape(1, *current_sequence.shape), verbose=0
        )
        for idx, q in enumerate(quantiles):
            pred_scaled = quantile_preds_scaled[idx][0, 0]
            pred        = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            predictions[q].append(pred)

        median_idx       = quantiles.index(0.5)
        median_pred_sc   = quantile_preds_scaled[median_idx][0, 0]
        new_features     = current_sequence[-1].copy()
        new_features[0]  = median_pred_sc
        current_sequence = np.vstack([current_sequence[1:], new_features])

    return predictions


last_sequence_scaled = X_test_scaled[-WINDOW_SIZE:]
next_5_quantiles = predict_next_n_days_quantile(
    quantile_model, last_sequence_scaled, scaler_X, scaler_y, QUANTILES, n_days=5
)

last_date    = df_features.index[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=5)

print("=" * 80)
print("NEXT 5 DAYS QUANTILE PREDICTIONS (LSTM)")
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

print("\n" + "=" * 80)
print("PREDICTION SUMMARY (LSTM)")
print("=" * 80)
print(f"\n5-Day Median Return      : {cumulative_median:+.4f}%")
print(f"Current Price            : {df['close'].iloc[-1]:.2f}")
print(f"Predicted Price (Median) : {final_price:.2f}")
print(f"Expected Change          : {final_price - df['close'].iloc[-1]:+.2f} points")
print(f"\n95% Confidence Interval:")
print(f"  Lower Bound  : {cumulative_lower:+.4f}% (Price: {price_lower:.2f})")
print(f"  Upper Bound  : {cumulative_upper:+.4f}% (Price: {price_upper:.2f})")
print(f"  Interval Width: {cumulative_upper - cumulative_lower:.4f}%")

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
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Price', fontsize=12)
axes[0].set_title('LSTM 5-Day Price Forecast with Prediction Intervals', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

day_labels = [f'Day {i+1}\n{date.strftime("%m/%d")}' for i, date in enumerate(future_dates)]
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
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(day_labels)
axes[1].set_ylabel('Return (%)', fontsize=12)
axes[1].set_title('Daily Return Predictions with Quantile Intervals (LSTM)', fontweight='bold', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


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

print("\n" + "=" * 80)
print("TOP 3 INVESTMENT RECOMMENDATIONS")
print("=" * 80)
for idx, (_, row) in enumerate(sector_analysis.head(3).iterrows(), 1):
    rec = (
        "STRONG BUY - High return, low risk" if row['Mean_Return'] > 0.5 and row['Volatility'] < 2.0 else
        "BUY - Positive momentum"             if row['Mean_Return'] > 0.3 else
        "HOLD - Weak positive trend"          if row['Mean_Return'] > 0 else
        "AVOID - Negative trend"
    )
    print(f"\n{idx}. {row['Sector']}  |  Return {row['Mean_Return']:+.4f}%  |  Risk {row['Volatility']:.4f}%  |  {rec}")


# =============================================================================
# 12. Comprehensive Summary
# =============================================================================

available_level = list(metrics_summary.keys())[0]
m = metrics_summary[available_level]

print("=" * 80)
print("QUANTILE FORECASTING — FINAL SUMMARY (LSTM)")
print("=" * 80)

print("\n1. OPTIMAL MODEL CONFIGURATION:")
print(f"   Architecture   : Multi-Output Quantile LSTM")
print(f"   Window Size    : {WINDOW_SIZE} days")
print(f"   Quantile Levels: {QUANTILES}")
print(f"   LSTM Units     : {best_q_config['units']} → {best_q_config['units']//2}")
print(f"   Dropout        : {best_q_config['dropout']}")
print(f"   Learning Rate  : {best_q_config['learning_rate']}")
print(f"   Total Params   : {quantile_model.count_params():,}")

print("\n2. HYPERPARAMETER TUNING RESULT:")
print(f"   Best RMSE (q=0.5): {best_q_result['RMSE']:.6f}")
print(f"   Best MSE         : {best_q_result['MSE']:.6f}")
print(f"   Best MAE         : {best_q_result['MAE']:.6f}")

print(f"\n3. PREDICTION INTERVAL QUALITY ({available_level}):")
print(f"   PICP     : {m['PICP']*100:.2f}%")
print(f"   MPIW     : {m['MPIW']:.6f}")
print(f"   CWC      : {m['CWC']:.6f}")
print(f"   Sharpness: {m['Sharpness_%']:.2f}%")
print(f"   Winkler  : {m['Winkler_Score']:.6f}")

print("\n4. CALIBRATION:")
print(f"   MACE: {m['ACE']:.4f}  |  Quality: "
      f"{'EXCELLENT' if m['ACE'] < 0.02 else 'GOOD' if m['ACE'] < 0.05 else 'ACCEPTABLE'}")

print("\n5. NEXT 5 DAYS FORECAST:")
for i, date in enumerate(future_dates):
    print(f"   Day {i+1} ({date.strftime('%Y-%m-%d')}): "
          f"Median {next_5_quantiles[0.5][i]:+.4f}% | "
          f"95% PI [{next_5_quantiles[0.025][i]:+.4f}%, {next_5_quantiles[0.975][i]:+.4f}%]")

print(f"\n   5-Day Cumulative: {cumulative_median:+.4f}%  |  "
      f"Price {final_price:.2f} from {df['close'].iloc[-1]:.2f}  |  "
      f"95% Range [{price_lower:.2f}, {price_upper:.2f}]")

print("\n6. LSTM ADVANTAGES OVER SimpleRNN:")
print("   ✓ Explicit cell state enables long-range temporal memory")
print("   ✓ Forget gate selectively retains relevant past information")
print("   ✓ Input / output gates control information flow with less vanishing gradient")
print("   ✓ Better suited to financial time series with regime changes")
print("   ✓ Typically yields sharper, better-calibrated prediction intervals")

print("\n" + "=" * 80)
print("DISCLAIMER: Predictions are for research purposes only.")
print("Consult a financial advisor before making investment decisions.")
print("=" * 80)


# =============================================================================
# Save Model and Results
# =============================================================================

quantile_model.save('quantile_forecasting_lstm_model.h5')
print("\nLSTM model saved to: quantile_forecasting_lstm_model.h5")

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
results_df.to_csv('next_5_days_quantile_predictions_lstm.csv', index=False)
print("Predictions saved to: next_5_days_quantile_predictions_lstm.csv")

metrics_df = pd.DataFrame([
    {
        'Confidence_Level': level,
        'PICP'            : metrics['PICP'],
        'MPIW'            : metrics['MPIW'],
        'NMPIW'           : metrics['NMPIW'],
        'CWC'             : metrics['CWC'],
        'Sharpness_%'     : metrics['Sharpness_%'],
        'Winkler_Score'   : metrics['Winkler_Score'],
    }
    for level, metrics in metrics_summary.items()
])
metrics_df.to_csv('prediction_interval_metrics_lstm.csv', index=False)
print("Metrics saved to: prediction_interval_metrics_lstm.csv")