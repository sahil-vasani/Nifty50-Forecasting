"""
=============================================================================
Nifty50 % Change Forecasting — LSTM | Two Separate Quantile Models
Window Size Grid Search over [20, 30, 40]
=============================================================================
Upper bound model  : q = 0.95
Lower bound model  : q = 0.05
Horizons           : t+1 … t+5 (next-day % change)
Grid search        : window_size in {20, 30, 40}  — picks best by mean PICP
Metrics            : PICP, MPIW, Winkler Score
Plots              : 1) Per-horizon PI chart (best window), 2) Grid comparison
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
import random

warnings.filterwarnings("ignore")

# ── Random Seeds ──────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZES = [20, 30, 40]
HORIZONS     = 5
HIDDEN       = 256
LAYERS       = 4
BATCH        = 32
EPOCHS       = 100
LR           = 5e-4
TRAIN_RATIO  = 0.80
Q_UPPER      = 0.95          # upper quantile
Q_LOWER      = 0.05          # lower quantile
NOMINAL      = Q_UPPER - Q_LOWER   # 0.90 nominal coverage
PATIENCE     = 15            # early-stopping patience

print(f"Device : {DEVICE}  |  Grid: window_sizes={WINDOW_SIZES}")
print(f"Quantiles : q_lower={Q_LOWER}  q_upper={Q_UPPER}  |  Nominal={NOMINAL:.0%}")


# =============================================================================
# 1. Data Loading & Preprocessing
# =============================================================================

df = pd.read_csv("../nifty_final_dataset.csv", parse_dates=["date"]).sort_values("date")
df["target"] = df["close"].pct_change() * 100
df.dropna(inplace=True)

CANDIDATE = [
    "log_ret", "vol_5", "vol_15", "dist_ma_5", "dist_ma_20",
    "rsi_14", "macd", "bb_width", "body", "upper_wick", "lower_wick",
    "close_pos", "vix", "ret_5", "ret_15"
]
FEATURES = [c for c in CANDIDATE if c in df.columns]
print(f"\nFeatures used ({len(FEATURES)}): {FEATURES}")

scaler = StandardScaler()
X_all  = scaler.fit_transform(df[FEATURES].values).astype(np.float32)
y_all  = df["target"].values.astype(np.float32)


# =============================================================================
# 2. Sequence Helpers
# =============================================================================

def make_sequences(X, y, seq_len, horizons):
    """Create rolling-window (X, Y) pairs for multi-horizon forecasting."""
    Xs, Ys = [], []
    for i in range(len(X) - seq_len - horizons + 1):
        Xs.append(X[i : i + seq_len])
        Ys.append(y[i + seq_len : i + seq_len + horizons])
    return np.array(Xs), np.array(Ys)


def get_loaders(seq_len):
    X_seq, Y_seq = make_sequences(X_all, y_all, seq_len, HORIZONS)
    split = int(len(X_seq) * TRAIN_RATIO)
    tr = TensorDataset(
        torch.from_numpy(X_seq[:split]),
        torch.from_numpy(Y_seq[:split])
    )
    te = TensorDataset(
        torch.from_numpy(X_seq[split:]),
        torch.from_numpy(Y_seq[split:])
    )
    return (
        DataLoader(tr, BATCH, shuffle=True),
        DataLoader(te, BATCH, shuffle=False),
        Y_seq[split:]
    )


# =============================================================================
# 3. Model  ← KEY CHANGE: nn.RNN → nn.LSTM
# =============================================================================

class LSTMModel(nn.Module):
    """
    Multi-horizon LSTM with a shared backbone and linear output head.

    Differences from SimpleRNN version:
      • nn.RNN  → nn.LSTM  (adds input, forget, output gates + cell state)
      • Forward unpacks (h_n, c_n) tuple from LSTM hidden state
      • No `nonlinearity` arg (LSTM uses tanh + sigmoid internally)
      • All other hyper-params (hidden, layers, dropout, head) identical

    Args:
        in_dim  : number of input features
        hidden  : hidden size of LSTM
        layers  : number of stacked LSTM layers
        out_dim : number of forecast horizons
    """
    def __init__(self, in_dim, hidden, layers, out_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden, layers,
            batch_first=True,
            dropout=0.2          # applied between LSTM layers (not on last layer)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        # LSTM returns (output, (h_n, c_n))  ← tuple, unlike RNN's single h_n
        out, (h_n, c_n) = self.lstm(x)   # out: (B, T, hidden)
        return self.head(out[:, -1])      # take last time-step → (B, out_dim)


def pinball_loss(pred, target, q):
    """Asymmetric pinball (quantile) loss."""
    e = target - pred
    return torch.where(e >= 0, q * e, (q - 1) * e).mean()


# =============================================================================
# 4. Training with Early Stopping
# =============================================================================

def train_one_model(tr_dl, q, label, epochs=EPOCHS, patience=PATIENCE):
    """
    Train a single quantile LSTM model with early stopping.

    Returns:
        model   : trained LSTMModel
        log     : list of per-epoch training losses
    """
    model = LSTMModel(len(FEATURES), HIDDEN, LAYERS, HORIZONS).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    log            = []
    best_loss      = float("inf")
    best_weights   = None
    patience_count = 0

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = pinball_loss(model(xb), yb, q)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()

        ep_loss /= len(tr_dl)
        sched.step()
        log.append(ep_loss)

        if ep_loss < best_loss:
            best_loss    = ep_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"    [{label}] Early stop at epoch {ep}  best_loss={best_loss:.4f}")
                break

        if ep % 20 == 0:
            print(f"    [{label} q={q}] Ep {ep:3d}  loss={ep_loss:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, log


def infer(model, te_dl):
    """Run inference on test DataLoader, return numpy array (N, horizons)."""
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in te_dl:
            out.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(out)


# =============================================================================
# 5. Evaluation Metrics
# =============================================================================

def compute_metrics(p_lo, p_up, act):
    """
    Compute PICP, MPIW, and Winkler Score per horizon.

    Args:
        p_lo : (N, H) lower quantile predictions
        p_up : (N, H) upper quantile predictions
        act  : (N, H) actual values

    Returns:
        picp, avg_width, winkler — each shape (H,)
    """
    picp      = ((act >= p_lo) & (act <= p_up)).mean(axis=0)
    avg_width = (p_up - p_lo).mean(axis=0)
    alpha     = 1 - NOMINAL

    penalty = (
        np.where(act < p_lo, (2 / alpha) * (p_lo - act), 0) +
        np.where(act > p_up, (2 / alpha) * (act - p_up), 0)
    )
    winkler = (avg_width + penalty).mean(axis=0)
    return picp, avg_width, winkler


# =============================================================================
# 6. Grid Search
# =============================================================================

results = {}   # window -> {picp, width, winkler, p_up, p_lo, act, h_up, h_lo}

for win in WINDOW_SIZES:
    print(f"\n{'='*60}")
    print(f"  Window Size = {win}")
    print(f"{'='*60}")

    tr_dl, te_dl, act = get_loaders(win)

    print(f"  Training Upper LSTM model (q={Q_UPPER}) ...")
    m_up, h_up = train_one_model(tr_dl, Q_UPPER, "UP")

    print(f"  Training Lower LSTM model (q={Q_LOWER}) ...")
    m_lo, h_lo = train_one_model(tr_dl, Q_LOWER, "LO")

    p_up = infer(m_up, te_dl)
    p_lo = infer(m_lo, te_dl)

    # Enforce monotonicity: upper must be >= lower
    p_up = np.maximum(p_up, p_lo)

    picp, width, wink = compute_metrics(p_lo, p_up, act)

    results[win] = dict(
        picp=picp, width=width, winkler=wink,
        p_up=p_up, p_lo=p_lo, act=act,
        h_up=h_up, h_lo=h_lo
    )

    print(f"\n  Window={win}  Mean PICP={picp.mean():.3f}  "
          f"Mean MPIW={width.mean():.4f}%  Mean Winkler={wink.mean():.3f}")


# ── Best window by mean PICP ──────────────────────────────────────────────────
best_win = max(results, key=lambda w: results[w]["picp"].mean())
print(f"\n★ Best window = {best_win}  "
      f"(mean PICP = {results[best_win]['picp'].mean():.3f})")


# =============================================================================
# 7. Print Results Table
# =============================================================================

print("\n" + "╔" + "═"*62 + "╗")
print("║   LSTM Two-Model Grid Search Summary                         ║")
print("╠" + "═"*10 + "╦" + "═"*8 + "╦" + "═"*10 + "╦" + "═"*10 + "╦" + "═"*9 + "╦" + "═"*10 + "╣")
print("║  Window   ║ Horizon ║   PICP   ║   Width  ║ Winkler  ║ Best?    ║")
print("╠" + "═"*10 + "╬" + "═"*8 + "╬" + "═"*10 + "╬" + "═"*10 + "╬" + "═"*9 + "╬" + "═"*10 + "╣")

for win in WINDOW_SIZES:
    r = results[win]
    for h in range(HORIZONS):
        flag  = "✓" if r["picp"][h] >= NOMINAL else "✗"
        bmark = "★ BEST" if (win == best_win and h == 0) else ""
        print(f"║  {win:<8} ║  t+{h+1}   "
              f"║ {r['picp'][h]:.3f}{flag}  "
              f"║ {r['width'][h]:7.4f}% "
              f"║{r['winkler'][h]:8.3f}  "
              f"║{bmark:<10}║")
    print("╠" + "═"*10 + "╬" + "═"*8 + "╬" + "═"*10 + "╬" + "═"*10 + "╬" + "═"*9 + "╬" + "═"*10 + "╣")

print("╚" + "═"*62 + "╝")


# =============================================================================
# 8. Next 5-Day Forecast using Best Window
# =============================================================================

r_best = results[best_win]

# Use last `best_win` rows of X_all as the seed sequence
last_seq = X_all[-best_win:].copy()   # (win, features)

# Re-train best models for the forecast
tr_dl_b, _, _ = get_loaders(best_win)
print(f"\nRe-training LSTM models for window={best_win} for 5-day forecast ...")

m_up_b, _ = train_one_model(tr_dl_b, Q_UPPER, "FORECAST_UP", epochs=EPOCHS)
m_lo_b, _ = train_one_model(tr_dl_b, Q_LOWER, "FORECAST_LO", epochs=EPOCHS)

forecast_upper  = []
forecast_lower  = []
current_seq     = last_seq.copy()

for day in range(HORIZONS):
    x_in = torch.tensor(
        current_seq.reshape(1, best_win, len(FEATURES)),
        dtype=torch.float32
    ).to(DEVICE)
    with torch.no_grad():
        pred_up = m_up_b(x_in)[0, 0].cpu().item()
        pred_lo = m_lo_b(x_in)[0, 0].cpu().item()

    forecast_upper.append(pred_up)
    forecast_lower.append(pred_lo)

    # Roll window: use midpoint as synthetic next observation
    mid_pred       = (pred_up + pred_lo) / 2.0
    new_row        = current_seq[-1].copy()
    new_row[0]     = mid_pred   # update log_ret feature with prediction
    current_seq    = np.vstack([current_seq[1:], new_row])

forecast_mid = [(u + l) / 2 for u, l in zip(forecast_upper, forecast_lower)]
last_date    = df["date"].iloc[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=HORIZONS)

print("\n" + "="*70)
print("NEXT 5-DAY FORECAST  (Best Window = {best_win})".format(best_win=best_win))
print("="*70)
print(f"{'Date':<14} {'Lower (5%)':>12} {'Mid':>10} {'Upper (95%)':>12}")
print("-"*50)
for i, date in enumerate(future_dates):
    print(f"{date.strftime('%Y-%m-%d'):<14} "
          f"{forecast_lower[i]:>+10.4f}%  "
          f"{forecast_mid[i]:>+9.4f}%  "
          f"{forecast_upper[i]:>+10.4f}%")


# =============================================================================
# 9. Plots
# =============================================================================

PLOT_N    = 100
H_COLORS  = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#C62828"]
WIN_COLOR = {20: "#E91E63", 30: "#009688", 40: "#FF5722"}

# ── Figure 1: Best-window per-horizon PI chart ────────────────────────────────
fig1, axes = plt.subplots(HORIZONS, 1, figsize=(16, 4 * HORIZONS), sharex=False)
fig1.suptitle(
    f"Nifty50 % Change — LSTM  |  Two Quantile Models  |  Best Window = {best_win}\n"
    f"q_lower={Q_LOWER}  q_upper={Q_UPPER}  |  Nominal Coverage = {NOMINAL:.0%}",
    fontsize=13, fontweight="bold", y=1.002
)

r = results[best_win]
for h in range(HORIZONS):
    ax  = axes[h]
    idx = np.arange(PLOT_N)
    a   = r["act"][-PLOT_N:, h]
    lo  = r["p_lo"][-PLOT_N:, h]
    up  = r["p_up"][-PLOT_N:, h]

    # Shaded prediction band
    ax.fill_between(idx, lo, up, alpha=0.25, color=H_COLORS[h], label="90% PI Band")

    # Actual, upper, lower lines
    ax.plot(idx, a,  color="black",       lw=1.8, label="Actual",            zorder=3)
    ax.plot(idx, up, "--", color="crimson",   lw=1.2, label=f"q={Q_UPPER} (Upper)", alpha=0.85)
    ax.plot(idx, lo, "--", color="steelblue", lw=1.2, label=f"q={Q_LOWER} (Lower)", alpha=0.85)
    ax.axhline(0, color="grey", lw=0.8, ls=":", alpha=0.6)

    # Points inside/outside band
    inside  = (a >= lo) & (a <= up)
    outside = ~inside
    ax.scatter(idx[inside],  a[inside],  color="green", s=15, zorder=4, alpha=0.7, label="Inside PI")
    ax.scatter(idx[outside], a[outside], color="red",   s=25, zorder=5, marker="x", label="Outside PI")

    flag = "✓" if r["picp"][h] >= NOMINAL else "✗"
    ax.set_title(
        f"t+{h+1} Horizon  |  PICP = {r['picp'][h]:.3f}{flag}  "
        f"|  MPIW = {r['width'][h]:.4f}%  |  Winkler = {r['winkler'][h]:.3f}",
        color=H_COLORS[h], fontweight="bold", fontsize=11
    )
    ax.set_ylabel("% Change")
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.grid(alpha=0.25)

axes[-1].set_xlabel("Test Sample Index", fontsize=11)
plt.tight_layout()
fig1.savefig("plot1_pi_best_window.png", dpi=150, bbox_inches="tight")
print("\nSaved → plot1_pi_best_window.png")
plt.close(fig1)


# ── Figure 2: Grid comparison (PICP, MPIW, Winkler) + Forecast ───────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
fig2.suptitle(
    f"Nifty50 LSTM — Grid Search Comparison  |  Windows {WINDOW_SIZES}\n"
    f"Best Window = {best_win}  |  q_lower={Q_LOWER}  q_upper={Q_UPPER}",
    fontsize=13, fontweight="bold"
)

horizon_labels = [f"t+{h+1}" for h in range(HORIZONS)]

# -- (0,0) PICP comparison
ax = axes2[0, 0]
x  = np.arange(HORIZONS)
bar_w = 0.25
for k, win in enumerate(WINDOW_SIZES):
    offset = (k - 1) * bar_w
    bars = ax.bar(
        x + offset, results[win]["picp"], bar_w,
        label=f"win={win}", color=WIN_COLOR[win],
        alpha=0.85, edgecolor="white"
    )
    if win == best_win:
        for bar in bars:
            bar.set_edgecolor("black"); bar.set_linewidth(1.8)
ax.axhline(NOMINAL, color="black", ls="--", lw=1.5, label=f"Nominal {NOMINAL:.0%}")
ax.set_xticks(x); ax.set_xticklabels(horizon_labels)
ax.set_ylim(0, 1.15); ax.set_title("PICP by Window & Horizon", fontweight="bold")
ax.set_ylabel("Coverage"); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

# -- (0,1) MPIW comparison
ax = axes2[0, 1]
for win in WINDOW_SIZES:
    lw = 2.5 if win == best_win else 1.2
    ls = "-"  if win == best_win else "--"
    ax.plot(horizon_labels, results[win]["width"], "o" + ls, lw=lw,
            color=WIN_COLOR[win],
            label=f"win={win}" + (" ★" if win == best_win else ""))
ax.set_title("Mean Interval Width (MPIW)", fontweight="bold")
ax.set_ylabel("% Change Width"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# -- (1,0) Winkler Score comparison
ax = axes2[1, 0]
for win in WINDOW_SIZES:
    lw = 2.5 if win == best_win else 1.2
    ls = "-"  if win == best_win else "--"
    ax.plot(horizon_labels, results[win]["winkler"], "s" + ls, lw=lw,
            color=WIN_COLOR[win],
            label=f"win={win}" + (" ★" if win == best_win else ""))
ax.set_title("Winkler Score  (lower = better)", fontweight="bold")
ax.set_ylabel("Winkler Score"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# -- (1,1) 5-Day Forecast
ax = axes2[1, 1]
day_labels = [f"t+{d+1}\n{dt.strftime('%m/%d')}" for d, dt in enumerate(future_dates)]
bar_colors = ["#2E7D32" if m > 0 else "#C62828" for m in forecast_mid]
ax.bar(range(HORIZONS), forecast_mid, color=bar_colors, alpha=0.6, label="Mid Forecast")
yerr_lo = [forecast_mid[i] - forecast_lower[i] for i in range(HORIZONS)]
yerr_up = [forecast_upper[i] - forecast_mid[i] for i in range(HORIZONS)]
ax.errorbar(
    range(HORIZONS), forecast_mid,
    yerr=[yerr_lo, yerr_up],
    fmt="none", ecolor="black", capsize=8, capthick=2, linewidth=2,
    label=f"90% PI"
)
ax.axhline(0, color="black", ls="--", lw=1.2)
ax.set_xticks(range(HORIZONS)); ax.set_xticklabels(day_labels, fontsize=9)
ax.set_ylabel("% Change"); ax.set_title("5-Day Forecast with PI", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
fig2.savefig("plot2_grid_comparison.png", dpi=150, bbox_inches="tight")
print("Saved → plot2_grid_comparison.png")
plt.close(fig2)


# =============================================================================
# 10. Save Results to CSV
# =============================================================================

rows = []
for win in WINDOW_SIZES:
    r = results[win]
    for h in range(HORIZONS):
        rows.append({
            "Window_Size" : win,
            "Horizon"     : f"t+{h+1}",
            "PICP"        : round(r["picp"][h], 4),
            "MPIW_pct"    : round(r["width"][h], 4),
            "Winkler"     : round(r["winkler"][h], 4),
            "Best"        : "YES" if win == best_win else ""
        })

pd.DataFrame(rows).to_csv("grid_search_results.csv", index=False)
print("Saved → grid_search_results.csv")

# Forecast CSV
forecast_df = pd.DataFrame({
    "Date"       : [d.strftime("%Y-%m-%d") for d in future_dates],
    "Lower_5pct" : [round(v, 4) for v in forecast_lower],
    "Mid"        : [round(v, 4) for v in forecast_mid],
    "Upper_95pct": [round(v, 4) for v in forecast_upper],
})
forecast_df.to_csv("next5days_forecast.csv", index=False)
print("Saved → next5days_forecast.csv")


print("\n" + "="*70)
print("DONE — All outputs saved.")
print("  plot1_pi_best_window.png   — per-horizon PI chart (best window)")
print("  plot2_grid_comparison.png  — grid comparison + 5-day forecast")
print("  grid_search_results.csv    — full metrics table")
print("  next5days_forecast.csv     — 5-day return forecast")
print("="*70)
print(f"\nDISCLAIMER: For research purposes only. Not financial advice.")