"""
Nifty50 % Change Forecasting — LSTM | Two Separate Quantile Models
+ Window Size Grid Search over [20, 30, 40]
==================================================================
Upper bound model  : q = 0.95
Lower bound model  : q = 0.05
Horizons           : t+1 … t+5 (next-day % change)
Grid search        : window_size ∈ {20, 30, 40}  — picks best by mean PICP
Metrics            : PICP, MPIW, Winkler Score
Device             : CUDA if available, else CPU
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings; warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZES = [20, 30, 40]
HORIZONS     = 5
HIDDEN       = 96
LAYERS       = 2
BATCH        = 64
EPOCHS       = 100
LR           = 8e-4
TRAIN_RATIO  = 0.80
Q_UPPER      = 0.95
Q_LOWER      = 0.05
NOMINAL      = Q_UPPER - Q_LOWER
print(f"Device : {DEVICE}  |  Grid: window_sizes={WINDOW_SIZES}")

# ── 1. Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("../nifty_final_dataset.csv", parse_dates=["date"]).sort_values("date")
df["target"] = df["close"].pct_change() * 100
df.dropna(inplace=True)

CANDIDATE = ["log_ret","vol_5","vol_15","dist_ma_5","dist_ma_20",
             "rsi_14","macd","bb_width","body","upper_wick","lower_wick",
             "close_pos","vix","ret_5","ret_15"]
FEATURES  = [c for c in CANDIDATE if c in df.columns]
print(f"Features : {FEATURES}\n")

scaler = StandardScaler()
X_all  = scaler.fit_transform(df[FEATURES].values).astype(np.float32)
y_all  = df["target"].values.astype(np.float32)

# ── 2. Helpers ──────────────────────────────────────────────────────────────────
def make_sequences(X, y, seq, horizons):
    Xs, Ys = [], []
    for i in range(len(X) - seq - horizons + 1):
        Xs.append(X[i:i+seq])
        Ys.append(y[i+seq : i+seq+horizons])
    return np.array(Xs), np.array(Ys)

def get_loaders(seq_len):
    X_seq, Y_seq = make_sequences(X_all, y_all, seq_len, HORIZONS)
    split = int(len(X_seq) * TRAIN_RATIO)
    tr = TensorDataset(torch.from_numpy(X_seq[:split]), torch.from_numpy(Y_seq[:split]))
    te = TensorDataset(torch.from_numpy(X_seq[split:]), torch.from_numpy(Y_seq[split:]))
    return (DataLoader(tr, BATCH, shuffle=True),
            DataLoader(te, BATCH, shuffle=False),
            Y_seq[split:])

# ── 3. LSTM Model ──────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, in_dim, hidden, layers, out_dim):
        super().__init__()

        self.lstm = nn.LSTM(
            in_dim,
            hidden,
            layers,
            batch_first=True,
            dropout=0.25,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)

        # Residual connection (IMPORTANT)
        last = out[:, -1, :]
        return self.fc(last)

def pinball(pred, target, q):
    e = target - pred
    return torch.mean(torch.maximum(q * e, (q - 1) * e))

# ── 4. Train one quantile model ─────────────────────────────────────────────────
def train_one(tr_dl, q, label):
    
    model = LSTMModel(len(FEATURES), HIDDEN, LAYERS, HORIZONS).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5
    )

    best_loss = float("inf")
    patience = 10
    wait = 0

    log = []

    for ep in range(1, EPOCHS+1):

        model.train()
        ep_loss = 0

        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            opt.zero_grad()
            pred = model(xb)

            loss = pinball(pred, yb, q)
            loss.backward()

            # ✅ Gradient clipping (VERY IMPORTANT)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()
            ep_loss += loss.item()

        ep_loss /= len(tr_dl)
        log.append(ep_loss)

        scheduler.step(ep_loss)

        # ✅ Early stopping
        if ep_loss < best_loss:
            best_loss = ep_loss
            wait = 0
        else:
            wait += 1

        if ep % 20 == 0:
            print(f"    [{label}] Ep {ep:3d}  loss={ep_loss:.4f}")

        if wait >= patience:
            print(f"    [{label}] Early stopping at epoch {ep}")
            break

    return model, log

def infer(model, te_dl):
    model.eval(); out = []
    with torch.no_grad():
        for xb, _ in te_dl:
            out.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(out)

def compute_metrics(p_lo, p_up, act):
    picp      = ((act >= p_lo) & (act <= p_up)).mean(axis=0)
    avg_width = (p_up - p_lo).mean(axis=0)
    alpha     = 1 - NOMINAL
    winkler   = (avg_width
                 + np.where(act < p_lo, 2/alpha*(p_lo - act), 0)
                 + np.where(act > p_up, 2/alpha*(act - p_up), 0)).mean(axis=0)
    return picp, avg_width, winkler

# ── 5. Grid Search ──────────────────────────────────────────────────────────────
results = {}

for win in WINDOW_SIZES:
    print(f"\n{'='*55}\n Window = {win}\n{'='*55}")
    tr_dl, te_dl, act = get_loaders(win)

    print(f"  → Upper LSTM (q={Q_UPPER})")
    m_up, h_up = train_one(tr_dl, Q_UPPER, "UP")
    print(f"  → Lower LSTM (q={Q_LOWER})")
    m_lo, h_lo = train_one(tr_dl, Q_LOWER, "LO")

    p_up = infer(m_up, te_dl)
    p_lo = infer(m_lo, te_dl)
    p_lo = np.minimum(p_lo, p_up)
    p_up = np.maximum(p_up, p_lo)
    picp, width, wink = compute_metrics(p_lo, p_up, act)

    results[win] = dict(picp=picp, width=width, winkler=wink,
                        p_up=p_up, p_lo=p_lo, act=act,
                        h_up=h_up, h_lo=h_lo)
    print(f"  Mean PICP={picp.mean():.3f}  MPIW={width.mean():.4f}%")

# ── 6. Best window ─────────────────────────────────────────────────────────────
best_win = max(results, key=lambda w: results[w]["picp"].mean())
print(f"\n★ Best window = {best_win}  (mean PICP={results[best_win]['picp'].mean():.3f})")

# ── 7. Summary Table ────────────────────────────────────────────────────────────
print("\n╔" + "═"*65 + "╗")
print("║   LSTM Two-Model Grid Search Summary                             ║")
print("╠" + "═"*8 + "╦" + "═"*8 + "╦" + "═"*10 + "╦" + "═"*10 + "╦" + "═"*12 + "╦" + "═"*12 + "╣")
print("║  Win   ║ Horiz  ║   PICP   ║  Width   ║  Winkler   ║  Best?     ║")
print("╠" + "═"*8 + "╬" + "═"*8 + "╬" + "═"*10 + "╬" + "═"*10 + "╬" + "═"*12 + "╬" + "═"*12 + "╣")
for win in WINDOW_SIZES:
    r = results[win]
    for h in range(HORIZONS):
        flag  = "✓" if r["picp"][h] >= NOMINAL else "✗"
        bmark = " ★ BEST" if win == best_win and h == 0 else ""
        print(f"║  {win:<5} ║  t+{h+1}  ║ {r['picp'][h]:.3f}{flag}  ║ {r['width'][h]:7.4f}% ║{r['winkler'][h]:10.4f}  ║{bmark:<12}║")
    print("╠" + "═"*8 + "╬" + "═"*8 + "╬" + "═"*10 + "╬" + "═"*10 + "╬" + "═"*12 + "╬" + "═"*12 + "╣")
print("╚" + "═"*65 + "╝")

# ── 8. Plots ───────────────────────────────────────────────────────────────────
WIN_COLORS = {20:"#E91E63", 30:"#009688", 40:"#FF5722"}
H_COLORS   = ["#1E90FF","#32CD32","#FFA500","#9932CC","#DC143C"]
PLOT_N     = 100

fig = plt.figure(figsize=(24, 22))
gs  = gridspec.GridSpec(5, 3, hspace=0.55, wspace=0.35)
fig.suptitle(f"Nifty50 % Change — LSTM  |  Two Quantile Models  |  Window Grid Search {WINDOW_SIZES}\n"
             f"Upper q={Q_UPPER}  Lower q={Q_LOWER}  |  Best window={best_win} ★",
             fontsize=13, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# — Row 0: Loss per window
for k, win in enumerate(WINDOW_SIZES):
    ax = fig.add_subplot(gs[0, k])
    ax.plot(results[win]["h_up"], color="crimson",   lw=1.5, label=f"q={Q_UPPER}")
    ax.plot(results[win]["h_lo"], color="steelblue", lw=1.5, label=f"q={Q_LOWER}")
    star = " ★" if win == best_win else ""
    ax.set_title(f"LSTM Loss — window={win}{star}",
                 fontweight="bold" if win==best_win else "normal")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Pinball Loss")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# — Rows 1-2: Best window — per-horizon intervals
r = results[best_win]
for h in range(HORIZONS):
    row, col = divmod(h, 3)
    ax = fig.add_subplot(gs[row+1, col])
    idx = np.arange(PLOT_N)
    a, lo, up = r["act"][-PLOT_N:,h], r["p_lo"][-PLOT_N:,h], r["p_up"][-PLOT_N:,h]
    ax.fill_between(idx, lo, up, alpha=0.18, color="gold", label="90% Band")
    ax.plot(idx, a,  color="black",     lw=1.2, label="Actual")
    ax.plot(idx, up, "--", color="crimson",   lw=0.9, label=f"q={Q_UPPER}")
    ax.plot(idx, lo, "--", color="steelblue", lw=0.9, label=f"q={Q_LOWER}")
    flag = "✓" if r["picp"][h] >= NOMINAL else "✗"
    ax.set_title(f"t+{h+1} | win={best_win}★ | PICP={r['picp'][h]:.3f}{flag}  W={r['winkler'][h]:.3f}",
                 color=H_COLORS[h], fontweight="bold")
    ax.set_xlabel("Test samples"); ax.set_ylabel("% Change")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# — Row 3: PICP comparison
ax_cmp = fig.add_subplot(gs[3, :2])
x, bw = np.arange(HORIZONS), 0.25
for k, win in enumerate(WINDOW_SIZES):
    bars = ax_cmp.bar(x + (k-1)*bw, results[win]["picp"], bw,
                      label=f"win={win}", color=WIN_COLORS[win],
                      alpha=0.85, edgecolor="black" if win==best_win else "white",
                      linewidth=2 if win==best_win else 0.5)
ax_cmp.axhline(NOMINAL, color="black", ls="--", lw=1.5, label=f"Nominal {NOMINAL:.0%}")
ax_cmp.set_xticks(x); ax_cmp.set_xticklabels([f"t+{h+1}" for h in range(HORIZONS)])
ax_cmp.set_ylim(0, 1.15); ax_cmp.set_title("PICP — All Windows vs Nominal")
ax_cmp.set_ylabel("Coverage"); ax_cmp.legend(); ax_cmp.grid(alpha=0.3, axis="y")
plt.tight_layout(rect=[0, 0, 1, 0.96])

# — Row 3: MPIW comparison
ax_mpiw = fig.add_subplot(gs[3, 2])
for win in WINDOW_SIZES:
    lw = 2.5 if win == best_win else 1.2
    ax_mpiw.plot([f"t+{h+1}" for h in range(HORIZONS)], results[win]["width"],
                 "o-", lw=lw, color=WIN_COLORS[win],
                 label=f"win={win}" + (" ★" if win==best_win else ""))
ax_mpiw.set_title("MPIW per Horizon"); ax_mpiw.set_ylabel("Width (%)")
ax_mpiw.legend(fontsize=8); ax_mpiw.grid(alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# — Row 4: Winkler comparison
ax_wink = fig.add_subplot(gs[4, :2])
for win in WINDOW_SIZES:
    lw = 2.5 if win == best_win else 1.2
    ax_wink.plot([f"t+{h+1}" for h in range(HORIZONS)], results[win]["winkler"],
                 "s-", lw=lw, color=WIN_COLORS[win],
                 label=f"win={win}" + (" ★" if win==best_win else ""))
ax_wink.set_title("Winkler Score per Horizon  (lower = better)")
ax_wink.set_ylabel("Winkler Score"); ax_wink.legend(); ax_wink.grid(alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# — Row 4: Best window selection bar
ax_sel = fig.add_subplot(gs[4, 2])
mean_p = [results[w]["picp"].mean() for w in WINDOW_SIZES]
bars   = ax_sel.bar([f"win={w}" for w in WINDOW_SIZES], mean_p,
                    color=[WIN_COLORS[w] for w in WINDOW_SIZES], edgecolor="white")
bars[WINDOW_SIZES.index(best_win)].set_edgecolor("black"); bars[WINDOW_SIZES.index(best_win)].set_linewidth(2.5)
ax_sel.axhline(NOMINAL, color="black", ls="--", lw=1.3, label=f"Nominal {NOMINAL:.0%}")
ax_sel.set_ylim(0, 1.1); ax_sel.set_title("Mean PICP — Best Window Selection ★")
ax_sel.legend()
for bar, v in zip(bars, mean_p):
    ax_sel.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
ax_sel.grid(alpha=0.3, axis="y")

plt.savefig("lstm_two_models.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved → lstm_two_models.png")