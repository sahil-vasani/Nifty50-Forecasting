"""
Nifty50 % Change Forecasting — SimpleRNN | Two Separate Quantile Models
+ Window Size Grid Search over [20, 30, 40]
=======================================================================
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
WINDOW_SIZES = [20, 30, 40]          # ← grid search space
HORIZONS     = 5
HIDDEN       = 64
LAYERS       = 2
BATCH        = 64
EPOCHS       = 80
LR           = 1e-3
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

# ── 3. Model ───────────────────────────────────────────────────────────────────
class RNNModel(nn.Module):
    def __init__(self, in_dim, hidden, layers, out_dim):
        super().__init__()
        self.rnn  = nn.RNN(in_dim, hidden, layers, batch_first=True,
                           dropout=0.2, nonlinearity="tanh")
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 32), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, out_dim))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])

def pinball(pred, target, q):
    e = target - pred
    return torch.where(e >= 0, q * e, (q - 1) * e).mean()

# ── 4. Train one quantile model ─────────────────────────────────────────────────
def train_one(tr_dl, q, label, epochs=EPOCHS):
    model = RNNModel(len(FEATURES), HIDDEN, LAYERS, HORIZONS).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    log   = []
    for ep in range(1, epochs+1):
        model.train(); ep_loss = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pinball(model(xb), yb, q).backward()
            opt.step(); ep_loss += pinball(model(xb.detach()), yb.detach(), q).item()
        sched.step(); log.append(ep_loss / len(tr_dl))
        if ep % 20 == 0: print(f"    [{label}] Ep {ep:3d}  loss={log[-1]:.4f}")
    return model, log

def infer(model, te_dl):
    model.eval(); out = []
    with torch.no_grad():
        for xb, _ in te_dl:
            out.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(out)

def metrics(p_lo, p_up, act):
    picp      = ((act >= p_lo) & (act <= p_up)).mean(axis=0)
    avg_width = (p_up - p_lo).mean(axis=0)
    alpha     = 1 - NOMINAL
    winkler   = (avg_width
                 + np.where(act < p_lo, 2/alpha*(p_lo - act), 0)
                 + np.where(act > p_up, 2/alpha*(act - p_up), 0)).mean(axis=0)
    return picp, avg_width, winkler

# ── 5. Grid Search ──────────────────────────────────────────────────────────────
results = {}   # win -> {picp, width, winkler, p_up, p_lo, act, h_up, h_lo}

for win in WINDOW_SIZES:
    print(f"\n{'='*55}")
    print(f" Window = {win}")
    print(f"{'='*55}")
    tr_dl, te_dl, act = get_loaders(win)

    print(f"  → Upper model (q={Q_UPPER})")
    m_up, h_up = train_one(tr_dl, Q_UPPER, "UP")
    print(f"  → Lower model (q={Q_LOWER})")
    m_lo, h_lo = train_one(tr_dl, Q_LOWER, "LO")

    p_up = infer(m_up, te_dl)
    p_lo = infer(m_lo, te_dl)
    picp, width, wink = metrics(p_lo, p_up, act)

    results[win] = dict(picp=picp, width=width, winkler=wink,
                        p_up=p_up, p_lo=p_lo, act=act,
                        h_up=h_up, h_lo=h_lo)
    print(f"  Mean PICP={picp.mean():.3f}  MPIW={width.mean():.4f}%")

# ── 6. Pick best window ─────────────────────────────────────────────────────────
best_win = max(results, key=lambda w: results[w]["picp"].mean())
print(f"\n★ Best window = {best_win}  (mean PICP={results[best_win]['picp'].mean():.3f})")

# ── 7. Summary Table ────────────────────────────────────────────────────────────
print("\n╔" + "═"*62 + "╗")
print("║   RNN Two-Model Grid Search Summary                          ║")
print("╠" + "═"*10 + "╦" + "═"*8 + "╦" + "═"*10 + "╦" + "═"*10 + "╦" + "═"*9 + "╦" + "═"*10 + "╣")
print("║  Window   ║ Horizon ║   PICP   ║   Width  ║ Winkler  ║ Best?    ║")
print("╠" + "═"*10 + "╬" + "═"*8 + "╬" + "═"*10 + "╬" + "═"*10 + "╬" + "═"*9 + "╬" + "═"*10 + "╣")
for win in WINDOW_SIZES:
    r = results[win]
    for h in range(HORIZONS):
        flag  = "✓" if r["picp"][h] >= NOMINAL else "✗"
        bmark = "★ BEST" if win == best_win and h == 0 else ""
        print(f"║  {win:<8} ║  t+{h+1}   ║ {r['picp'][h]:.3f}{flag}  ║ {r['width'][h]:7.4f}% ║{r['winkler'][h]:8.3f}  ║{bmark:<10}║")
    print("╠" + "═"*10 + "╬" + "═"*8 + "╬" + "═"*10 + "╬" + "═"*10 + "╬" + "═"*9 + "╬" + "═"*10 + "╣")
print("╚" + "═"*62 + "╝")

# ── 8. Plots ───────────────────────────────────────────────────────────────────
H_COLORS  = ["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336"]
WIN_COLORS = {20:"#E91E63", 30:"#009688", 40:"#FF5722"}
PLOT_N = 100

fig = plt.figure(figsize=(22, 20))
gs  = gridspec.GridSpec(5, 3, hspace=0.55, wspace=0.35)
fig.suptitle(f"Nifty50 % Change — SimpleRNN  |  Two Quantile Models  |  Window Grid Search {WINDOW_SIZES}\n"
             f"q_upper={Q_UPPER}  q_lower={Q_LOWER}  |  Best window={best_win}",
             fontsize=13, fontweight="bold", y=0.99)

# — Row 0: Loss curves per window
for k, win in enumerate(WINDOW_SIZES):
    ax = fig.add_subplot(gs[0, k])
    ax.plot(results[win]["h_up"], color="crimson",   lw=1.5, label=f"Upper q={Q_UPPER}")
    ax.plot(results[win]["h_lo"], color="steelblue", lw=1.5, label=f"Lower q={Q_LOWER}")
    star = " ★" if win == best_win else ""
    ax.set_title(f"Loss — window={win}{star}", fontweight="bold" if win==best_win else "normal")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Pinball Loss")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

# — Row 1-2: Best window — per-horizon interval plots
r = results[best_win]
for h in range(HORIZONS):
    row, col = divmod(h, 3)
    ax = fig.add_subplot(gs[row+1, col])
    idx = np.arange(PLOT_N)
    a, lo, up = r["act"][-PLOT_N:,h], r["p_lo"][-PLOT_N:,h], r["p_up"][-PLOT_N:,h]
    ax.fill_between(idx, lo, up, alpha=0.20, color="gold", label="90% Band")
    ax.plot(idx, a,  color="black",     lw=1.2, label="Actual")
    ax.plot(idx, up, "--", color="crimson",   lw=0.9, label=f"q={Q_UPPER}")
    ax.plot(idx, lo, "--", color="steelblue", lw=0.9, label=f"q={Q_LOWER}")
    flag = "✓" if r["picp"][h] >= NOMINAL else "✗"
    ax.set_title(f"t+{h+1} | win={best_win}★ | PICP={r['picp'][h]:.3f}{flag}  W={r['width'][h]:.3f}%",
                 color=H_COLORS[h], fontweight="bold")
    ax.set_xlabel("Test samples"); ax.set_ylabel("% Change")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

# — Row 3: PICP comparison across windows
ax_cmp = fig.add_subplot(gs[3, :2])
x = np.arange(HORIZONS)
bar_w = 0.25
for k, win in enumerate(WINDOW_SIZES):
    offset = (k - 1) * bar_w
    bars = ax_cmp.bar(x + offset, results[win]["picp"], bar_w,
                      label=f"win={win}", color=WIN_COLORS[win],
                      alpha=0.85, edgecolor="white",
                      linewidth=2 if win==best_win else 0.5)
    if win == best_win:
        for bar in bars:
            bar.set_edgecolor("black")
ax_cmp.axhline(NOMINAL, color="black", ls="--", lw=1.5, label=f"Nominal {NOMINAL:.0%}")
ax_cmp.set_xticks(x); ax_cmp.set_xticklabels([f"t+{h+1}" for h in range(HORIZONS)])
ax_cmp.set_ylim(0, 1.15); ax_cmp.set_title("PICP Comparison — All Windows vs. Nominal")
ax_cmp.set_ylabel("Coverage"); ax_cmp.legend(); ax_cmp.grid(alpha=0.3, axis="y")

# — Row 3: MPIW comparison
ax_w = fig.add_subplot(gs[3, 2])
for win in WINDOW_SIZES:
    lw = 2.5 if win == best_win else 1.2
    ls = "-" if win == best_win else "--"
    ax_w.plot([f"t+{h+1}" for h in range(HORIZONS)], results[win]["width"],
              "o"+ls, lw=lw, color=WIN_COLORS[win],
              label=f"win={win}" + (" ★" if win==best_win else ""))
ax_w.set_title("Mean Interval Width (MPIW) per Horizon")
ax_w.set_ylabel("% Change Width"); ax_w.legend(fontsize=8); ax_w.grid(alpha=0.3)

# — Row 4: Winkler comparison
ax_wink = fig.add_subplot(gs[4, :2])
for win in WINDOW_SIZES:
    lw = 2.5 if win == best_win else 1.2
    ax_wink.plot([f"t+{h+1}" for h in range(HORIZONS)], results[win]["winkler"],
                 "s-", lw=lw, color=WIN_COLORS[win],
                 label=f"win={win}" + (" ★" if win==best_win else ""))
ax_wink.set_title("Winkler Score per Horizon  (lower = better)")
ax_wink.set_ylabel("Winkler Score"); ax_wink.legend(); ax_wink.grid(alpha=0.3)

# — Row 4: Mean-PICP summary bar
ax_sum = fig.add_subplot(gs[4, 2])
mean_picps = [results[w]["picp"].mean() for w in WINDOW_SIZES]
bars = ax_sum.bar([f"win={w}" for w in WINDOW_SIZES], mean_picps,
                  color=[WIN_COLORS[w] for w in WINDOW_SIZES], edgecolor="white")
bars[WINDOW_SIZES.index(best_win)].set_edgecolor("black")
bars[WINDOW_SIZES.index(best_win)].set_linewidth(2.5)
ax_sum.axhline(NOMINAL, color="black", ls="--", lw=1.3, label=f"Nominal {NOMINAL:.0%}")
ax_sum.set_ylim(0, 1.1); ax_sum.set_title("Mean PICP — Best Window Selection")
ax_sum.set_ylabel("Mean Coverage"); ax_sum.legend()
for bar, v in zip(bars, mean_picps):
    ax_sum.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
ax_sum.grid(alpha=0.3, axis="y")

plt.savefig("rnn_two_models.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved → rnn_two_models.png")