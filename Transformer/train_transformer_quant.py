import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from torch.utils.data import Dataset, DataLoader
import pickle

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv("nifty_features.csv")
target = 'target'

# =========================================================
# FEATURE SELECTION (from uncertainty paper)
# reduces noise, improves interval quality
# =========================================================
features = [
    'log_ret','ret_1','ret_5','ret_15',
    'vol_5','vol_15','body','range',
    'upper_wick','lower_wick','close_pos',
    'dist_ma_5','dist_ma_20','volume_spike',
    'bank_ret','it_ret'
]

X = df[features]
y = df[target]

mi = mutual_info_regression(X, y)
selected_features = pd.Series(mi, index=features)\
                    .sort_values(ascending=False)\
                    .head(10).index.tolist()

print("Selected Features:", selected_features)

X = df[selected_features].values
y = y.values

# =========================================================
# SCALING
# =========================================================
scaler = StandardScaler()
X = scaler.fit_transform(X)

pickle.dump(scaler, open("scaler.pkl","wb"))
pickle.dump(selected_features, open("features.pkl","wb"))

# =========================================================
# SEQUENCE CREATION
# =========================================================
SEQ_LEN = 20

def create_seq(X,y):
    Xs, ys = [], []
    for i in range(len(X)-SEQ_LEN):
        Xs.append(X[i:i+SEQ_LEN])
        ys.append(y[i+SEQ_LEN])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_seq(X,y)

# =========================================================
# TIME SPLIT (QUANT STYLE)
# =========================================================
total = len(X_seq)
train_size = int(0.7*total)
calib_size = int(0.15*total)

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]

X_calib = X_seq[train_size:train_size+calib_size]
y_calib = y_seq[train_size:train_size+calib_size]

X_test = X_seq[train_size+calib_size:]
y_test = y_seq[train_size+calib_size:]

# =========================================================
# DATASET
# =========================================================
class TSData(Dataset):
    def __init__(self,X,y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TSData(X_train,y_train), batch_size=32, shuffle=True)

# =========================================================
# TRANSFORMER MODEL (from scratch)
# =========================================================
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(d_model, 2)

    def forward(self,x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:,-1,:]

        out = self.fc(x)

        lower = out[:,0]
        upper = lower + torch.abs(out[:,1])  # valid interval

        return lower, upper

model = TransformerModel(len(selected_features))

# =========================================================
# TUBE LOSS (paper implementation)
# =========================================================
def tube_loss(y, l, u, t=0.9, r=0.5, delta=0.01):
    loss = 0
    for i in range(len(y)):
        yi = y[i]

        if yi > u[i]:
            loss += t*(yi-u[i])
        elif yi < l[i]:
            loss += t*(l[i]-yi)
        else:
            mid = r*u[i] + (1-r)*l[i]
            if yi >= mid:
                loss += (1-t)*(u[i]-yi)
            else:
                loss += (1-t)*(yi-l[i])

        loss += delta*torch.abs(u[i]-l[i])

    return loss/len(y)

# =========================================================
# TRAIN
# =========================================================
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    model.train()
    total_loss = 0

    for xb,yb in train_loader:
        optimizer.zero_grad()

        l,u = model(xb)
        loss = tube_loss(yb,l,u)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =========================================================
# CONFORMAL CALIBRATION (paper 2)
# =========================================================
model.eval()
errors = []

with torch.no_grad():
    for i in range(len(X_calib)):
        x = torch.tensor(X_calib[i], dtype=torch.float32).unsqueeze(0)
        y_true = y_calib[i]

        l,u = model(x)
        l,u = l.item(), u.item()

        err = max(l - y_true, y_true - u)
        errors.append(err)

# save errors (for dynamic confidence)
pickle.dump(errors, open("calibration_errors.pkl","wb"))

# =========================================================
# SAVE MODEL
# =========================================================
torch.save(model.state_dict(), "transformer_model.pth")

print("Training Complete ✅")