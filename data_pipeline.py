import pandas as pd
import numpy as np
import yfinance as yf

# =========================
# CONFIG
# =========================
START = "2018-01-01"
END = "2026-03-31"
TICKER = "^NSEI"

SECTOR_TICKERS = {
    "bank": "^NSEBANK",
    "it": "^CNXIT",
    "pharma": "^CNXPHARMA",
    "auto": "^CNXAUTO",
    "fmcg": "^CNXFMCG",
    "metal": "^CNXMETAL",
    "energy": "^CNXENERGY"
}

# =========================
# DOWNLOAD NIFTY
# =========================
def download_data():
    df = yf.download(TICKER, start=START, end=END, auto_adjust=True)

    # flatten columns
    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]

    df = df.reset_index()
    df.rename(columns={'Date': 'date'}, inplace=True)

    return df


# =========================
# FEATURE ENGINEERING (TECHNICAL)
# =========================
def create_features(df):

    # ---------- Returns ----------
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_15'] = df['close'].pct_change(15)

    # ---------- Volatility ----------
    df['vol_5'] = df['log_ret'].rolling(5).std()
    df['vol_15'] = df['log_ret'].rolling(15).std()

    # ---------- Candle ----------
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['close_pos'] = (df['close'] - df['low']) / (df['range'] + 1e-9)

    # ---------- Moving Avg ----------
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['dist_ma_5'] = (df['close'] - df['ma_5']) / df['ma_5']
    df['dist_ma_20'] = (df['close'] - df['ma_20']) / df['ma_20']

    # ---------- Volume ----------
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_spike'] = df['volume'] / (df['volume_ma_5'] + 1e-9)

    # ---------- Target ----------
    df['target'] = df['close'].pct_change().shift(-1)

    return df


# =========================
# ADD SECTOR FEATURES
# =========================
def add_sector_features(df):

    for name, ticker in SECTOR_TICKERS.items():
        print(f"Downloading {name}...")

        temp = yf.download(ticker, start=START, end=END, auto_adjust=True)

        temp.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in temp.columns]

        temp = temp.reset_index()
        temp.rename(columns={'Date': 'date'}, inplace=True)

        temp[f"{name}_ret"] = temp['close'].pct_change()

        df = df.merge(temp[['date', f"{name}_ret"]], on='date', how='left')

    return df


# =========================
# CONTAGION FEATURES 🔥
# =========================
def add_contagion_features(df):

    # lag features
    for col in df.columns:
        if "_ret" in col and col != "ret_1":
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag2"] = df[col].shift(2)

    # rolling correlation
    for col in df.columns:
        if "_ret" in col and col != "ret_1":
            df[f"corr_nifty_{col}"] = df["ret_1"].rolling(10).corr(df[col])

    return df

def advanced_features(df):
    
    # =========================
    # 1. TREND FEATURES 🔥
    # =========================
    df["trend_strength"] = df["ma_5"] - df["ma_20"]

    # =========================
    # 2. MOMENTUM 🔥
    # =========================
    df["momentum"] = df["close"] - df["close"].shift(5)

    # =========================
    # 3. RSI (VERY IMPORTANT) 🔥
    # =========================
    delta = df["close"].diff()

    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # =========================
    # 4. MARKET REGIME 🔥
    # =========================
    df["high_vol"] = (df["vol_5"] > df["vol_15"]).astype(int)

    # =========================
    # 5. SECTOR DOMINANCE 🔥 (VERY IMPORTANT)
    # =========================
    sector_cols = [
        "bank_ret","it_ret","pharma_ret",
        "auto_ret","fmcg_ret","metal_ret","energy_ret"
    ]

    df["sector_mean"] = df[sector_cols].mean(axis=1)
    df["sector_std"] = df[sector_cols].std(axis=1)

    # strongest sector each day
    df["strongest_sector"] = df[sector_cols].idxmax(axis=1)

    # =========================
    # 6. RELATIVE STRENGTH 🔥
    # =========================
    for col in sector_cols:
        df[f"{col}_vs_nifty"] = df[col] - df["ret_1"]

    # =========================
    # CLEAN
    # =========================
    df = df.dropna().reset_index(drop=True)

    return df


# =========================
# FINAL PIPELINE
# =========================
def build_dataset():

    df = download_data()

    df = create_features(df)

    df = add_sector_features(df)

    df = add_contagion_features(df)
    
    df = advanced_features(df)

    # clean
    df = df.dropna().reset_index(drop=True)
    
    return df


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    df = build_dataset()

    print("Final Shape:", df.shape)

    df.to_csv("nifty_final_dataset.csv", index=False)

    print("Dataset saved ✅")