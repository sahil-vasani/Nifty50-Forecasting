import pandas as pd
import numpy as np
import yfinance as yf

# =========================
# CONFIG
# =========================
TICKER = "^NSEI"   # NIFTY50
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"

# =========================
# DOWNLOAD DATA
# =========================
def download_data():
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True)

    # Flatten columns
    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]

    df = df.reset_index()

    # ✅ FIX: standardize date column
    df.rename(columns={'Date': 'date'}, inplace=True)

    return df

# =========================
# FEATURE ENGINEERING
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

    # ---------- Candle Features ----------
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['close_pos'] = (df['close'] - df['low']) / (df['range'] + 1e-9)

    # ---------- Moving Averages ----------
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['dist_ma_5'] = (df['close'] - df['ma_5']) / df['ma_5']
    df['dist_ma_20'] = (df['close'] - df['ma_20']) / df['ma_20']

    # ---------- Volume ----------
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_spike'] = df['volume'] / (df['volume_ma_5'] + 1e-9)

    # ---------- Target (Next Day Return) ----------
    df['target'] = df['close'].pct_change().shift(-1)

    # Drop NaN
    df = df.dropna().reset_index(drop=True)

    return df

# =========================
# ADD SECTOR DATA (OPTIONAL)
# =========================
def add_sector_features(df):
    print("Adding sector features...")

    # -------- BANK NIFTY --------
    bank = yf.download("^NSEBANK", start=START_DATE, end=END_DATE, auto_adjust=True)

    # Flatten columns
    bank.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in bank.columns]

    bank['bank_ret'] = bank['close'].pct_change()
    bank = bank[['bank_ret']]

    # -------- IT INDEX --------
    it = yf.download("^CNXIT", start=START_DATE, end=END_DATE, auto_adjust=True)

    # Flatten columns
    it.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in it.columns]

    it['it_ret'] = it['close'].pct_change()
    it = it[['it_ret']]

    # -------- RESET INDEX FOR MERGE --------
    bank = bank.reset_index()
    it = it.reset_index()

    # Rename date column for consistency
    bank.rename(columns={'Date': 'date'}, inplace=True)
    it.rename(columns={'Date': 'date'}, inplace=True)

    # -------- MERGE --------
    df = df.merge(bank, on='date', how='left')
    df = df.merge(it, on='date', how='left')

    df = df.dropna().reset_index(drop=True)

    return df

# =========================
# FINAL PIPELINE
# =========================
def build_dataset():
    df = download_data()
    df = create_features(df)
    df = add_sector_features(df)

    # Select final features
    features = [
        'log_ret',
        'ret_1', 'ret_5', 'ret_15',
        'vol_5', 'vol_15',
        'body', 'range', 'upper_wick', 'lower_wick', 'close_pos',
        'dist_ma_5', 'dist_ma_20',
        'volume_spike',
        'bank_ret', 'it_ret'
    ]

    X = df[features]
    y = df['target']

    return X, y, df

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    X, y, df = build_dataset()

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    df.to_csv("nifty_features.csv", index=False)
    print("Dataset saved as nifty_features.csv")