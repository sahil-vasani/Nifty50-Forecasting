import pandas as pd
import numpy as np
from scipy.optimize import minimize
import pickle

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../nifty_final_dataset.csv")
df = df.dropna().reset_index(drop=True)

# =========================
# RETURNS (in %)
# =========================
returns = df["target"].values * 100

# =========================
# GARCH(1,1) LOG LIKELIHOOD
# =========================
def garch_loglik(params, returns):
    omega, alpha, beta = params
    
    T = len(returns)
    var = np.zeros(T)
    
    # initialize variance
    var[0] = np.var(returns)
    
    for t in range(1, T):
        var[t] = omega + alpha * returns[t-1]**2 + beta * var[t-1]
    
    # avoid zero variance
    var = np.clip(var, 1e-6, None)
    
    loglik = -0.5 * np.sum(
        np.log(2 * np.pi) +
        np.log(var) +
        (returns**2) / var
    )
    
    return -loglik  # minimize

# =========================
# INITIAL GUESS
# =========================
init_params = [0.1, 0.1, 0.8]

bounds = [
    (1e-6, 1),   # omega
    (0, 1),      # alpha
    (0, 1)       # beta
]

# =========================
# FIT MODEL
# =========================
result = minimize(
    garch_loglik,
    init_params,
    args=(returns,),
    bounds=bounds
)

omega, alpha, beta = result.x

print("\n===== TRAINED PARAMETERS =====")
print(f"Omega : {omega:.6f}")
print(f"Alpha : {alpha:.6f}")
print(f"Beta  : {beta:.6f}")

# =========================
# SAVE MODEL
# =========================
model = {
    "omega": omega,
    "alpha": alpha,
    "beta": beta
}

with open("garch_scratch.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully!")