import numpy as np
import pandas as pd
from arch import arch_model
import time

# --- Load data ---
returns = pd.read_csv("data/aapl_returns.csv")["Log Return"].values

INITIAL_PORTFOLIO = 1_000_000
NUM_SIMULATIONS = 100_000

# --- Fit GARCH(1,1) with Student-t innovations ---
model = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
res = model.fit(disp="off")


start = time.perf_counter()

# --- Simulate 1-step ahead returns ---
sim = res.forecast(horizon=1, method='simulation', simulations=NUM_SIMULATIONS)

# shape: (time, simulations, horizon)
sim_returns = sim.simulations.values[-1, :, 0]

# --- Convert to PnL ---
pnl = INITIAL_PORTFOLIO * sim_returns

# --- VaR ---
var_95 = np.percentile(pnl, 5)
var_99 = np.percentile(pnl, 1)

# --- Expected Shortfall ---
es_95 = pnl[pnl <= var_95].mean()
es_99 = pnl[pnl <= var_99].mean()
end = time.perf_counter

runtime = end - start

print(f"Runtime: {runtime:.6f} seconds")

# --- Output ---
print(f"95% VaR: ${var_95:,.2f}")
print(f"99% VaR: ${var_99:,.2f}")
print(f"95% ES:  ${es_95:,.2f}")
print(f"99% ES:  ${es_99:,.2f}")