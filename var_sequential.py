import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

returns = pd.read_csv("data/aapl_returns.csv")["Log Return"].values

NUM_SIMULATIONS = 10_000
INITIAL_PORTFOLIO = 1_000_000  # $1M example



start = time.perf_counter()

simulated_pnl = []

for _ in range(NUM_SIMULATIONS):
    sampled_return = np.random.choice(returns)
    pnl = INITIAL_PORTFOLIO * sampled_return
    simulated_pnl.append(pnl)

simulated_pnl = np.array(simulated_pnl)

var_95 = np.percentile(simulated_pnl, 5)
var_99 = np.percentile(simulated_pnl, 1)

end = time.perf_counter()
runtime = end - start


print(f"95% VaR: ${var_95:,.2f}")
print(f"99% VaR: ${var_99:,.2f}")
print(f"Runtime: {runtime:.4f} seconds")


plt.hist(simulated_pnl, bins=50)
plt.axvline(var_95, color='orange', label='95% VaR')
plt.axvline(var_99, color='red', label='99% VaR')

plt.title("Monte Carlo VaR Simulation")
plt.xlabel("Profit / Loss")
plt.ylabel("Frequency")
plt.legend()

plt.savefig("histogram.png")
plt.close()
