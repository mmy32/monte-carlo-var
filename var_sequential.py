import numpy as np
import pandas as pd
import time
from arch import arch_model

NUM_SIMULATIONS = 10_000
INITIAL_PORTFOLIO = 1_000_000


def load_returns():
    returns = pd.read_csv("data/aapl_returns.csv")["Log Return"].values
    return returns * 100  # arch expects % returns


def fit_garch_model(returns):
    model = arch_model(
        returns,
        vol="Garch",
        p=1,
        q=1,
        dist="t"  # Student-t distribution
    )
    res = model.fit(disp="off")
    return res


def simulate_garch_t(res, n_simulations):
    sim_data = res.forecast(horizon=1, method="simulation", simulations=n_simulations)

    # Extract simulated returns
    simulated_returns = sim_data.simulations.values[-1, :, 0]

    return simulated_returns / 100  # convert back to decimal


def compute_var_es(pnl, alpha):
    var_threshold = np.percentile(pnl, (1 - alpha) * 100)
    tail_losses = pnl[pnl <= var_threshold]
    es = tail_losses.mean()

    return var_threshold, es


def run_var_simulation():
    returns = load_returns()

    # Fit model
    res = fit_garch_model(returns)

    # Simulate returns
    simulated_returns = simulate_garch_t(res, NUM_SIMULATIONS)

    pnl = INITIAL_PORTFOLIO * simulated_returns
    return pnl


def measure_runtime():
    returns = load_returns()

    start = time.perf_counter()

    res = fit_garch_model(returns)
    simulate_garch_t(res, NUM_SIMULATIONS)

    end = time.perf_counter()

    return end - start


# Optional CLI run
if __name__ == "__main__":
    pnl = run_var_simulation()

    var_95, es_95 = compute_var_es(pnl, 0.95)
    var_99, es_99 = compute_var_es(pnl, 0.99)

    print(f"95% VaR: ${var_95:,.2f}")
    print(f"95% ES : ${es_95:,.2f}")
    print(f"99% VaR: ${var_99:,.2f}")
    print(f"99% ES : ${es_99:,.2f}")