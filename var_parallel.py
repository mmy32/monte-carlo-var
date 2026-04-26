import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor
import os
import multiprocessing as mp
from arch import arch_model

NUM_SIMULATIONS = 10_000
INITIAL_PORTFOLIO = 1_000_000


def load_returns():
    returns = pd.read_csv("data/aapl_returns.csv")["Log Return"].values
    return returns * 100  # arch expects % returns


# --- Fit model ONCE ---
def fit_garch_model(returns):
    model = arch_model(
        returns,
        vol="Garch",
        p=1,
        q=1,
        dist="t"
    )
    res = model.fit(disp="off")
    return res


# --- Worker: simulate from fitted model ---
def simulate_batch(n_sims, model_params, seed):
    np.random.seed(seed)

    # Recreate model with same params (lightweight vs refitting)
    am = arch_model(None, vol="Garch", p=1, q=1, dist="t")
    sim_data = am.simulate(model_params, n_sims)

    simulated_returns = sim_data["data"].values / 100
    pnl = INITIAL_PORTFOLIO * simulated_returns

    return pnl


def run_parallel_simulation(num_simulations, model_params):
    num_workers = os.cpu_count() or 4
    batch_size = num_simulations // num_workers

    futures = []
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            n_sims = (
                batch_size
                if i < num_workers - 1
                else num_simulations - batch_size * (num_workers - 1)
            )

            futures.append(
                executor.submit(
                    simulate_batch,
                    n_sims,
                    model_params,
                    i
                )
            )

        for future in futures:
            results.append(future.result())

    return np.concatenate(results)


def compute_var_es(pnl, alpha):
    var_threshold = np.percentile(pnl, (1 - alpha) * 100)
    tail_losses = pnl[pnl <= var_threshold]
    es = tail_losses.mean()

    return var_threshold, es


# ✅ Flask entry point
def run_var_simulation():
    returns = load_returns()

    # Fit once
    res = fit_garch_model(returns)

    # Extract parameters (important!)
    model_params = res.params

    pnl = run_parallel_simulation(
        NUM_SIMULATIONS,
        model_params
    )

    return pnl


def measure_runtime():
    returns = load_returns()

    start = time.perf_counter()

    res = fit_garch_model(returns)
    model_params = res.params

    run_parallel_simulation(
        NUM_SIMULATIONS,
        model_params
    )

    end = time.perf_counter()

    return end - start


# Optional CLI
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    pnl = run_var_simulation()

    var_95, es_95 = compute_var_es(pnl, 0.95)
    var_99, es_99 = compute_var_es(pnl, 0.99)

    print(f"95% VaR: ${var_95:,.2f}")
    print(f"95% ES : ${es_95:,.2f}")
    print(f"99% VaR: ${var_99:,.2f}")
    print(f"99% ES : ${es_99:,.2f}")