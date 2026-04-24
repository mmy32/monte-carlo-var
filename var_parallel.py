import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os


NUM_SIMULATIONS = 10_000
INITIAL_PORTFOLIO = 1_000_000


# --- Worker function ---
def simulate_batch(n_sims, returns, initial_portfolio, seed):
    rng = np.random.default_rng(seed)
    sampled_returns = rng.choice(returns, size=n_sims)
    pnl = initial_portfolio * sampled_returns
    return pnl


# --- Parallel execution ---
def run_parallel_simulation(num_simulations, returns, initial_portfolio):
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
                    returns,
                    initial_portfolio,
                    i
                )
            )

        for future in futures:
            results.append(future.result())

    return np.concatenate(results)


# --- Risk metrics ---
def compute_var_es(pnl, alpha):
    """
    alpha = 0.95 -> 95% VaR / ES
    """
    var_threshold = np.percentile(pnl, (1 - alpha) * 100)

    # Tail losses (<= VaR threshold)
    tail_losses = pnl[pnl <= var_threshold]

    es = tail_losses.mean() if len(tail_losses) > 0 else var_threshold

    return var_threshold, es


def main():
    returns = pd.read_csv("data/aapl_returns.csv")["Log Return"].values

    start = time.perf_counter()

    simulated_pnl = run_parallel_simulation(
        NUM_SIMULATIONS,
        returns,
        INITIAL_PORTFOLIO
    )

    # --- VaR & ES ---
    var_95, es_95 = compute_var_es(simulated_pnl, 0.95)
    var_99, es_99 = compute_var_es(simulated_pnl, 0.99)

    end = time.perf_counter()
    runtime = end - start

    print(f"95% VaR: ${var_95:,.2f}")
    print(f"95% ES : ${es_95:,.2f}")
    print(f"99% VaR: ${var_99:,.2f}")
    print(f"99% ES : ${es_99:,.2f}")
    print(f"Runtime: {runtime:.4f} seconds")

    # --- Plot ---
    plt.hist(simulated_pnl, bins=50)

    plt.axvline(var_95, color='orange', linestyle='--', label='95% VaR')
    plt.axvline(es_95, color='orange', linestyle='-', label='95% ES')

    plt.axvline(var_99, color='red', linestyle='--', label='99% VaR')
    plt.axvline(es_99, color='red', linestyle='-', label='99% ES')

    plt.title("Monte Carlo VaR & ES Simulation")
    plt.xlabel("Profit / Loss")
    plt.ylabel("Frequency")
    plt.legend()

    plt.savefig("histogram.png")
    plt.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()