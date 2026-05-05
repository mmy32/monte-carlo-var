import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import var_parallel


# ---------------------------
# load_returns
# ---------------------------
def test_load_returns():
    fake_df = pd.DataFrame({"Log Return": [0.01, 0.02, -0.01]})

    with patch("var_parallel.pd.read_csv", return_value=fake_df):
        returns = var_parallel.load_returns()

    # should scale by 100
    assert np.allclose(returns, np.array([1.0, 2.0, -1.0]))


# ---------------------------
# fit_garch_model
# ---------------------------
def test_fit_garch_model():
    mock_model = MagicMock()
    mock_res = MagicMock()

    with patch("var_parallel.arch_model", return_value=mock_model):
        mock_model.fit.return_value = mock_res

        res = var_parallel.fit_garch_model(np.array([1, 2, 3]))

    mock_model.fit.assert_called_once_with(disp="off")
    assert res == mock_res


# ---------------------------
# simulate_batch
# ---------------------------
def test_simulate_batch():
    fake_data = pd.DataFrame({"data": np.array([1.0, 2.0, 3.0])})

    with patch("var_parallel.arch_model") as mock_arch:
        mock_instance = mock_arch.return_value
        mock_instance.simulate.return_value = fake_data

        pnl = var_parallel.simulate_batch(
            n_sims=3,
            model_params=np.array([1, 2, 3]),
            seed=42
        )

    # scaled back from % and multiplied by portfolio
    expected = var_parallel.INITIAL_PORTFOLIO * (np.array([1, 2, 3]) / 100)

    assert np.allclose(pnl, expected)


# ---------------------------
# compute_var_es
# ---------------------------
def test_compute_var_es():
    pnl = np.array([-10, -5, 0, 5, 10])

    var, es = var_parallel.compute_var_es(pnl, 0.8)

    # numpy interpolation → -6
    assert np.isclose(var, -6)

    # only -10 is below -6
    assert np.isclose(es, -10)


# ---------------------------
# run_parallel_simulation
# ---------------------------
def test_run_parallel_simulation():
    fake_result = np.array([1, 2, 3])

    mock_future = MagicMock()
    mock_future.result.return_value = fake_result

    mock_executor = MagicMock()
    mock_executor.__enter__.return_value.submit.return_value = mock_future

    with patch("var_parallel.ProcessPoolExecutor", return_value=mock_executor):
        with patch("var_parallel.os.cpu_count", return_value=2):
            result = var_parallel.run_parallel_simulation(
                num_simulations=6,
                model_params=np.array([1, 2])
            )

    # 2 workers → 2 batches → concatenated
    assert len(result) == 6


# ---------------------------
# run_var_simulation
# ---------------------------
def test_run_var_simulation():
    fake_returns = np.array([1, 2, 3])
    fake_params = np.array([0.1, 0.2])
    fake_pnl = np.array([100, 200, 300])

    mock_res = MagicMock()
    mock_res.params = fake_params

    with patch("var_parallel.load_returns", return_value=fake_returns), \
         patch("var_parallel.fit_garch_model", return_value=mock_res), \
         patch("var_parallel.run_parallel_simulation", return_value=fake_pnl):

        pnl = var_parallel.run_var_simulation()

    assert np.array_equal(pnl, fake_pnl)


# ---------------------------
# measure_runtime
# ---------------------------
def test_measure_runtime():
    fake_returns = np.array([1, 2, 3])
    fake_params = np.array([0.1, 0.2])

    mock_res = MagicMock()
    mock_res.params = fake_params

    with patch("var_parallel.load_returns", return_value=fake_returns), \
         patch("var_parallel.fit_garch_model", return_value=mock_res), \
         patch("var_parallel.run_parallel_simulation"), \
         patch("var_parallel.time.perf_counter", side_effect=[1.0, 2.5]):

        runtime = var_parallel.measure_runtime()

    assert runtime == 1.5