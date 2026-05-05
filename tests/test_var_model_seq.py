import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import var_sequential  # replace with actual filename (without .py)


# -------------------------
# load_returns
# -------------------------
def test_load_returns():
    fake_df = pd.DataFrame({"Log Return": [0.01, -0.02, 0.03]})

    with patch("var_sequential.pd.read_csv", return_value=fake_df):
        returns = var_sequential.load_returns()

    expected = np.array([1.0, -2.0, 3.0])
    assert np.allclose(returns, expected)


# -------------------------
# fit_garch_model
# -------------------------
def test_fit_garch_model():
    mock_model = MagicMock()
    mock_result = MagicMock()

    with patch("var_sequential.arch_model", return_value=mock_model):
        mock_model.fit.return_value = mock_result

        res = var_sequential.fit_garch_model(np.array([1, 2, 3]))

    mock_model.fit.assert_called_once_with(disp="off")
    assert res == mock_result


# -------------------------
# simulate_garch_t
# -------------------------
def test_simulate_garch_t():
    # fake structure returned by arch forecast
    fake_simulations = np.zeros((1, 5, 1))
    fake_simulations[-1, :, 0] = np.array([1, 2, 3, 4, 5])

    fake_forecast = MagicMock()
    fake_forecast.simulations.values = fake_simulations

    mock_res = MagicMock()
    mock_res.forecast.return_value = fake_forecast

    out = var_sequential.simulate_garch_t(mock_res, n_simulations=5)

    expected = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    assert np.allclose(out, expected)


# -------------------------
# compute_var_es
# -------------------------
def test_compute_var_es():
    pnl = np.array([-10, -5, 0, 5, 10])

    var, es = var_sequential.compute_var_es(pnl, 0.8)

    # 20th percentile = -6 (numpy interpolation)
    assert np.isclose(var, -6)

    # only -10 is below threshold
    assert np.isclose(es, -10)


# -------------------------
# run_var_simulation
# -------------------------
def test_run_var_simulation():
    fake_returns = np.array([1, 2, 3])

    fake_fit = MagicMock()
    fake_simulated = np.array([0.01, 0.02, 0.03])

    with patch("var_sequential.load_returns", return_value=fake_returns), \
         patch("var_sequential.fit_garch_model", return_value=fake_fit), \
         patch("var_sequential.simulate_garch_t", return_value=fake_simulated):

        pnl = var_sequential.run_var_simulation()

    expected = var_sequential.INITIAL_PORTFOLIO * fake_simulated

    assert np.allclose(pnl, expected)


# -------------------------
# measure_runtime
# -------------------------
def test_measure_runtime():
    fake_returns = np.array([1, 2, 3])

    mock_fit = MagicMock()

    with patch("var_sequential.load_returns", return_value=fake_returns), \
         patch("var_sequential.fit_garch_model", return_value=mock_fit), \
         patch("var_sequential.simulate_garch_t"), \
         patch("var_sequential.time.perf_counter", side_effect=[1.0, 2.0]):

        runtime = var_sequential.measure_runtime()

    assert runtime == 1.0