import numpy as np
from var_model import compute_var_es, simulate_batch


def test_compute_var_es_simple():
    pnl = np.array([-10, -5, 0, 5, 10])

    var, es = compute_var_es(pnl, 0.8)

    # 20th percentile ≈ -5
    assert np.isclose(var, -5)
    assert np.isclose(es, (-10 - 5) / 2)


def test_simulate_batch_shape():
    fake_params = np.array([0.0, 0.1, 0.8, 5.0])  # not exact, just placeholder
    pnl = simulate_batch(100, fake_params, seed=42)

    assert len(pnl) == 100


def test_var_monotonicity():
    pnl = np.random.normal(0, 1, 10000)

    var_95, _ = compute_var_es(pnl, 0.95)
    var_99, _ = compute_var_es(pnl, 0.99)

    # 99% VaR should be worse (more negative)
    assert var_99 <= var_95