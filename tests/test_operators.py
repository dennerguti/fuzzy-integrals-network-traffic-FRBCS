"""Consistency tests: vectorised implementations == reference implementations.

Run with:  python -m pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fintraffic.forecasting import (AggregationConfig, aggregate_window,  # noqa: E402
                                    forecast)
from fintraffic.operators import (OPERATORS, choquet_integral, choquet_integral_batch,  # noqa: E402
                                  pma, pma_batch, power_measure)

RNG = np.random.default_rng(42)
# Bursty integer series with zeros, similar to packet counts.
SERIES = np.round(RNG.lognormal(6, 1.5, 3000) * (RNG.random(3000) > 0.02))
Q_VALUES = [0.1, 0.5, 1.0, 2.5, 3.0, 10.0]
CONFIGS = [AggregationConfig(), AggregationConfig(True, False, True),
           AggregationConfig(False, False, False)]


def test_power_measure_bounds():
    assert power_measure(5, 5, 3.0) == 1.0
    assert power_measure(0, 5, 3.0) == 0.0


@pytest.mark.parametrize("op", OPERATORS)
@pytest.mark.parametrize("q", Q_VALUES)
def test_batch_equals_reference(op, q):
    windows = RNG.random((200, 15))
    windows[:5] = 0.0  # constant windows
    expected = [choquet_integral(w, op, q) for w in windows]
    got = choquet_integral_batch(windows, op, q)
    if OPERATORS[op] == "dCC":
        # Python's x**2 (libm pow) and NumPy's x*x may differ in the last bit.
        np.testing.assert_allclose(got, expected, rtol=1e-14, atol=1e-15)
    else:
        np.testing.assert_array_equal(got, expected)


def test_pma_batch_equals_reference():
    windows = RNG.integers(0, 5000, (200, 30)).astype(float)
    np.testing.assert_array_equal(pma_batch(windows), [pma(w) for w in windows])


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("window", [5, 20])
def test_forecast_equals_reference(config, window):
    start, end = 70, 600
    for op in OPERATORS:
        for q in (0.5, 3.0):
            expected = [round(aggregate_window(SERIES[t - window:t].tolist(), op, q, config))
                        for t in range(start, end)]
            np.testing.assert_array_equal(
                forecast(SERIES, window, op, q, start, end, config), expected)


def test_choquet_with_uniform_measure_is_mean():
    x = RNG.random(10)
    assert choquet_integral(x, "Choquet", q=1.0) == pytest.approx(x.mean())


def test_normalized_outputs_in_unit_interval():
    windows = RNG.random((500, 20))
    windows /= windows.max(axis=1, keepdims=True)
    for op in OPERATORS:
        for q in Q_VALUES:
            out = choquet_integral_batch(windows, op, q)
            if op in (4, 5, 6, 7, 8):
                out = out / windows.shape[1]
            assert out.max() <= 1 + 1e-12, (OPERATORS[op], q)
