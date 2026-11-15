"""One-step-ahead sliding-window forecasting and evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_absolute_error

from .operators import (CODOMAIN_0_N, OPERATORS, choquet_integral, choquet_integral_batch,
                        operator_code, pma, pma_batch)


@dataclass(frozen=True)
class AggregationConfig:
    """How each window is pre-processed before being aggregated.

    normalize
        Divide the window by its maximum (Eq. 15), aggregate, and multiply the
        result back by the maximum, so the MAE is in packets.
    rescale_by_n
        Divide by ``n`` the operators whose codomain is [0, n]
        (CF-nonAVG, CF1F2, dCF, dCC, dXC), bringing them back to [0, 1].
    cap_at_one
        Apply ``min{1, sum}`` to CF-AVG, CF-nonAVG and CF1F2.

    The default (True, True, False) is the configuration used in the paper.
    ``AggregationConfig(normalize=True, rescale_by_n=False, cap_at_one=True)``
    reproduces the capped variant.
    """

    normalize: bool = True
    rescale_by_n: bool = True
    cap_at_one: bool = False


DEFAULT_CONFIG = AggregationConfig()


# ---------------------------------------------------------------------------
# Aggregation of windows
# ---------------------------------------------------------------------------
def aggregate_window(window: Sequence[float], operator: int | str, q: float,
                     config: AggregationConfig = DEFAULT_CONFIG) -> float:
    """Forecast from a single window (reference, non-vectorised version)."""
    op = operator_code(operator)
    if not config.normalize:
        return choquet_integral(window, op, q, config.cap_at_one)
    peak = max(window)
    if peak == 0:
        return 0.0
    result = choquet_integral([v / peak for v in window], op, q, config.cap_at_one)
    if config.rescale_by_n and op in CODOMAIN_0_N:
        result = result / len(window)
    return result * peak


def aggregate_windows(windows: np.ndarray, operator: int | str, q: float,
                      config: AggregationConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Vectorised :func:`aggregate_window` (one forecast per row)."""
    op = operator_code(operator)
    windows = np.asarray(windows, dtype=float)
    if not config.normalize:
        return choquet_integral_batch(windows, op, q, config.cap_at_one)
    peak = windows.max(axis=1)
    zero = peak == 0
    safe_peak = np.where(zero, 1.0, peak)
    result = choquet_integral_batch(windows / safe_peak[:, None], op, q, config.cap_at_one)
    if config.rescale_by_n and op in CODOMAIN_0_N:
        result = result / windows.shape[1]
    return np.where(zero, 0.0, result * peak)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
def windows_for_targets(series: np.ndarray, window: int, start: int, end: int) -> np.ndarray:
    """Rows ``series[t - window : t]`` for every target index ``t`` in ``[start, end)``."""
    if start < window:
        raise ValueError(f"start ({start}) must be >= window ({window}).")
    return sliding_window_view(np.asarray(series, dtype=float), window)[start - window:end - window]


def forecast(series: np.ndarray, window: int, method: int | str, q: float | None = None,
             start: int | None = None, end: int | None = None,
             config: AggregationConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Rounded one-step-ahead forecasts for targets ``t`` in ``[start, end)``.

    ``method`` is ``"PMA"`` or a Choquet-based operator (name or code).
    Forecasts are rounded to the nearest integer (number of packets).
    """
    start = window if start is None else start
    end = len(series) if end is None else end
    W = windows_for_targets(series, window, start, end)
    if method == "PMA":
        preds = pma_batch(W)
    else:
        if q is None:
            raise ValueError("q is required for Choquet-based operators.")
        preds = aggregate_windows(W, method, q, config)
    return np.round(preds)  # round-half-to-even, same as Python's round()


def mae(series: np.ndarray, window: int, method: int | str, q: float | None,
        start: int, end: int, config: AggregationConfig = DEFAULT_CONFIG) -> float:
    """MAE of the rounded forecasts for targets in ``[start, end)``."""
    preds = forecast(series, window, method, q, start, end, config)
    return mean_absolute_error(np.asarray(series[start:end], dtype=float), preds)


# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Split:
    """Evaluation periods (indices of the targets).

    * ``full``       – ``[start, n)``: every window is evaluated on the same targets;
    * ``validation`` – first half of ``full``, used to select (window, q);
    * ``test``       – second half of ``full``, used to report the error.
    """

    start: int
    mid: int
    end: int

    @classmethod
    def from_series(cls, n: int, start: int) -> "Split":
        return cls(start=start, mid=start + (n - start) // 2, end=n)

    @property
    def full(self) -> tuple[int, int]:
        return self.start, self.end

    @property
    def validation(self) -> tuple[int, int]:
        return self.start, self.mid

    @property
    def test(self) -> tuple[int, int]:
        return self.mid, self.end


def method_label(operator: int | str) -> str:
    return "PMA" if operator == "PMA" else OPERATORS[operator_code(operator)]


def run_grid(series: np.ndarray, windows: Iterable[int], q_values: Iterable[float],
             start: int, end: int | None = None, operators: Iterable[int] = OPERATORS,
             config: AggregationConfig = DEFAULT_CONFIG, verbose: bool = True) -> pd.DataFrame:
    """MAE of PMA and of every (operator, q) for each window, on targets ``[start, end)``.

    Returns a long DataFrame with columns ``window, q, method, mae``
    (``q`` is NaN for PMA).
    """
    end = len(series) if end is None else end
    q_values = list(q_values)
    rows = []
    for w in windows:
        if verbose:
            print(f"  window {w}")
        rows.append({"window": w, "q": np.nan, "method": "PMA",
                     "mae": mae(series, w, "PMA", None, start, end, config)})
        for q in q_values:
            for op in operators:
                rows.append({"window": w, "q": q, "method": method_label(op),
                             "mae": mae(series, w, op, q, start, end, config)})
    return pd.DataFrame(rows)


def select_on_validation(series: np.ndarray, windows: Iterable[int], q_values: Iterable[float],
                         split: Split, operators: Iterable[int] = OPERATORS,
                         config: AggregationConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """For each operator, pick (window, q) with the lowest validation MAE and report test MAE."""
    windows, q_values = list(windows), list(q_values)
    rows = []
    for op in operators:
        scores = {(w, q): mae(series, w, op, q, *split.validation, config)
                  for w in windows for q in q_values}
        (w_best, q_best), val = min(scores.items(), key=lambda item: item[1])
        rows.append({"method": method_label(op), "window": w_best, "q": q_best,
                     "mae_validation": val,
                     "mae_test": mae(series, w_best, op, q_best, *split.test, config)})
    return pd.DataFrame(rows).sort_values("mae_test").reset_index(drop=True)


def q_curves(series: np.ndarray, operator: int | str, windows: Iterable[int],
             q_values: Sequence[float], split: Split,
             config: AggregationConfig = DEFAULT_CONFIG) -> dict[int, list[float]]:
    """Validation MAE as a function of q, for each window."""
    return {w: [mae(series, w, operator, q, *split.validation, config) for q in q_values]
            for w in windows}


def q_curve_summary(series: np.ndarray, operator: int | str, curves: dict[int, list[float]],
                    q_values: Sequence[float], split: Split, tolerance: float = 0.02,
                    config: AggregationConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Best q per window (validation), gain over q = 1, tolerance band and test MAE."""
    q_values = list(q_values)
    rows = []
    for w, maes in curves.items():
        maes = np.asarray(maes)
        q_best = q_values[int(maes.argmin())]
        mae_q1 = maes[q_values.index(1)]
        band = [q for q, m in zip(q_values, maes) if m <= maes.min() * (1 + tolerance)]
        rows.append({
            "window": w,
            "q_best": q_best,
            "mae_val_best": maes.min(),
            "mae_val_q1": mae_q1,
            "gain_vs_q1_%": 100 * (mae_q1 - maes.min()) / mae_q1,
            f"q_band_{tolerance:.0%}": f"[{min(band):g}, {max(band):g}]",
            "mae_test_best": mae(series, w, operator, q_best, *split.test, config),
            "mae_test_q1": mae(series, w, operator, 1, *split.test, config),
            "best_on_grid_edge": q_best in (q_values[0], q_values[-1]),
        })
    return pd.DataFrame(rows)


def compare_with_pma(series: np.ndarray, operator: int | str, best_q: dict[int, float],
                     split: Split, fixed_q: float = 3.0,
                     config: AggregationConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Test MAE of PMA vs. ``operator`` with the selected q and with a fixed q."""
    rows = []
    for w, q in best_q.items():
        pma_mae = mae(series, w, "PMA", None, *split.test, config)
        best = mae(series, w, operator, q, *split.test, config)
        fixed = mae(series, w, operator, fixed_q, *split.test, config)
        rows.append({
            "window": w, "q_best": q, "PMA_test": pma_mae,
            "op_qbest_test": best, "reduction_qbest_vs_PMA_%": 100 * (pma_mae - best) / pma_mae,
            f"op_q{fixed_q:g}_test": fixed,
            f"reduction_q{fixed_q:g}_vs_PMA_%": 100 * (pma_mae - fixed) / pma_mae,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sanity checks on grid results
# ---------------------------------------------------------------------------
def best_q_table(results: pd.DataFrame) -> pd.DataFrame:
    """Best q (lowest MAE) for each window (rows) and operator (columns)."""
    df = results[results["method"] != "PMA"]
    best = df.loc[df.groupby(["method", "window"])["mae"].idxmin()]
    return best.pivot(index="window", columns="method", values="q")[
        [m for m in OPERATORS.values() if m in set(best["method"])]]


def duplicated_tables(results: pd.DataFrame, tol: float = 0.5) -> list[tuple[str, str]]:
    """Pairs of operators whose (window x q) MAE tables are virtually identical."""
    tables = _tables(results)
    names = list(tables)
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
            if np.abs(tables[a].values - tables[b].values).max() < tol]


def q_independent(results: pd.DataFrame, tol: float = 0.5) -> list[str]:
    """Operators whose MAE does not change with q (suspicious)."""
    return [m for m, t in _tables(results).items() if np.ptp(t.values, axis=1).max() < tol]


def _tables(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = results[results["method"] != "PMA"]
    return {m: df[df["method"] == m].pivot(index="window", columns="q", values="mae")
            for m in df["method"].unique()}


def dxc_dcf_identity(results: pd.DataFrame, pairs=((2.0, 0.5), (4.0, 1.0))) -> pd.DataFrame:
    """Check the identity d-XC(q) == dCF(q/4) on the grid results."""
    rows = []
    for q_dxc, q_dcf in pairs:
        a = results[(results.method == "dXC") & np.isclose(results.q, q_dxc)].set_index("window")["mae"]
        b = results[(results.method == "dCF") & np.isclose(results.q, q_dcf)].set_index("window")["mae"]
        if len(a) and len(b):
            rows.append({"q_dXC": q_dxc, "q_dCF": q_dcf, "max_abs_diff": float(np.abs(a - b).max())})
    return pd.DataFrame(rows)


# Re-exported for convenience (reference implementations, one window at a time).
__all__ = [
    "AggregationConfig", "DEFAULT_CONFIG", "Split", "aggregate_window", "aggregate_windows",
    "best_q_table", "compare_with_pma", "duplicated_tables", "dxc_dcf_identity", "forecast",
    "mae", "method_label", "pma", "q_curve_summary", "q_curves", "q_independent", "run_grid",
    "select_on_validation", "windows_for_targets",
]
