"""Poisson Moving Average (PMA) and Choquet-based fuzzy integrals with the Power Measure.

Two implementations of every operator are provided:

* ``choquet_integral`` / ``pma`` – scalar, readable reference versions that follow
  the definitions in the paper term by term (one window at a time);
* ``choquet_integral_batch`` / ``pma_batch`` – NumPy versions that evaluate many
  windows at once. They perform exactly the same floating-point operations in the
  same order, so the results are identical to the reference versions
  (see ``tests/test_operators.py``), but they are orders of magnitude faster.
"""

from __future__ import annotations

from math import exp, factorial, sqrt
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Operator catalogue
# ---------------------------------------------------------------------------
#: Integer code -> short name of each Choquet-based operator.
OPERATORS: dict[int, str] = {
    0: "Choquet",    # standard Choquet integral
    1: "CC-min",     # C_C-integral with the minimum
    2: "CT",         # C_T-integral
    3: "CF-AVG",     # C_F-integral (averaging)
    4: "CF-nonAVG",  # C_F-integral (non-averaging)
    5: "CF1F2",      # C_{F1F2}-integral
    6: "dCF",        # d-C_F-integral
    7: "dCC",        # d-C_C-integral
    8: "dXC",        # d-X_C-integral
}

#: Name -> integer code.
OPERATOR_CODES: dict[str, int] = {name: code for code, name in OPERATORS.items()}

#: Operators whose first term is the raw smallest value x_(1) (Definitions 6, 8 and 9).
FIRST_TERM_RAW = (5, 6, 7, 8)

#: Operators that the original definitions cap with min{1, sum} (Definitions 5 and 6).
CAPPED_AT_ONE = (3, 4, 5)

#: Operators whose codomain is [0, n] (instead of [0, 1]) for inputs in [0, 1].
CODOMAIN_0_N = (4, 5, 6, 7, 8)


def operator_code(operator: int | str) -> int:
    """Accept an operator code (0-8) or its name and return the code."""
    if isinstance(operator, str):
        try:
            return OPERATOR_CODES[operator]
        except KeyError as exc:
            raise ValueError(f"Unknown operator {operator!r}. Options: {list(OPERATOR_CODES)}") from exc
    if operator not in OPERATORS:
        raise ValueError(f"Operator code must be between 0 and 8, got {operator}.")
    return int(operator)


def power_measure(cardinality: int, n: int, q: float) -> float:
    """Power Measure m(A) = (|A| / n) ** q. With q = 1 it is the uniform measure."""
    return (cardinality / n) ** q


# ---------------------------------------------------------------------------
# PMA baseline
# ---------------------------------------------------------------------------
def pma_weights(window: int) -> list[float]:
    """Weights of the Poisson Moving Average for a window of ``window`` values."""
    return [2 * (window**k * exp(-window) / factorial(k)) for k in range(1, window + 1)]


def pma(values: Sequence[float]) -> float:
    """PMA forecast for one window (values ordered from oldest to newest)."""
    weights = pma_weights(len(values))
    return sum(v * w for v, w in zip(values, weights))


def pma_batch(windows: np.ndarray) -> np.ndarray:
    """PMA forecast for each row of ``windows`` (shape: n_windows x window)."""
    windows = np.asarray(windows, dtype=float)
    weights = pma_weights(windows.shape[1])
    total = np.zeros(windows.shape[0])
    for k, w in enumerate(weights):  # same summation order as ``pma``
        total = total + windows[:, k] * w
    return total


# ---------------------------------------------------------------------------
# Choquet-based integrals (one window)
# ---------------------------------------------------------------------------
def _term(op: int, cur: float, prev: float, m: float) -> float:
    """Contribution of the i-th sorted element to the integral."""
    diff = cur - prev
    if op == 0:  # Choquet
        return diff * m
    if op == 1:  # CC-min
        return min(cur, m) - min(prev, m)
    if op == 2:  # CT
        den = diff + m - diff * m
        return 0.0 if den == 0 else (diff * m) / den
    if op == 3:  # CF-AVG
        return diff if diff < m else min(diff / 2, m)
    if op == 4:  # CF-nonAVG
        if diff == 0:
            return 0.0
        return (diff + m) / 2 if diff <= m else min(diff / 2, m)
    if op == 5:  # CF1F2
        return sqrt(cur * m) - prev * m**2
    if op == 6:  # dCF
        return sqrt(max(0.0, sqrt(cur) - sqrt(prev))) * m
    if op == 7:  # dCC
        part_cur = min(cur, m) * max(cur**2, m**2)
        part_prev = min(prev, m) * max(prev**2, m**2)
        return sqrt(max(0.0, sqrt(part_cur) - sqrt(part_prev)))
    if op == 8:  # dXC
        return sqrt(max(0.0, sqrt(cur * m) - sqrt(prev * m)))
    raise ValueError(f"Operator code must be between 0 and 8, got {op}.")


def choquet_integral(values: Sequence[float], operator: int | str, q: float = 1.0,
                     cap_at_one: bool = False) -> float:
    """Choquet-based integral of ``values`` w.r.t. the Power Measure (|A|/n)^q.

    Parameters
    ----------
    values : sequence of float
        Inputs (the function sorts them internally).
    operator : int or str
        Operator code (0-8) or name, see :data:`OPERATORS`.
    q : float
        Power Measure exponent (q > 0).
    cap_at_one : bool
        Apply the ``min{1, sum}`` cap of Definitions 5-6 to CF-AVG, CF-nonAVG
        and CF1F2. Disabled by default (see the README, "Methodological notes").
    """
    op = operator_code(operator)
    x = sorted(values)
    n = len(x)
    if n == 0:
        return 0.0

    total = 0.0
    for i in range(n):
        cur = x[i]
        prev = 0.0 if i == 0 else x[i - 1]
        m = power_measure(n - i, n, q)  # A_i = {x_(i), ..., x_(n)}
        if i == 0 and op in FIRST_TERM_RAW:
            total += cur
            continue
        total += _term(op, cur, prev, m)

    if cap_at_one and op in CAPPED_AT_ONE:
        total = min(1.0, total)
    return total


# ---------------------------------------------------------------------------
# Choquet-based integrals (many windows at once)
# ---------------------------------------------------------------------------
def _term_batch(op: int, cur: np.ndarray, prev: np.ndarray, m: float) -> np.ndarray:
    diff = cur - prev
    if op == 0:
        return diff * m
    if op == 1:
        return np.minimum(cur, m) - np.minimum(prev, m)
    if op == 2:
        den = diff + m - diff * m
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den == 0, 0.0, (diff * m) / den)
    if op == 3:
        return np.where(diff < m, diff, np.minimum(diff / 2, m))
    if op == 4:
        return np.where(diff == 0, 0.0,
                        np.where(diff <= m, (diff + m) / 2, np.minimum(diff / 2, m)))
    if op == 5:
        return np.sqrt(cur * m) - prev * m**2
    if op == 6:
        return np.sqrt(np.maximum(0.0, np.sqrt(cur) - np.sqrt(prev))) * m
    if op == 7:
        part_cur = np.minimum(cur, m) * np.maximum(cur * cur, m**2)
        part_prev = np.minimum(prev, m) * np.maximum(prev * prev, m**2)
        return np.sqrt(np.maximum(0.0, np.sqrt(part_cur) - np.sqrt(part_prev)))
    if op == 8:
        return np.sqrt(np.maximum(0.0, np.sqrt(cur * m) - np.sqrt(prev * m)))
    raise ValueError(f"Operator code must be between 0 and 8, got {op}.")


def choquet_integral_batch(windows: np.ndarray, operator: int | str, q: float = 1.0,
                           cap_at_one: bool = False) -> np.ndarray:
    """Vectorised :func:`choquet_integral` applied to each row of ``windows``."""
    op = operator_code(operator)
    x = np.sort(np.asarray(windows, dtype=float), axis=1)
    n_rows, n = x.shape
    total = np.zeros(n_rows)
    if n == 0:
        return total

    zeros = np.zeros(n_rows)
    for i in range(n):
        cur = x[:, i]
        prev = zeros if i == 0 else x[:, i - 1]
        m = power_measure(n - i, n, q)
        if i == 0 and op in FIRST_TERM_RAW:
            total = total + cur
            continue
        total = total + _term_batch(op, cur, prev, m)

    if cap_at_one and op in CAPPED_AT_ONE:
        total = np.minimum(1.0, total)
    return total
