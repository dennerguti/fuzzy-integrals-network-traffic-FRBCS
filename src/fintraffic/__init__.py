"""Choquet-based fuzzy integrals with the Power Measure for network-traffic forecasting."""

from .data import describe, load_raw, load_series
from .forecasting import (DEFAULT_CONFIG, AggregationConfig, Split, aggregate_window,
                          aggregate_windows, best_q_table, compare_with_pma, duplicated_tables,
                          dxc_dcf_identity, forecast, mae, q_curve_summary, q_curves,
                          q_independent, run_grid, select_on_validation)
from .operators import OPERATORS, choquet_integral, choquet_integral_batch, pma, pma_weights

__version__ = "1.0.0"
