"""Loading and diagnostics of the network-traffic series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV as ``time`` (HH:MM:SS) and ``packets``.

    The file has **no header**: its first line is already an observation
    (``08:56:01,8``). Reading it without ``header=None`` silently drops it.
    """
    return pd.read_csv(path, header=None, names=["time", "packets"])


def load_series(path: str | Path) -> np.ndarray:
    """Packet counts as a float array, in file order."""
    return load_raw(path)["packets"].astype(float).to_numpy()


def describe(raw: pd.DataFrame) -> dict:
    """Basic diagnostics: size, NaN, zeros, skewness and timestamp regularity."""
    s = raw["packets"].astype(float)
    t = pd.to_timedelta(raw["time"].astype(str), errors="coerce")
    steps = t.diff().dt.total_seconds().dropna()
    typical = steps.mode().iloc[0] if len(steps) else np.nan
    return {
        "observations": int(len(s)),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "skewness": float(s.skew()),
        "nan": int(raw.isna().sum().sum()),
        "zeros": int((s == 0).sum()),
        "invalid_timestamps": int(t.isna().sum()),
        "repeated_timestamps": int(t.duplicated().sum()),
        "typical_step_s": float(typical),
        "negative_steps": int((steps < 0).sum()),   # day changes / out of order
        "gaps": int((steps > typical).sum()),
    }
