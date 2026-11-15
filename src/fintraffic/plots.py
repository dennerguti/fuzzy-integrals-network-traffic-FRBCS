"""Figures of the paper."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from .operators import OPERATORS

#: Operator names as written in the paper figures.
PAPER_LABELS = {
    "Choquet": "Choquet", "CC-min": "CC-min", "CT": r"$C_T$", "CF-AVG": r"$C_F$ (AVG)",
    "CF-nonAVG": r"$C_F$ (N-AVG)", "CF1F2": r"$C_{F_1F_2}$", "dCF": r"$dC_F$",
    "dCC": "dCC", "dXC": "d-XC",
}


def paper_style() -> None:
    """Seaborn theme with large fonts and TrueType fonts embedded in PDFs."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.8)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def _save(fig: plt.Figure, path: str | Path | None, png: bool = False) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if png:
        fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")


def _plot_pma(ax: plt.Axes, results: pd.DataFrame) -> None:
    pma = results[results["method"] == "PMA"].sort_values("window")
    ax.plot(pma["window"], pma["mae"], color="black", linestyle="--", marker="s",
            markersize=9, linewidth=3, label="PMA")


def plot_series(series: np.ndarray, path=None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(series, linewidth=0.6)
    ax.set_title("Network traffic series")
    ax.set_xlabel("Time index (5 s steps)")
    ax.set_ylabel("Packets")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_methods(results: pd.DataFrame, q: float, methods: Sequence[str] | None = None,
                 include_pma: bool = True, labels: dict | None = None,
                 title: str | None = None, path=None) -> plt.Figure:
    """MAE vs window size of several operators for a fixed q (PMA as dashed black line)."""
    methods = list(OPERATORS.values()) if methods is None else list(methods)
    labels = PAPER_LABELS if labels is None else labels
    data = results[results["method"].isin(methods) & np.isclose(results["q"], q)].copy()
    data["label"] = data["method"].map(lambda m: labels.get(m, m))
    order = [labels.get(m, m) for m in methods if m != "PMA"]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=data, x="window", y="mae", hue="label", hue_order=order,
                 marker="o", markersize=9, linewidth=2.5, ax=ax)
    if include_pma:
        _plot_pma(ax, results)
    ax.set_title(title or f"Choquet-based operators with q = {q:g}", fontsize=22)
    ax.set_xlabel("Window size", fontsize=20)
    ax.set_ylabel("MAE", fontsize=20)
    ax.set_xticks(sorted(results["window"].unique()))
    ax.tick_params(axis="both", labelsize=17)
    ax.legend(title="Method", fontsize=16, title_fontsize=18,
              bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_operator_by_q(results: pd.DataFrame, operator: str,
                       q_values: Sequence[float] | None = None, include_pma: bool = True,
                       path=None) -> plt.Figure:
    """MAE vs window size of one operator, one line per q."""
    data = results[results["method"] == operator].copy()
    q_values = sorted(data["q"].unique()) if q_values is None else list(q_values)
    data = data[data["q"].isin(q_values)]
    data["q_label"] = pd.Categorical(data["q"].map(lambda v: f"{v:g}"),
                                     categories=[f"{v:g}" for v in q_values], ordered=True)

    fig, ax = plt.subplots(figsize=(13, 8))
    sns.lineplot(data=data.sort_values(["q", "window"]), x="window", y="mae", hue="q_label",
                 palette="viridis", marker="o", linewidth=2, ax=ax)
    if include_pma:
        _plot_pma(ax, results)
    ax.set_title(f"{PAPER_LABELS.get(operator, operator)} for different values of q")
    ax.set_xlabel("Sliding window size")
    ax.set_ylabel("MAE")
    ax.set_xticks(sorted(data["window"].unique()))
    ax.legend(title="q", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_two_configs(results: pd.DataFrame, method_1: str, q_1: float, method_2: str,
                     q_2: float, include_pma: bool = True, path=None
                     ) -> tuple[plt.Figure, pd.DataFrame]:
    """Compare two (operator, q) configurations across windows. Returns figure and table."""
    sel = []
    for m, q in ((method_1, q_1), (method_2, q_2)):
        d = results[(results["method"] == m) & np.isclose(results["q"], q)].copy()
        if d.empty:
            raise ValueError(f"No results for {m} with q = {q}.")
        d["config"] = f"{PAPER_LABELS.get(m, m)} (q={q:g})"
        sel.append(d)
    data = pd.concat(sel, ignore_index=True)

    fig, ax = plt.subplots(figsize=(13, 8))
    sns.lineplot(data=data, x="window", y="mae", hue="config", style="config", markers=True,
                 dashes=False, linewidth=2.5, markersize=9, ax=ax)
    if include_pma:
        _plot_pma(ax, results)
    ax.set_title(" vs. ".join(data["config"].unique()), fontsize=22, pad=15)
    ax.set_xlabel("Sliding window size", fontsize=19)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=19)
    ax.set_xticks(sorted(data["window"].unique()))
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(title="Method", title_fontsize=18, fontsize=17,
              bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, path)

    a, b = sel[0].set_index("window")["mae"], sel[1].set_index("window")["mae"]
    names = sel[0]["config"].iloc[0], sel[1]["config"].iloc[0]
    table = pd.DataFrame({names[0]: a, names[1]: b})
    table["best"] = np.where(a < b, names[0], np.where(b < a, names[1], "tie"))
    table["abs_diff"] = (a - b).abs()
    return fig, table.reset_index()


def plot_q_curves(curves: dict[int, list[float]], q_values: Sequence[float], operator: str,
                  zoom: tuple[float, float] | None = (2, 4),
                  ticks: Sequence[float] = (0.1, 0.2, 0.5, 1, 2, 3, 5, 10),
                  path=None) -> plt.Figure:
    """Validation MAE vs q (log scale), one curve per window, minima marked with a star."""
    q_values = list(q_values)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {}
    for w, maes in curves.items():
        (line,) = ax.plot(q_values, maes, marker="o", markersize=5, linewidth=2,
                          label=f"Window {w}")
        colors[w] = line.get_color()
        k = int(np.argmin(maes))
        ax.plot(q_values[k], maes[k], marker="*", markersize=15, color=colors[w],
                markeredgecolor="black", zorder=5)
    ax.axvline(1, color="gray", linestyle="--", linewidth=1.2, label="q = 1 (additive measure)")

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_title(f"{operator}: validation MAE as a function of q", fontsize=18, pad=12)
    ax.set_xlabel("Power Measure exponent (q)", fontsize=15, labelpad=8)
    ax.set_ylabel("Mean Absolute Error (validation)", fontsize=15, labelpad=8)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(alpha=0.3)

    if zoom is not None:
        axz = inset_axes(ax, width="45%", height="45%", loc="upper right", borderpad=1.5)
        lo, hi = np.inf, -np.inf
        for w, maes in curves.items():
            sel = [(q, m) for q, m in zip(q_values, maes) if zoom[0] <= q <= zoom[1]]
            if not sel:
                continue
            qs, ms = zip(*sel)
            axz.plot(qs, ms, marker="o", markersize=3, linewidth=1.5, color=colors[w])
            k = int(np.argmin(maes))
            if zoom[0] <= q_values[k] <= zoom[1]:
                axz.plot(q_values[k], maes[k], marker="*", markersize=14, color=colors[w],
                         markeredgecolor="black", zorder=5)
            lo, hi = min(lo, min(ms)), max(hi, max(ms))
        margin = 0.04 * (hi - lo)
        axz.set_xlim(zoom[0] - 0.05, zoom[1] + 0.05)
        axz.set_ylim(lo - margin, hi + margin)
        axz.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axz.tick_params(labelsize=10)
        axz.grid(alpha=0.3)
        axz.set_title(f"Zoom: q in [{zoom[0]:g}, {zoom[1]:g}]", fontsize=11)
        mark_inset(ax, axz, loc1=3, loc2=4, fc="none", ec="0.5", lw=0.8)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="none", marker="*", markersize=14,
                          markerfacecolor="lightgray", markeredgecolor="black"))
    labels.append("Minimum of each curve")
    ax.legend(handles, labels, title="Sliding window", title_fontsize=13, fontsize=12,
              bbox_to_anchor=(1.02, 1), loc="upper left")
    if zoom is None:  # inset axes are not compatible with tight_layout
        fig.tight_layout()
    _save(fig, path, png=True)
    return fig
