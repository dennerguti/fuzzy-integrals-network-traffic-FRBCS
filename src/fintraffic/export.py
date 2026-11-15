"""Tables (wide CSV and LaTeX) built from grid results."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .operators import OPERATORS

#: Display name and LaTeX label of each operator (adjust to the paper notation).
LATEX_NAMES = {
    "Choquet":   ("Choquet integral",         "choquet"),
    "CC-min":    ("$C_C$-integral (min)",     "ccmin"),
    "CT":        ("$C_T$-integral",           "ct"),
    "CF-AVG":    ("$C_F$-integral (AVG)",     "cfavg"),
    "CF-nonAVG": ("$C_F$-integral (non-AVG)", "cfnonavg"),
    "CF1F2":     ("$C_{F_1F_2}$-integral",    "cf1f2"),
    "dCF":       ("$d$-$C_F$-integral",       "dcf"),
    "dCC":       ("$d$-$C_C$-integral",       "dcc"),
    "dXC":       ("$d$-$X_C$-integral",       "dxc"),
}


def wide_table(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (window, q), one column per method (PMA repeated for every q)."""
    choquet = (results[results["method"] != "PMA"]
               .pivot_table(index=["window", "q"], columns="method", values="mae")
               .reset_index())
    choquet.columns.name = None
    pma = (results[results["method"] == "PMA"][["window", "mae"]]
           .rename(columns={"mae": "PMA"}))
    table = choquet.merge(pma, on="window", how="left").sort_values(["window", "q"])
    ops = [m for m in OPERATORS.values() if m in table.columns]
    return table[["window", "q", "PMA", *ops]].reset_index(drop=True)


def latex_table(results: pd.DataFrame, operator: str, q_values: Sequence[float],
                highlight_best: bool = True, color: str = "") -> str:
    """LaTeX table (window x q) of the MAE of one operator."""
    tab = (results[results["method"] == operator]
           .pivot(index="window", columns="q", values="mae")[list(q_values)])
    title, label = LATEX_NAMES[operator]
    header = " & ".join([r"\textbf{Window}"] + [rf"\textbf{{$q={q:g}$}}" for q in q_values])

    body = []
    for window, row in tab.iterrows():
        best = row.min()
        cells = [rf"\textbf{{{v:.4f}}}" if highlight_best and v == best else f"{v:.4f}"
                 for v in row]
        body.append(f"{window} & " + " & ".join(cells) + r" \\")

    caption = (rf"Results (MAE) for the {title} with the Power Measure exponent "
               rf"$q \in [{min(q_values):g}, {max(q_values):g}]$.")
    if highlight_best:
        caption += " The lowest MAE for each window is shown in bold."

    lines = [r"\begin{table*}[t]", color, r"\centering", rf"\caption{{{caption}}}",
             rf"\label{{tab:{label}_q}}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{c" + "c" * len(q_values) + "}", r"\toprule",
             header + r" \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}%", r"}",
             r"\end{table*}"]
    return "\n".join(line for line in lines if line)


def latex_tables(results: pd.DataFrame, q_values: Sequence[float], **kwargs) -> str:
    """One LaTeX table per operator, concatenated."""
    return "\n\n".join(latex_table(results, op, q_values, **kwargs) for op in OPERATORS.values())
