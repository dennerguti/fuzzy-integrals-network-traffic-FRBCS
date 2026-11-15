"""Reproduce every table and figure of the network-traffic experiments.

Usage
-----
    python scripts/run_experiments.py                       # default: data/compilado.csv -> results/
    python scripts/run_experiments.py --data my.csv --out out/
    python scripts/run_experiments.py --excel-br            # CSVs with ';' and ',' (Excel pt-BR)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fintraffic import config, plots  # noqa: E402
from fintraffic.data import describe, load_raw  # noqa: E402
from fintraffic.export import latex_tables, wide_table  # noqa: E402
from fintraffic.forecasting import (Split, best_q_table, compare_with_pma,  # noqa: E402
                                    duplicated_tables, dxc_dcf_identity, q_curve_summary,
                                    q_curves, q_independent, run_grid, select_on_validation)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "compilado.csv",
                   help="CSV without header: time (HH:MM:SS), packets")
    p.add_argument("--out", type=Path, default=ROOT / "results", help="output directory")
    p.add_argument("--excel-br", action="store_true",
                   help="write CSVs with ';' separator and ',' decimal (Excel pt-BR)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out, fig_dir = args.out, args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_kw = (dict(sep=";", decimal=",", encoding="utf-8-sig") if args.excel_br
              else dict(encoding="utf-8"))

    def save_csv(df: pd.DataFrame, name: str) -> None:
        df.to_csv(out / name, index=False, **csv_kw)
        print(f"  -> {out / name}")

    def section(title: str) -> None:
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")

    # ------------------------------------------------------------------ data
    section("1. Data")
    raw = load_raw(args.data)
    series = raw["packets"].astype(float).to_numpy()
    for k, v in describe(raw).items():
        print(f"  {k:>20}: {v:g}" if isinstance(v, float) else f"  {k:>20}: {v}")
    split = Split.from_series(len(series), config.TEST_START)
    print(f"\n  Evaluation targets: t = {split.start}..{split.end - 1}")
    print(f"  Validation: t = {split.start}..{split.mid - 1} | Test: t = {split.mid}..{split.end - 1}")
    plots.plot_series(series, fig_dir / "series.pdf")
    plt.close("all")

    plots.paper_style()

    # ------------------------------------------------------- grid (paper q)
    section("2. MAE grid for the paper tables (q in [0.1, 10])")
    paper = run_grid(series, config.WINDOWS, config.Q_PAPER, *split.full)
    save_csv(paper, "mae_grid_paper.csv")
    save_csv(wide_table(paper), "mae_by_operator_paper.csv")
    (out / "tables_paper.tex").write_text(latex_tables(paper, config.Q_PAPER), encoding="utf-8")
    print(f"  -> {out / 'tables_paper.tex'}")

    tab_q = best_q_table(paper)
    print("\n  Best q per window (rows) and operator (columns):")
    print(tab_q.to_string())
    summary = pd.DataFrame({"windows_with_best_q>1": (tab_q > 1).sum(),
                            "windows_at_q=10": (tab_q == 10).sum(),
                            "total_windows": tab_q.notna().sum()})
    print("\n" + summary.to_string())
    majority = (summary["windows_with_best_q>1"] > summary["total_windows"] / 2).sum()
    print(f"\n  Operators with best q > 1 in most windows: {majority} of {len(summary)}")
    print("\n  Top 5 configurations:")
    print(paper[paper.method != "PMA"].nsmallest(5, "mae").to_string(index=False))
    print("\n  Identity d-XC(q) == dCF(q/4):")
    print(dxc_dcf_identity(paper).to_string(index=False))

    # ------------------------------------------------- grid (exploratory q)
    section("3. Exploratory MAE grid (q in 0.1..1.0 extended up to 3)")
    grid = run_grid(series, config.WINDOWS, config.Q_GRID, *split.full)
    save_csv(grid, "mae_grid.csv")
    save_csv(wide_table(grid), "mae_by_operator.csv")
    print(f"\n  Operator pairs with identical tables: {duplicated_tables(grid) or 'none'}")
    print(f"  Operators whose MAE does not depend on q: {q_independent(grid) or 'none'}")
    save_csv(best_q_table(grid).reset_index(), "best_q_by_window.csv")

    # ------------------------------------- selection on validation / test
    section("4. (window, q) selected on validation, MAE reported on test")
    selection = select_on_validation(series, config.WINDOWS, config.Q_GRID, split)
    print(selection.to_string(index=False))
    save_csv(selection, "selection_validation_test.csv")

    # --------------------------------------------------- CC-min q curves
    section("5. CC-min: validation MAE vs q")
    curves = q_curves(series, "CC-min", config.CURVE_WINDOWS, config.Q_CURVE, split)
    ccmin = q_curve_summary(series, "CC-min", curves, config.Q_CURVE, split)
    print(ccmin.round(2).to_string(index=False))
    save_csv(ccmin, "ccmin_q_by_window.csv")
    plots.plot_q_curves(curves, config.Q_CURVE, "CC-min", path=fig_dir / "ccmin_mae_vs_q.pdf")

    # ----------------------------------------------------- CC-min vs PMA
    section("6. CC-min vs PMA on the test period")
    best_q = dict(zip(ccmin["window"], ccmin["q_best"]))
    vs_pma = compare_with_pma(series, "CC-min", best_q, split, fixed_q=3.0)
    print(vs_pma.round(2).to_string(index=False))
    save_csv(vs_pma, "ccmin_vs_pma_test.csv")
    red = vs_pma["reduction_qbest_vs_PMA_%"]
    print(f"\n  Reduction vs PMA: {red.min():.1f}% to {red.max():.1f}%")

    # ------------------------------------------------------------ figures
    section("7. Figures")
    plots.plot_methods(paper, q=3.0, path=fig_dir / "comparison_operators_q3.pdf")
    plots.plot_methods(grid, q=0.5, methods=[m for m in plots.PAPER_LABELS if m != "CF1F2"],
                       title="Comparison of methods, q = 0.5",
                       path=fig_dir / "comparison_operators_q05.pdf")
    plots.plot_operator_by_q(grid, "dCC", path=fig_dir / "dcc_by_q.pdf")
    _, table = plots.plot_two_configs(grid, "dCC", 0.2, "CT", 0.1,
                                      path=fig_dir / "comparison_dCC_q02_CT_q01.pdf")
    save_csv(table, "comparison_dCC_q02_CT_q01.csv")
    plt.close("all")
    print(f"  -> {fig_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
