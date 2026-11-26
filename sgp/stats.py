"""Means and paired two-sided Wilcoxon signed-rank tests over (image, class) pairs, and LaTeX cells."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ALPHA = 0.05
LOWER_IS_BETTER = {"OOD_z"}


def aligned(rows: pd.DataFrame, methods: list[str], metric: str) -> dict:
    """{method: values} over the pairs where the metric is finite for every method,
    in the same pair order (so they can be tested pairwise)."""
    wide = rows[rows["method"].isin(methods)].pivot_table(index=["sample", "cls"], columns="method",
                                                          values=metric, aggfunc="first")
    wide = wide[methods].replace([np.inf, -np.inf], np.nan).dropna()
    return {m: wide[m].to_numpy() for m in methods}


def p_value(a, b) -> float:
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:  # all differences zero
        return float("nan")


def better(metric: str, a: float, b: float) -> bool:
    return a < b if metric in LOWER_IS_BETTER else a > b


def cell(metric: str, value: float, bold: bool = False, mark: str = "") -> str:
    s = f"{value:.2f}" if metric == "OOD_z" else f"{value:.3f}"
    s = f"\\textbf{{{s}}}" if bold else s
    return s + (f"$^{{{mark}}}$" if mark else "")
