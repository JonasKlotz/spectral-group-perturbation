"""Table I: number n of band groups.

HSIC attribution with the Black baseline on ConvNeXtV2-Base for n = 1 (JP), 3, 4, 5 and
10 (single bands), at the same number of forward passes: 1024 masks for JP and
floor(1024 / n) per group otherwise. Evaluated on 300 random correctly predicted pairs of
the VALIDATION split, so the choice of n does not use the test pairs of Table II.

    python experiment_table_1.py run [--shard i/n] [--limit N]
    python experiment_table_1.py table
"""
from sgp.config import RESULTS, pairs_path
from sgp.runner import METRICS, Method, cli, load_rows, run
from sgp.stats import ALPHA, aligned, cell, p_value

OUT = f"{RESULTS}/table_1"
MODEL = "convnextv2"
SPLIT, N_PAIRS = "val", 300
N_GROUPS = [1, 3, 4, 5, 10]
REFERENCE = 3  # the n used for Table II; the other rows are tested against it
METHODS = [Method(f"hsic_n{n}", "hsic", n, "black", {"n_masks": 1024 // n}) for n in N_GROUPS]
COLUMNS = METRICS + ["OOD_z"]


def run_table_1(model, **kwargs):
    run(model, SPLIT, pairs_path(model, SPLIT, N_PAIRS), METHODS, OUT, **kwargs)


def table_1():
    rows = load_rows(OUT)
    names = [m.name for m in METHODS]
    ref = f"hsic_n{REFERENCE}"
    cells = {name: [] for name in names}
    for metric in COLUMNS:
        values = aligned(rows, names, metric)  # pairs where the metric is finite for every n
        means = {name: values[name].mean() for name in names}
        best = min(means.values()) if metric == "OOD_z" else max(means.values())
        for name in names:
            sig = name != ref and p_value(values[name], values[ref]) < ALPHA
            cells[name].append(cell(metric, means[name], bold=means[name] == best, mark="\\ddagger" if sig else ""))
        print(f"{metric:<6} n_pairs={len(values[ref])}  " + "  ".join(f"{n}:{means[n]:.3f}" for n in names))

    label = {1: "1 (JP)", 3: "3 (SGP)", 10: "10 (single band)"}
    lines = ["\\begin{tabular}{l" + "c" * len(COLUMNS) + "}", "\\toprule",
             "$n$ & IROF$\\uparrow$ & ROAD$\\uparrow$ & FE$\\uparrow$ & AUC$\\uparrow$ & RRA$\\uparrow$ & OOD z$\\downarrow$ \\\\",
             "\\midrule"]
    lines += [f"{label.get(m.n_groups, m.n_groups)} & " + " & ".join(cells[m.name]) + " \\\\" for m in METHODS]
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(f"{OUT}/table_1.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}/table_1.tex")


if __name__ == "__main__":
    cli(__doc__, run_table_1, table_1, [MODEL])
