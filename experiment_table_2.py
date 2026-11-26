"""Table II: JP vs. SGP.

ConvNeXtV2-Base and MobileViT-S, HSIC attribution and Occlusion, Black and Global mean
baselines, JP (n = 1) and SGP (n = 3), on 1000 random correctly predicted test pairs per
model. Equal forward passes: HSIC 1024 masks (JP) vs. 3 x 341; Occlusion with a 50 px
window, stride 5 px for JP (226 passes) and 8 px for SGP (246 passes).

    python experiment_table_2.py run [--model M] [--shard i/n] [--limit N]
    python experiment_table_2.py table
"""
from sgp.config import RESULTS, pairs_path
from sgp.runner import METRICS, Method, cli, load_rows, run
from sgp.stats import ALPHA, aligned, better, cell, p_value

OUT = f"{RESULTS}/table_2"
MODELS = ["convnextv2", "mobilevit"]
MODEL_NAMES = {"convnextv2": "ConvNeXtV2-Base", "mobilevit": "MobileViT-S"}
SPLIT, N_PAIRS = "test", 1000
BASELINES = {"black": "Black", "gmean": "Global mean"}
METHODS = [m for b in BASELINES for m in (
    Method(f"hsic_jp_{b}", "hsic", 1, b, {"n_masks": 1024}),
    Method(f"hsic_sgp_{b}", "hsic", 3, b, {"n_masks": 341}),
    Method(f"occlusion_jp_{b}", "occlusion", 1, b, {"window": 50, "stride": 5}),
    Method(f"occlusion_sgp_{b}", "occlusion", 3, b, {"window": 50, "stride": 8}),
)]
COLUMNS = METRICS + ["OOD_z"]


def run_table_2(model, **kwargs):
    run(model, SPLIT, pairs_path(model, SPLIT, N_PAIRS), METHODS, OUT, **kwargs)


def table_2():
    """Per model, baseline and explainer: means of JP and SGP on the pairs where the metric is
    finite for both, better value in bold, SGP marked with dagger / asterisk if significantly
    better / worse than JP."""
    rows = load_rows(OUT)
    lines = ["\\begin{tabular}{lll" + "c" * len(COLUMNS) * len(MODELS) + "}", "\\toprule",
             " & & & " + " & ".join(f"\\multicolumn{{{len(COLUMNS)}}}{{c}}{{{MODEL_NAMES[m]}}}" for m in MODELS) + " \\\\",
             "Baseline & Method & Grouping & " + " & ".join(
                 ["IROF$\\uparrow$", "ROAD$\\uparrow$", "FE$\\uparrow$", "AUC$\\uparrow$", "RRA$\\uparrow$",
                  "OOD z$\\downarrow$"] * len(MODELS)) + " \\\\", "\\midrule"]
    for baseline, baseline_name in BASELINES.items():
        for explainer, explainer_name in (("hsic", "HSIC"), ("occlusion", "Occlusion")):
            jp, sgp = f"{explainer}_jp_{baseline}", f"{explainer}_sgp_{baseline}"
            cells = {jp: [], sgp: []}
            for model in MODELS:
                model_rows = rows[rows["model"] == model]
                for metric in COLUMNS:
                    v = aligned(model_rows, [jp, sgp], metric)
                    a, b = v[jp].mean(), v[sgp].mean()
                    sig = p_value(v[sgp], v[jp]) < ALPHA
                    mark = ("\\dagger" if better(metric, b, a) else "\\ast") if sig else ""
                    cells[jp].append(cell(metric, a, bold=better(metric, a, b)))
                    cells[sgp].append(cell(metric, b, bold=better(metric, b, a), mark=mark))
                    print(f"{model:<11} {baseline:<6} {explainer:<10} {metric:<6} JP {a:.3f}  SGP {b:.3f}{'*' if sig else ''}"
                          f"  (n={len(v[jp])})")
            first = explainer == "hsic"
            lines.append(f"{baseline_name if first else ''} & {explainer_name} & JP & " + " & ".join(cells[jp]) + " \\\\")
            lines.append(f" & & SGP & " + " & ".join(cells[sgp]) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    with open(f"{OUT}/table_2.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}/table_2.tex")


if __name__ == "__main__":
    cli(__doc__, run_table_2, table_2, MODELS)
