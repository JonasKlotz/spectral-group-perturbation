# Spectral Group Perturbation (SGP)

Code for *Spectral Group Perturbation for Perturbation-Based Explanations of Multispectral
Remote Sensing Images* (IGARSS 2027). The SGP partitions the bands of a multispectral image
into groups of correlated bands, perturbs each group separately in a perturbation-based
explanation method (Occlusion, HSIC attribution), and sums the rectified group maps into one
spatial explanation.

## Layout

```
prepare.py              inputs: predictions, image-class pairs, OOD features and null baseline, band correlation
experiment_table_1.py   Table I:  number of band groups n = 1, 3, 4, 5, 10 (HSIC, ConvNeXtV2, 300 val pairs)
experiment_table_2.py   Table II: JP vs. SGP, 2 models x {HSIC, Occlusion} x {Black, Global mean}, 1000 test pairs
experiment_fig_2.py     Fig. 2:   Occlusion maps (JP, SGP groups) for the 1000 test pairs, and the figure panels
sgp/
  config.py             paths and constants
  data.py               BigEarthNet v2.0 with reference maps
  models.py             reBEN classifiers (Hugging Face) and the GAP feature encoder
  groups.py             band groups by clustering the band correlation (Eq. 2)
  explainers.py         Occlusion and HSIC with group-wise perturbation, rectification and aggregation (Eq. 3-4)
  metrics.py            IROF, ROAD (multi-label), FE, AUC, RRA
  ood.py                OOD z of the perturbed images
  runner.py             the loop shared by all experiments
  stats.py              Wilcoxon tests and table cells
  quantus/              vendored, modified Quantus 0.5.3 (multi-label), only the metrics used here
scripts/run.slurm       runs an experiment in parallel shards on one GPU, then writes its table
```

## Setup

```bash
uv sync
export BENV2_BASE_PATH=/path/to/BigEarthNet-V2   # BENv2.lmdb, BENv2_Reference_Maps.lmdb, metadata.parquet
export SGP_OUTPUTS=/path/to/outputs              # default /data/jonas/outputs/rs_perturbation
```

The model weights ([ConvNeXtV2-Base](https://huggingface.co/BIFOLD-BigEarthNetv2-0/convnextv2_base-s2-v0.2.0),
[MobileViT-S](https://huggingface.co/BIFOLD-BigEarthNetv2-0/mobilevit_s-s2-v0.2.0)) are downloaded from Hugging Face.

## Running

```bash
uv run python prepare.py                       # once; skips existing outputs

uv run python experiment_table_1.py run        # or: sbatch --job-name=table1 scripts/run.slurm experiment_table_1.py
uv run python experiment_table_1.py table      # -> $SGP_OUTPUTS/paper/table_1/table_1.tex

uv run python experiment_table_2.py run --model convnextv2 --shard 0/3
uv run python experiment_table_2.py table      # -> $SGP_OUTPUTS/paper/table_2/table_2.tex

uv run python experiment_fig_2.py run
uv run python experiment_fig_2.py plot --pairs 79139:11 79224:11   # -> $SGP_OUTPUTS/paper/fig_2/img, values.tex
```

`run` options: `--model`, `--shard i/n` (every n-th image from i; shards of one experiment
are merged by `table`), `--limit N` (first N images, for tests), `--device`. An interrupted
run continues where it stopped. Results are one CSV row per (image, class) pair and method in
`$SGP_OUTPUTS/paper/<experiment>/<model>/rows_*.csv`.

## Settings

| | |
|---|---|
| Band groups | average-linkage clustering of 1 - r over training pixels; n = 3: C1 = B02-B05, C2 = B06-B8A, C3 = B11-B12 |
| HSIC | 7 x 7 grid, a cell is perturbed with probability 0.3, 1024 masks (JP) or floor(1024 / n) per group; SGP with the direction gate, JP without |
| Occlusion | 50 x 50 px window, stride 5 px (JP, 226 forward passes) and 8 px (SGP, 246) |
| Baselines | Black: image minimum; Global mean: image mean over all bands and pixels |
| Aggregation | sum over groups of the rectified group map times the number of bands in the group |
| Pairs | correctly predicted (image, class) pairs drawn uniformly at random (seed 42) |
| OOD z | 32 perturbed copies per group, mean cosine distance of ConvNeXtV2 GAP features to the 25 nearest training images, standardised with the unperturbed test images |
| Tests | paired two-sided Wilcoxon signed-rank test, p < 0.05 |
