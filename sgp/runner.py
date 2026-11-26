"""The loop shared by all experiments: explain every (image, class) pair with every method,
evaluate the maps and write one CSV row per pair and method.

Output of run(): {out_dir}/{model}/rows_{i}of{n}.csv per shard (appended batch by batch, so
an interrupted run continues where it stopped) and, with save_maps, the group maps
{out_dir}/{model}/maps/{method}/{sample}_{class}.npy, shape (n_groups, H, W), unrectified.
"""
import argparse
import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sgp.config import NUM_CLASSES
from sgp.data import BENv2, load_pairs
from sgp.explainers import HSIC, Occlusion, aggregate
from sgp.groups import band_groups, describe
from sgp.metrics import METRICS, build_metrics, evaluate
from sgp.models import load_model
from sgp.ood import OODScore


@dataclass(frozen=True)
class Method:
    """One explanation setting: explainer, number of band groups (1 = JP), baseline and the
    explainer's budget (HSIC: n_masks per group; Occlusion: window and stride in px)."""
    name: str
    explainer: str  # "hsic" or "occlusion"
    n_groups: int
    baseline: str   # "black" or "gmean"
    params: dict = field(default_factory=dict, hash=False)

    def build(self, model):
        groups = band_groups(self.n_groups)
        cls = {"hsic": HSIC, "occlusion": Occlusion}[self.explainer]
        return cls(model, groups, self.baseline, **self.params)


def run(model_name: str, split: str, pairs_file: str, methods: list[Method], out_dir: str, shard: str = "0/1",
        device: str = "cuda", limit: int | None = None, metrics: bool = True, ood: bool = True,
        save_maps: bool = False, batch_size: int = 32):
    pairs = load_pairs(pairs_file)
    samples = np.unique(pairs[:, 0])
    if limit is not None:
        samples = samples[:limit]
    i_shard, n_shards = (int(v) for v in shard.split("/"))
    samples = samples[i_shard::n_shards]
    pair_set = set(map(tuple, pairs.tolist()))

    out = os.path.join(out_dir, model_name)
    os.makedirs(out, exist_ok=True)
    rows_path = os.path.join(out, f"rows_{i_shard}of{n_shards}.csv")
    if os.path.exists(rows_path):
        samples = samples[~np.isin(samples, pd.read_csv(rows_path)["sample"].unique())]
    print(f"{model_name} / {split}: {len(samples)} images to do, shard {shard}, output {out}")

    model = load_model(model_name, device)
    dataset = BENv2(split)
    explainers = {m.name: m.build(model) for m in methods}
    for m in methods:
        print(f"  {m.name:<24} {describe(explainers[m.name].groups)}  {m.params}")
    metric_fns = build_metrics() if metrics else {}
    ood_score = OODScore(device) if ood else None

    for start in tqdm(range(0, len(samples), batch_size), desc=model_name):
        ids = samples[start:start + batch_size]
        x, y, s = dataset.batch(ids)
        xt = torch.from_numpy(x).to(device)
        with torch.no_grad():
            correct = (torch.sigmoid(model(xt)) > 0.5).cpu().numpy() & (y > 0)
        in_pairs = np.array([[(int(sid), c) in pair_set for c in range(NUM_CLASSES)] for sid in ids])
        y_explained = (correct & in_pairs).astype(np.int64)
        explained = list(zip(*np.nonzero(y_explained)))

        rows = []
        for m in methods:
            explainer = explainers[m.name]
            a = np.zeros((len(ids), NUM_CLASSES, 1, *x.shape[-2:]), dtype=np.float32)
            for i, c in explained:
                maps = explainer(xt[i:i + 1], int(c))
                a[i, c, 0] = aggregate(maps, explainer.groups).cpu().numpy()
                if save_maps:
                    os.makedirs(f"{out}/maps/{m.name}", exist_ok=True)
                    np.save(f"{out}/maps/{m.name}/{ids[i]}_{c}.npy", maps.cpu().numpy())
            scores = evaluate(metric_fns, model, x, y_explained, s, a, ids, device) if metrics else {}
            oods = {i: ood_score(explainer, xt[i:i + 1], int(ids[i])) for i in {i for i, _ in explained}} if ood else {}
            for i, c in explained:
                row = dict(model=model_name, method=m.name, sample=int(ids[i]), cls=int(c), size=int(s[i, c].sum()))
                row.update({k: float(v[i, c]) for k, v in scores.items()})
                if ood:
                    row["OOD_z"] = oods[i]
                rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(rows_path, mode="a", header=not os.path.exists(rows_path), index=False)


def load_rows(out_dir: str) -> pd.DataFrame:
    """All rows of all models and shards of one experiment."""
    files = sorted(glob.glob(os.path.join(out_dir, "*", "rows_*.csv")))
    if not files:
        raise FileNotFoundError(f"no results in {out_dir}; run the experiment first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def cli(description: str, run_fn, table_fn, models: list[str]):
    """python experiment_x.py run [--model M] [--shard i/n] [--limit N] [--device D] | table"""
    p = argparse.ArgumentParser(description=description)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="compute the explanations and metrics (one shard)")
    r.add_argument("--model", choices=models, default=None, help="only this model (default: all)")
    r.add_argument("--shard", default="0/1", help="i/n: every n-th image starting at i")
    r.add_argument("--limit", type=int, default=None, help="only the first N images (quick test)")
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub.add_parser("table", help="merge the shards and write the table")
    args = p.parse_args()
    if args.cmd == "run":
        for model in [args.model] if args.model else models:
            run_fn(model, shard=args.shard, limit=args.limit, device=args.device)
    else:
        table_fn()


__all__ = ["Method", "run", "load_rows", "cli", "METRICS"]
