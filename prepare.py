"""Inputs of the experiments. Every step is skipped if its output exists.

  1 predictions   sigmoid scores, labels and class pixel counts per image of a split
  2 pairs         N (image, class) pairs drawn uniformly from the correctly predicted ones
  3 features      GAP features of ConvNeXtV2 (train: reference bank, test: null baseline)
  4 ood_null      kNN distance of the unperturbed test images to the train features (OOD z)
  5 correlation   Pearson correlation of the 10 bands over training pixels (band groups)

The paper uses: test predictions and 1000 pairs for both models, 300 pairs from the
predictions of the first 9600 validation images (ConvNeXtV2), features of train and test,
and the correlation over 1000 shuffled training batches of 32 images.

    python prepare.py            # all steps
    python prepare.py pairs      # one step
"""
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from sgp.config import BAND_CORRELATION, BANDS, FEATURE_BANK, OOD_K, OOD_NULL, OUTPUTS, pairs_path, split_dir
from sgp.data import BENv2
from sgp.models import GAPEncoder, load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32
SEED = 42


def batches(dataset, max_batches=None, shuffle=False):
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=shuffle, num_workers=8,
                                         generator=torch.Generator().manual_seed(SEED))
    for i, (x, y, ref) in enumerate(tqdm(loader, total=max_batches or len(loader))):
        if max_batches is not None and i >= max_batches:
            break
        yield x, y.numpy(), ref.numpy()


def predictions(model_name, split, max_batches=None):
    path = f"{split_dir(model_name, split)}/pred_gt_nr_size.npz"
    if os.path.exists(path):
        return print(f"exists: {path}")
    model = load_model(model_name, DEVICE)
    out = {k: [] for k in ("y_true", "y_score", "pixel_counts")}
    for x, y, ref in batches(BENv2(split), max_batches):
        with torch.no_grad():
            out["y_score"].append(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy().astype(np.float32))
        out["y_true"].append(y.astype(np.uint8))
        counts = (ref[:, None] == np.arange(y.shape[1])[None, :, None, None]).sum(axis=(-2, -1))
        out["pixel_counts"].append((counts * y).astype(np.int32))
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["y_pred"] = (out["y_score"] > 0.5).astype(np.uint8)
    out["indices"] = np.arange(len(out["y_true"]), dtype=np.int64)
    out["n_gt"], out["n_pred"] = out["y_true"].sum(1).astype(np.int32), out["y_pred"].sum(1).astype(np.int32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    print(f"saved {path}")


def pairs(model_name, split, n):
    path = pairs_path(model_name, split, n)
    if os.path.exists(path):
        return print(f"exists: {path}")
    data = np.load(f"{split_dir(model_name, split)}/pred_gt_nr_size.npz")
    rows, classes = np.nonzero(data["y_true"] & data["y_pred"])
    candidates = np.stack([data["indices"][rows], classes], axis=1).astype(np.int64)
    keep = np.sort(np.random.default_rng(SEED).choice(len(candidates), size=min(n, len(candidates)), replace=False))
    np.save(path, candidates[keep])
    print(f"saved {path}: {len(keep)} of {len(candidates)} correctly predicted pairs")


def features(split):
    path = f"{OUTPUTS}/features/benv2/{split}/features.npy"
    if os.path.exists(path):
        return print(f"exists: {path}")
    encoder = GAPEncoder(load_model("convnextv2", DEVICE))
    feats = [encoder(x.to(DEVICE)).cpu().numpy().astype(np.float32) for x, _, _ in batches(BENv2(split))]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.concatenate(feats))
    print(f"saved {path}")


def ood_null():
    if os.path.exists(OOD_NULL):
        return print(f"exists: {OOD_NULL}")
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=OOD_K, metric="cosine").fit(np.load(FEATURE_BANK))
    d = knn.kneighbors(np.load(f"{OUTPUTS}/features/benv2/test/features.npy"))[0].mean(axis=1)
    np.savez(OOD_NULL, mean=float(d.mean()), std=float(d.std(ddof=1)), per_sample=d)
    print(f"saved {OOD_NULL}: mean {d.mean():.4f}, std {d.std(ddof=1):.4f}")


def correlation(max_batches=1000):
    if os.path.exists(BAND_CORRELATION):
        return print(f"exists: {BAND_CORRELATION}")
    C = len(BANDS)
    total, outer, n = np.zeros(C), np.zeros((C, C)), 0
    for x, _, _ in batches(BENv2("train"), max_batches, shuffle=True):
        flat = x.numpy().transpose(1, 0, 2, 3).reshape(C, -1).astype(np.float64)
        total, outer, n = total + flat.sum(1), outer + flat @ flat.T, n + flat.shape[1]
    mean = total / n
    cov = outer / n - np.outer(mean, mean)
    std = np.sqrt(np.diag(cov))
    os.makedirs(os.path.dirname(BAND_CORRELATION), exist_ok=True)
    np.save(BAND_CORRELATION, cov / np.outer(std, std))
    print(f"saved {BAND_CORRELATION} ({n:,} pixels)")


STEPS = {
    "predictions": lambda: (predictions("convnextv2", "test"), predictions("mobilevit", "test"),
                            predictions("convnextv2", "val", max_batches=300)),
    "pairs": lambda: (pairs("convnextv2", "test", 1000), pairs("mobilevit", "test", 1000), pairs("convnextv2", "val", 300)),
    "features": lambda: (features("train"), features("test")),
    "ood_null": ood_null,
    "correlation": correlation,
}

if __name__ == "__main__":
    for step in sys.argv[1:] or STEPS:
        print(f"== {step}")
        STEPS[step]()
