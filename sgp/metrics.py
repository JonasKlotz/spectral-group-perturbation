"""Evaluation metrics; higher is better for all of them.

  IROF  iterative removal of the most relevant of 25 SLIC superpixels, Black fill (Quantus)
  ROAD  remove and debias, most relevant pixels first (multi-label version below)
  FE    faithfulness estimate, steps of 400 px, Black fill (Quantus)
  AUC   ROC AUC of the map against the reference map of the class (Quantus)
  RRA   relevance rank accuracy against the reference map (Quantus)

Black = minimum of the image. AUC and RRA are NaN where the class has no reference pixels.
"""
import numpy as np
import torch

from sgp import quantus
from sgp.quantus.functions.perturb_func import noisy_linear_imputation

METRICS = ["IROF", "ROAD", "FE", "AUC", "RRA"]
_COMMON = dict(abs=False, normalise=False, return_aggregate=False, disable_warnings=True, multi_label=True)


class ROAD:
    """ROAD (Rong et al. 2022) for multi-label classification, without retraining. Quantus'
    ROAD scores single-label accuracy over a dataset; this one scores every (image, class)
    pair: the p % most relevant pixels (p = 10, 20, ..., 90) are replaced in all bands by
    noisy linear imputation (Quantus), and the score is 1 - mean_p sigmoid(f(x_p))_c /
    sigmoid(f(x))_c, clipped to [0, 1]. Ties in the map are broken randomly, seeded per pair."""

    def __init__(self, percentages=tuple(range(10, 100, 10)), noise: float = 0.01, seed: int = 0):
        self.percentages, self.noise, self.seed = list(percentages), noise, seed

    @torch.no_grad()
    def __call__(self, model, x_batch, y_batch, a_batch, sample_ids, **kwargs):
        B, K = y_batch.shape
        n_pix = x_batch.shape[-2] * x_batch.shape[-1]
        device = next(model.parameters()).device
        out = np.full((B, K), np.nan)
        for i in range(B):
            classes = np.flatnonzero(y_batch[i])
            if len(classes) == 0:
                continue
            x = x_batch[i]
            p0 = torch.sigmoid(model(torch.from_numpy(x[None]).to(device))).cpu().numpy()[0]
            for c in classes:
                a = a_batch[i, c].reshape(-1).astype(np.float64)
                tie = np.random.default_rng((self.seed, int(sample_ids[i]), int(c))).random(n_pix)
                ranked = np.lexsort((tie, -a))  # most relevant first, random among ties
                imgs = np.stack([noisy_linear_imputation(x, ranked[: int(n_pix * p / 100)], noise=self.noise)
                                 for p in self.percentages]).astype(x.dtype)
                probs = torch.sigmoid(model(torch.from_numpy(imgs).to(device))).cpu().numpy()[:, c]
                out[i, c] = float(np.clip(1.0 - np.mean(probs / (p0[c] + 1e-12)), 0.0, 1.0))
        return out


def build_metrics() -> dict:
    return {
        "IROF": quantus.IROF(segmentation_method="slic", perturb_baseline="black", perturb_order="morf", **_COMMON),
        "ROAD": ROAD(),
        "FE": quantus.FaithfulnessEstimate(features_in_step=400, perturb_baseline="black", **_COMMON),
        "AUC": quantus.AUC(**_COMMON),
        "RRA": quantus.RelevanceRankAccuracy(**_COMMON),
    }


def evaluate(metrics: dict, model, x, y, s, a, sample_ids, device) -> dict:
    """x (B, C, H, W), y (B, K) explained classes, s (B, K, 1, H, W) reference maps,
    a (B, K, 1, H, W) attribution maps -> {metric: (B, K) array}."""
    kw = dict(model=model, x_batch=x, y_batch=y, s_batch=s, a_batch=a, device=device, softmax=True, channel_first=True)
    out = {}
    for name, metric in metrics.items():
        extra = {"sample_ids": sample_ids} if isinstance(metric, ROAD) else {}
        out[name] = np.asarray(metric(**kw, **extra), dtype=float)
    return out
