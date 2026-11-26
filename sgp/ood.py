"""OOD z: how far the perturbed images of an explainer are from the training distribution.

For every band group, 32 perturbed copies of the image are drawn as the explainer builds
them (HSIC: 32 of its masks, Occlusion: 32 window positions). Each copy is encoded with the
GAP features of a fixed ConvNeXtV2, and its distance is the mean cosine distance to its
k = 25 nearest training images. The distances are averaged over the copies and the groups
and standardised with the mean and std of the same distance for the unperturbed test
images (null baseline, prepare.py). The value depends only on the image, not on the class.
"""
import numpy as np
import torch

from sgp.config import FEATURE_BANK, OOD_ENCODER, OOD_K, OOD_NULL
from sgp.models import GAPEncoder, load_model


class OODScore:
    def __init__(self, device, n: int = 32, seed: int = 0):
        self.encoder = GAPEncoder(load_model(OOD_ENCODER, device))
        self.bank = torch.nn.functional.normalize(torch.from_numpy(np.load(FEATURE_BANK)).float().to(device), dim=1)
        null = np.load(OOD_NULL)
        self.null_mean, self.null_std = float(null["mean"]), float(null["std"])
        self.n, self.seed = n, seed

    @torch.no_grad()
    def distance(self, xb) -> float:
        """Mean over the images in xb of the mean cosine distance to the k nearest training images."""
        d = []
        for s in range(0, len(xb), 256):
            q = torch.nn.functional.normalize(self.encoder(xb[s:s + 256]).float(), dim=1)
            d.append((1.0 - (q @ self.bank.T).topk(OOD_K, dim=1).values).clamp_min(0.0).mean(1))
        return float(torch.cat(d).mean())

    def __call__(self, explainer, x, sample_id: int) -> float:
        """z-score for one image x (1, C, H, W) and one explainer (with its groups and baseline)."""
        rng = np.random.default_rng((self.seed, sample_id))
        d = [self.distance(explainer.perturbations(x, idx, self.n, rng)) for idx in explainer.groups]
        return (float(np.mean(d)) - self.null_mean) / self.null_std
