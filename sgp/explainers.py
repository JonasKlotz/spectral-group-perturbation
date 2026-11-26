"""Occlusion and HSIC attribution with the spectral group perturbation (SGP).

Both explainers take the band groups C_1..C_n (n = 1: joint perturbation, JP) and return
one map per group, shape (n, H, W). Group i is perturbed with tau(x, theta, C_i) (Eq. 3):
only its bands are replaced inside the region, all other bands keep their values.
Both explain the logit of the target class.

Baselines (the same value for every band and every group):
  black  minimum of the image over all bands and pixels
  gmean  mean of the image over all bands and pixels ("Global mean")

aggregate() turns the group maps into the final map (Eq. 4). perturbations() returns
perturbed copies of an image as the explainer builds them, for OOD z (sgp.ood).
"""
import numpy as np
import scipy.stats
import torch
from captum.attr import Occlusion as CaptumOcclusion

from sgp.quantus.functions.perturb_func import baseline_replacement_by_indices


def aggregate(maps: torch.Tensor, groups) -> torch.Tensor:
    """Group maps (n, H, W) -> one map (H, W): every group map is rectified and counted once
    per band of its group, i.e. sum_i |C_i| * max(0, e_i). For JP this only scales the map,
    which none of the metrics (rank or correlation based) depends on."""
    weights = torch.tensor([len(g) for g in groups], dtype=maps.dtype, device=maps.device)
    return (torch.relu(maps) * weights[:, None, None]).sum(0)


class Occlusion:
    """Occlusion (Zeiler & Fergus 2014) via captum: a window slides over the image and the
    drop of the logit is assigned to the window. One captum run per band group, window and
    stride in pixels."""

    def __init__(self, model, groups, baseline: str, window: int = 50, stride: int = 10):
        self.model, self.groups, self.baseline = model, [list(g) for g in groups], baseline
        self.window, self.stride = window, stride

    def fill_value(self, x):
        return {"black": x.min(), "gmean": x.mean()}[self.baseline]

    def __call__(self, x, target: int) -> torch.Tensor:
        maps = []
        for idx in self.groups:
            def forward(sub, idx=idx):  # insert the (perturbed) group bands into the unperturbed image
                full = x.expand(sub.shape[0], -1, -1, -1).clone()
                full[:, idx] = sub
                return self.model(full)

            attrs = CaptumOcclusion(forward).attribute(
                x[:, idx], target=target, baselines=self.fill_value(x), perturbations_per_eval=64,
                sliding_window_shapes=(len(idx), self.window, self.window),
                strides=(len(idx), self.stride, self.stride))
            maps.append(attrs[0, 0])  # the window covers all bands of the group: same map in each
        return torch.stack(maps)

    def perturbations(self, x, idx, n, rng):
        H, W = x.shape[-2:]
        positions = [(r, c) for r in range(0, H - self.window + 1, self.stride)
                     for c in range(0, W - self.window + 1, self.stride)]
        pick = rng.choice(len(positions), min(n, len(positions)), replace=False)
        xb = x.repeat(len(pick), 1, 1, 1)
        for b, p in enumerate(pick):
            r, c = positions[p]
            xb[b, idx, r:r + self.window, c:c + self.window] = self.fill_value(x)
        return xb


class HSIC:
    """HSIC attribution (Novello et al. 2022): n_masks random binary masks on a grid x grid
    grid (Latin hypercube, a cell is perturbed with probability mask_ratio); the relevance of
    a cell is the HSIC between its mask values and the logit.

    JP (one group): HSIC as published. SGP: the RBF bandwidth of the logits is shared by
    all groups, and a cell is set to 0 in a group map if the mean logit is higher when the
    cell is perturbed than when it is not (direction gate, rectification of Section II-B)."""

    def __init__(self, model, groups, baseline: str, n_masks: int = 1024, grid: int = 7,
                 mask_ratio: float = 0.3, seed: int = 42, batch_size: int = 128):
        self.model, self.groups, self.baseline = model, [list(g) for g in groups], baseline
        self.n_masks, self.grid, self.mask_ratio, self.seed, self.batch_size = n_masks, grid, mask_ratio, seed, batch_size
        self._masks = None

    def cells(self, H, W) -> np.ndarray:
        """(H, W) grid cell index of every pixel."""
        rows, cols = np.arange(H) * self.grid // H, np.arange(W) * self.grid // W
        return np.arange(self.grid ** 2).reshape(self.grid, self.grid)[rows[:, None], cols[None, :]]

    def masks(self) -> np.ndarray:
        """(n_masks, cells) in {0, 1}, 0 = perturbed; the same masks for every image."""
        if self._masks is None:
            lhs = scipy.stats.qmc.LatinHypercube(d=self.grid ** 2, seed=self.seed).random(n=self.n_masks)
            self._masks = (lhs.astype(np.float32) >= self.mask_ratio).astype(np.float32)
        return self._masks

    def baseline_image(self, x) -> torch.Tensor:
        img = x[0].detach().cpu().numpy().astype(np.float32)
        if self.baseline == "gmean":
            base = np.full_like(img, float(img.mean()))
        else:  # black: quantus fills with the image minimum
            base = baseline_replacement_by_indices(arr=img, indices=np.arange(img[0].size), indexed_axes=(1, 2),
                                                   perturb_baseline=self.baseline)
        return torch.from_numpy(base.astype(np.float32)).unsqueeze(0).to(x.device, dtype=x.dtype)

    def _perturb(self, x, base, masks, cells, idx):
        m = torch.from_numpy(masks[:, cells]).float().to(x.device).unsqueeze(1)  # (B, 1, H, W)
        xb = x.expand(len(masks), -1, -1, -1).clone()
        xb[:, idx] = x[:, idx] * m + base[:, idx] * (1.0 - m)
        return xb

    @torch.no_grad()
    def __call__(self, x, target: int) -> torch.Tensor:
        H, W = x.shape[-2:]
        cells, masks, base = self.cells(H, W), self.masks(), self.baseline_image(x)
        logits = []
        for idx in self.groups:
            out = [self.model(self._perturb(x, base, masks[s:s + self.batch_size], cells, idx))[:, target].cpu()
                   for s in range(0, len(masks), self.batch_size)]
            logits.append(torch.cat(out).to(x.device))

        masks_t = torch.from_numpy(masks).to(x.device)
        K = _kernel_dirac(masks_t)
        sigma = _bandwidth(torch.cat(logits))  # JP: the bandwidth of its own logits
        cells_t = torch.from_numpy(cells).long().to(x.device)
        maps = []
        for y in logits:
            scores = _hsic(K, _kernel_rbf(y, sigma)).clamp(min=0.0)
            if len(self.groups) > 1:  # SGP: direction gate
                scores = scores * (_direction(masks_t, y) > 0).to(scores.dtype)
            maps.append(scores[cells_t])
        return torch.stack(maps)

    def perturbations(self, x, idx, n, rng):
        masks = self.masks()
        pick = masks[rng.choice(len(masks), min(n, len(masks)), replace=False)]
        return self._perturb(x, self.baseline_image(x), pick, self.cells(*x.shape[-2:]), idx)


def _kernel_dirac(masks):
    """(cells, N, N): 1 where two masks agree on the cell, 0 otherwise."""
    X = masks.T
    return 1.0 - (X.unsqueeze(2) - X.unsqueeze(1)) ** 2


def _bandwidth(y):
    d = (y.unsqueeze(1) - y.unsqueeze(0)) ** 2
    return torch.sqrt(torch.median(d[d > 0]) / 2.0 + 1e-08)


def _kernel_rbf(y, sigma):
    return torch.exp(-((y.unsqueeze(1) - y.unsqueeze(0)) ** 2) / (2.0 * sigma ** 2))


def _hsic(K, L):
    """HSIC estimate per cell, tr(K_c L_c) / N with centred kernels."""
    n = K.shape[1]
    Hc = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    Kc = torch.einsum("ijk,kl->ijl", torch.einsum("jk,ikl->ijl", Hc, K), Hc)
    Lc = Hc @ L @ Hc
    return torch.einsum("ijk,jk->i", Kc, Lc) / n


def _direction(masks, y):
    """Mean logit with the cell present minus with the cell perturbed, per cell."""
    y = y.unsqueeze(1)
    present = (masks * y).sum(0) / masks.sum(0).clamp(min=1.0)
    absent = ((1.0 - masks) * y).sum(0) / (1.0 - masks).sum(0).clamp(min=1.0)
    return present - absent
