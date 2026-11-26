"""Band groups (Eq. 2): average-linkage agglomerative clustering of the bands with the
distance d(c, c') = 1 - r(c, c'), r = Pearson correlation of the bands over training
pixels (prepare.py), cut at n clusters. n = 1 is the joint perturbation (JP), n = 10
the single band perturbation.

With the stored correlation this gives (band names in sgp.config.BANDS):
  n = 3: B02-B05 | B06-B8A | B11 B12                  (C1, C2, C3 of the paper)
  n = 4: B02-B04 | B05 | B06-B8A | B11 B12
  n = 5: B02 B03 | B04 | B05 | B06-B8A | B11 B12
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from sgp.config import BAND_CORRELATION, BANDS


def band_groups(n: int, correlation: np.ndarray | None = None) -> list[list[int]]:
    """n disjoint groups of band indices, each sorted, ordered by their first band."""
    C = len(BANDS)
    if n == 1:
        return [list(range(C))]
    if n == C:
        return [[c] for c in range(C)]
    r = np.load(BAND_CORRELATION) if correlation is None else correlation
    d = 1.0 - r
    np.fill_diagonal(d, 0.0)
    labels = fcluster(linkage(squareform(d, checks=False), method="average"), n, criterion="maxclust")
    groups = sorted([sorted(np.flatnonzero(labels == k).tolist()) for k in np.unique(labels)])
    if len(groups) != n:
        raise ValueError(f"clustering gave {len(groups)} groups instead of {n}")
    return groups


def describe(groups) -> str:
    return " | ".join(" ".join(BANDS[c] for c in g) for g in groups)
