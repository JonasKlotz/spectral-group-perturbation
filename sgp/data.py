"""BigEarthNet v2.0: 10 Sentinel-2 bands at 120 x 120 px, multi-labels and pixel-level
reference maps. Patches with clouds, shadows or seasonal snow are excluded (reBEN default)."""
import lmdb
import numpy as np
import pandas as pd
import torch
from configilm.extra._defaults import default_transform
from configilm.extra.BENv2_utils import band_combi_to_mean_std, ben_19_labels_to_multi_hot, stack_and_interpolate
from safetensors.numpy import load as safetensor_load

from sgp.config import BANDS, BENV2_ROOT, IMAGE_SIZE, NUM_CLASSES

SPLITS = {"train": "train", "val": "validation", "test": "test"}

# CORINE Land Cover code of the reference maps -> index of the 19 reBEN classes (sorted
# names); codes without one of the 19 classes are unlabeled (-1)
CLC_TO_CLASS = {
    111: 18, 112: 18, 121: 7, 211: 1, 212: 1, 213: 1, 221: 16, 222: 16, 223: 16, 231: 15,
    241: 16, 242: 5, 243: 10, 244: 0, 311: 3, 312: 6, 313: 12, 321: 14, 322: 13, 323: 13,
    324: 17, 331: 2, 333: 14, 411: 9, 412: 9, 421: 4, 422: 4, 511: 8, 512: 8, 521: 11,
    522: 11, 523: 11,
}


def _open(path):
    return lmdb.open(str(path), readonly=True, lock=False, meminit=False, readahead=True,
                     map_size=8 * 1024**3, max_spare_txns=16)


class BENv2(torch.utils.data.Dataset):
    """Item i: (image (10, 120, 120) normalised float32, multi-hot labels (19,), reference map (120, 120)).
    Patches are sorted by name, so an index identifies the same image in every run."""

    def __init__(self, split: str, root: str = BENV2_ROOT):
        meta = pd.read_parquet(f"{root}/metadata.parquet")
        meta = meta[~meta["contains_cloud_or_shadow"] & ~meta["contains_seasonal_snow"] & (meta["split"] == SPLITS[split])]
        self.patches = sorted(meta["patch_id"])
        self.labels = dict(zip(meta["patch_id"], meta["labels"]))
        self.root = root
        mean, std = band_combi_to_mean_std(len(BANDS))
        self.transform = default_transform(img_size=(IMAGE_SIZE, IMAGE_SIZE), mean=mean, std=std)
        self._images = self._maps = None  # opened lazily (one environment per process)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        if self._images is None:
            self._images = _open(f"{self.root}/BENv2.lmdb")
            self._maps = _open(f"{self.root}/BENv2_Reference_Maps.lmdb")
        key = self.patches[i]
        with self._images.begin(buffers=True) as txn:
            bands = safetensor_load(bytes(txn.get(key.encode())))
        image = self.transform(stack_and_interpolate({b: bands[b] for b in BANDS}, order=BANDS, img_size=IMAGE_SIZE,
                                                     upsample_mode="nearest"))
        labels = ben_19_labels_to_multi_hot(self.labels[key])
        with self._maps.begin(buffers=True) as txn:
            clc = safetensor_load(bytes(txn.get(f"{key}_reference_map".encode())))["Data"]
        ref_map = np.vectorize(lambda code: CLC_TO_CLASS.get(int(code), -1))(clc)
        return image, labels, ref_map

    def batch(self, indices):
        """x (B, 10, H, W), y (B, 19) and the reference maps one-hot per class s (B, 19, 1, H, W), all numpy."""
        items = [self[int(i)] for i in indices]
        x = torch.stack([it[0] for it in items]).numpy()
        y = torch.stack([torch.as_tensor(it[1]) for it in items]).numpy()
        ref = np.stack([it[2] for it in items])
        s = (ref[:, None, None] == np.arange(NUM_CLASSES)[None, :, None, None, None]).astype(np.float32)
        return x, y, s


def load_pairs(path: str) -> np.ndarray:
    """(N, 2) int array of (sample index, class)."""
    return np.load(path).astype(np.int64)
