"""Paths and constants shared by all experiments.

Every input artifact (predictions, image-class pairs, feature bank, OOD null baseline,
band correlation) is written by prepare.py; the experiments write to RESULTS.
"""
import os

OUTPUTS = os.environ.get("SGP_OUTPUTS", "/data/jonas/outputs/rs_perturbation")
RESULTS = f"{OUTPUTS}/paper"
BENV2_ROOT = os.environ.get("BENV2_BASE_PATH", "/data_read_only/BigEarthNet/BigEarthNet-V2")

# reBEN weights from Hugging Face; the predictions and pairs of each model live in its own folder
MODELS = {
    "convnextv2": "BIFOLD-BigEarthNetv2-0/convnextv2_base-s2-v0.2.0",
    "mobilevit": "BIFOLD-BigEarthNetv2-0/mobilevit_s-s2-v0.2.0",
}
MODEL_DIRS = {"convnextv2": OUTPUTS, "mobilevit": f"{OUTPUTS}/mobilevit_s"}

NUM_CLASSES = 19
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]  # 10 m + 20 m, input order
IMAGE_SIZE = 120

# OOD z: GAP features of a fixed encoder (ConvNeXtV2) for every explained model
OOD_ENCODER = "convnextv2"
OOD_K = 25
FEATURE_BANK = f"{OUTPUTS}/features/benv2/train/features.npy"
OOD_NULL = f"{OUTPUTS}/split_information/benv2/test/ood_null_baseline_vs_train_k{OOD_K}.npz"
BAND_CORRELATION = f"{OUTPUTS}/band_correlation/benv2/train/correlation.npy"


def split_dir(model: str, split: str) -> str:
    return f"{MODEL_DIRS[model]}/split_information/benv2/{split}"


def pairs_path(model: str, split: str, n: int) -> str:
    """(sample index, class) pairs drawn uniformly from the correctly predicted pairs."""
    return f"{split_dir(model, split)}/random_pairs_{n}.npy"
