"""Vendored and modified Quantus 0.5.3, trimmed to the metrics used in the paper.

Changes to upstream: a multi_label option (each explained class of an image is scored
separately), a perturb_order option for IROF ("morf" / "lerf"), and SLIC superpixels for
IROF fixed to n_segments=25, compactness=0.5, slic_zero=True.
"""
from sgp.quantus.metrics.faithfulness.faithfulness_estimate import FaithfulnessEstimate
from sgp.quantus.metrics.faithfulness.irof import IROF
from sgp.quantus.metrics.localisation.auc import AUC
from sgp.quantus.metrics.localisation.relevance_rank_accuracy import RelevanceRankAccuracy
