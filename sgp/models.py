"""The reBEN classifiers (ConvNeXtV2-Base, MobileViT-S) and the GAP feature encoder for OOD z."""
import torch
from configilm.ConfigILM import ConfigILM, ILMConfiguration
from huggingface_hub import PyTorchModelHubMixin

from sgp.config import MODELS


class BENv2Classifier(torch.nn.Module, PyTorchModelHubMixin):
    """Inference-only version of the reBEN BigEarthNetv2_0_ImageClassifier: same constructor
    and parameter names, so the published Hugging Face checkpoints load unchanged.
    The output is one logit per class."""

    def __init__(self, config: ILMConfiguration, lr: float = 1e-3, warmup=None):
        super().__init__()
        self.config = config
        self.model = ConfigILM(config)

    def forward(self, x):
        return self.model(x)


def load_model(name: str, device) -> torch.nn.Module:
    """name: a key of config.MODELS."""
    return BENv2Classifier.from_pretrained(MODELS[name]).to(device).eval()


class GAPEncoder:
    """Global average pooled output of the last ConvNeXtV2 block, (B, 1024)."""

    def __init__(self, model):
        self.model, self._out = model, None
        model.model.vision_encoder.stages[3].blocks[-1].register_forward_hook(
            lambda module, inputs, output: setattr(self, "_out", output.mean(dim=(-2, -1))))

    @torch.no_grad()
    def __call__(self, x):
        self.model(x)
        return self._out
