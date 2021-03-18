import pathlib

from .models import ConvTasNet, DCCRNet, DCUNet, DPRNNTasNet, DPTNet, LSTMTasNet, DeMask, WaveUNet, Demucs, SMoLnet, RegressionFCNN, Phasen, VAE, AutoEncoder
from .utils import deprecation_utils, torch_utils  # noqa

project_root = str(pathlib.Path(__file__).expanduser().absolute().parent.parent)
__version__ = "0.4.1"


def show_available_models():
    from .utils.hub_utils import MODELS_URLS_HASHTABLE

    print(" \n".join(list(MODELS_URLS_HASHTABLE.keys())))


def available_models():
    from .utils.hub_utils import MODELS_URLS_HASHTABLE

    return MODELS_URLS_HASHTABLE


__all__ = [
    "ConvTasNet",
    "DPRNNTasNet",
    "DPTNet",
    "LSTMTasNet",
    "DeMask",
    "DCUNet",
    "DCCRNet",
    "WaveUNet",
    "Demucs",
    "SMoLnet",
    "RegressionFCNN",
    "VAE",
    "AutoEncoder",
    "Phasen",
    "show_available_models",
]
