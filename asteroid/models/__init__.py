# Models
from .base_models import BaseModel
from .conv_tasnet import ConvTasNet
from .dccrnet import DCCRNet
from .dcunet import DCUNet
from .dprnn_tasnet import DPRNNTasNet
from .sudormrf import SuDORMRFImprovedNet, SuDORMRFNet
from .dptnet import DPTNet
from .lstm_tasnet import LSTMTasNet
from .demask import DeMask
from .wave_unet import WaveUNet
from .demucs import Demucs
from .smolnet import SMoLnet
from .fcnn_regression import RegressionFCNN
from .vae import VAE
from .phasen import Phasen
from .auto_encoder import AutoEncoder
from .segan import SEGAN
from .metric_gan import MetricGAN

# Sharing-related
from .publisher import save_publishable, upload_publishable

__all__ = [
    "ConvTasNet",
    "DPRNNTasNet",
    "SuDORMRFImprovedNet",
    "SuDORMRFNet",
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
    "SEGAN",
    "MetricGAN",
    "save_publishable",
    "upload_publishable",
]


def register_model(custom_model):
    """Register a custom model, gettable with `models.get`.

    Args:
        custom_model: Custom model to register.

    """
    if (
        custom_model.__name__ in globals().keys()
        or custom_model.__name__.lower() in globals().keys()
    ):
        raise ValueError(f"Model {custom_model.__name__} already exists. Choose another name.")
    globals().update({custom_model.__name__: custom_model})


def get(identifier):
    """Returns an model class from a string (case-insensitive).

    Args:
        identifier (str): the model name.

    Returns:
        :class:`torch.nn.Module`
    """
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret model name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret model name : {str(identifier)}")
