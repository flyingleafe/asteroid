import torch
import torch.nn.functional as F
from torch import nn
from .base_models import BaseEncoderMaskerDecoder
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import mag, magreim, apply_mag_mask
from ..masknn import norms, activations
from einops import rearrange


class SMoLnetDilatedLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                      padding=(kernel_size // 2 * dilation, 1), dilation=(dilation, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.main(x)

    
class SMoLnetLateLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                      padding=(kernel_size // 2, kernel_size // 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.main(x)

    
class SMoLnet(BaseEncoderMaskerDecoder):
    
    def __init__(
        self,
        target="TCS",
        n_filters=2048,
        kernel_size=2048,
        stride=1024,
        sample_rate=8000
    ):
        encoder, decoder = make_enc_dec(
            "stft",
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            sample_rate=sample_rate,
        )
        
        self.target = target
        
        
        
        masker = nn.Sequential(
            SMoLnetDilatedLayer()
        )