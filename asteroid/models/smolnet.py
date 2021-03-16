import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from .base_models import BaseEncoderMaskerDecoder
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import mag, reim, apply_complex_mask, angle, from_magphase
from ..masknn import norms, activations
from einops import rearrange


class SMoLnetDilatedLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                      padding=(kernel_size // 2 * dilation, 0), dilation=(dilation, 1)),
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
        inner_channels=64,
        n_filters=2048,
        kernel_size=2048,
        stride=1024,
        sample_rate=8000,
        total_layers=13
    ):
        encoder, decoder = make_enc_dec(
            "stft",
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            sample_rate=sample_rate,
        )
        
        assert target in ("TMS", "TCS", "cIRM")
        self.target = target
        self.inner_channels = inner_channels
        
        input_channels = 2 if self.target in ("cIRM", "TCS") else 1
        
        layers = []
        prev_ch = input_channels
        
        num_dilated_layers = int(np.log2(n_filters / 2))
        num_square_layers = total_layers - num_dilated_layers
        assert num_square_layers > 0
        
        for idx in range(num_dilated_layers):
            layers.append(SMoLnetDilatedLayer(prev_ch, self.inner_channels, dilation=2**idx))
            prev_ch = self.inner_channels

        for idx in range(num_square_layers):
            layers.append(SMoLnetLateLayer(self.inner_channels, self.inner_channels))
            
        if self.target == "TMS":
            layers.append(nn.Conv2d(self.inner_channels, 1, kernel_size=1))
            layers.append(nn.Softplus())
        else:
            layers.append(nn.Conv2d(self.inner_channels, 2, kernel_size=1))
            
        masker = nn.Sequential(*layers)
        super().__init__(encoder, masker, decoder)
        
    def forward_masker(self, tf_rep):
        tf_rep = tf_rep.unsqueeze(1)
        
        if self.target == "TMS":
            input_rep = mag(tf_rep)
        else:
            input_rep = torch.cat(reim(tf_rep), dim=1)
            
        output_rep = self.masker(input_rep)
        if self.target != "TMS":
            output_rep = torch.cat(torch.chunk(output_rep, 2, dim=1), dim=-2)
        
        return output_rep.squeeze(1)
    
    def apply_masks(self, tf_rep, est_masks):
        if self.target == "TMS":
            ang = angle(tf_rep)
            return from_magphase(est_masks, ang)
        
        elif self.target == "TCS":
            return est_masks
        
        else:
            return apply_complex_mask(tf_rep, est_masks)
        
    def get_model_args(self):
        fb_config = self.encoder.filterbank.get_config()
        fb_config.pop('fb_name')
        model_args = {
            **fb_config,
            'target': self.target,
            'inner_channels': self.inner_channels,
        }
        return model_args