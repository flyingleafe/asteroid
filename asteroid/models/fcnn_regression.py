import torch
import torch.nn.functional as F
from torch import nn
from .base_models import BaseEncoderMaskerDecoder
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import mag, magreim, apply_mag_mask
from ..masknn import norms, activations
from einops import rearrange
from tqdm import tqdm

def _unsqueeze_to_3d(x):
    """Normalize shape of `x` to [batch, n_chan, time]."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x

    
class RegressionFCNN(BaseEncoderMaskerDecoder):
    
    def __init__(
        self,
        activation="relu",
        hidden_layers=(2048, 2048, 2048),
        padding=3,
        dropout=0.2,
        n_filters=256,
        kernel_size=256,
        stride=128,
        sample_rate=8000
    ):
        encoder, decoder = make_enc_dec(
            "stft",
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            sample_rate=sample_rate,
        )
        
        self.activation = activation
        self.padding = padding
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.n_freq = n_filters // 2 + 1
        
        prev_dim = self.n_freq * (self.padding * 2 + 1)
        layers = []
        
        for n_hid in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, n_hid))
            layers.append(activations.get(activation)())
            if dropout != 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = n_hid
    
        layers.append(nn.Linear(prev_dim, self.n_freq))
        masker = nn.Sequential(*layers)
        super().__init__(encoder, masker, decoder)
        
        self.register_buffer('scaler_mean', torch.zeros(self.n_freq))
        self.register_buffer('scaler_std', torch.zeros(self.n_freq))
        self.has_scaler = False
        
    def compute_scaler(self, data_iter):
        count = 0
        total_sum = torch.zeros(self.n_freq)
        total_sum_2 = torch.zeros(self.n_freq)
        
        for batch in tqdm(data_iter, 'Computing scaler'):
            mix, _ = batch
            mix = _unsqueeze_to_3d(mix)
            tf_rep = self.forward_encoder(mix)
            log_mag = torch.log(mag(tf_rep))
            
            total_sum   += torch.sum(log_mag, dim=(0, 2))
            total_sum_2 += torch.sum(log_mag.pow(2), dim=(0, 2))
            count       += log_mag.shape[0] * log_mag.shape[2]
        
        mean = total_sum / count
        variance = (total_sum_2 / count - mean.pow(2)) * (count / (count - 1))
        std = torch.sqrt(variance)

        self.scaler_mean = mean
        self.scaler_std = std
        self.has_scaler = True
        
    def forward_masker(self, tf_rep):
        batch_size = tf_rep.shape[0]
        log_mag = torch.log(mag(tf_rep)).unsqueeze(1)
                
        if self.has_scaler:
            l = log_mag.shape[-1]
            mean = self.scaler_mean.view(-1, 1).expand(-1, l)
            std = self.scaler_std.view(-1, 1).expand(-1, l)
            log_mag -= mean
            log_mag /= std
 
        padded = F.pad(log_mag, (self.padding, self.padding, 0, 0), mode='replicate')
        stacks = F.unfold(padded, (self.n_freq, self.padding * 2 + 1))
            
        new_batch = rearrange(stacks, 'n k l -> (n l) k')
        unrolled_masks = self.masker(new_batch)
        masks = rearrange(unrolled_masks, '(n l) k -> n k l', n=batch_size)
        return masks
    
    def apply_masks(self, tf_rep, est_masks):
        return apply_mag_mask(tf_rep, est_masks)
    
    def get_model_args(self):
        fb_config = self.encoder.filterbank.get_config()
        fb_config.pop('fb_name')
        model_args = {
            **fb_config,
            'activation': self.activation,
            'padding': self.padding,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
        }
        return model_args