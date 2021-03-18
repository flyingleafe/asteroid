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

def _jitable_shape(tensor):
    """Gets shape of ``tensor`` as ``torch.Tensor`` type for jit compiler
    .. note::
        Returning ``tensor.shape`` of ``tensor.size()`` directly is not torchscript
        compatible as return type would not be supported.
    Args:
        tensor (torch.Tensor): Tensor
    Returns:
        torch.Tensor: Shape of ``tensor``
    """
    return torch.tensor(tensor.shape)

def _pad_x_to_y(x: torch.Tensor, y: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Right-pad or right-trim first argument to have same size as second argument
    Args:
        x (torch.Tensor): Tensor to be padded.
        y (torch.Tensor): Tensor to pad `x` to.
        axis (int): Axis to pad on.
    Returns:
        torch.Tensor, `x` padded to match `y`'s shape.
    """
    if axis != -1:
        raise NotImplementedError
    inp_len = y.shape[axis]
    output_len = x.shape[axis]
    return nn.functional.pad(x, [0, inp_len - output_len])

def _shape_reconstructed(reconstructed, size):
    """Reshape `reconstructed` to have same size as `size`
    Args:
        reconstructed (torch.Tensor): Reconstructed waveform
        size (torch.Tensor): Size of desired waveform
    Returns:
        torch.Tensor: Reshaped waveform
    """
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed

    
class AutoEncoder(BaseEncoderMaskerDecoder):
    # VAE encoder-latent_varitational-decoder inspired from : https://gitlab.inria.fr/smostafa/avse-vae/-/blob/master/train_VAE.py
    def __init__(
        self,
        activation="tanh",
        n_filters=256,
        kernel_size=256,
        stride=128,
        padding=3,
        hid_dim = 2048,
        z_dim = 64,
        sample_rate=8000
    ):
        self.padding = padding
        self.n_freq = n_filters // 2 + 1 
        prev_dim = self.n_freq * (self.padding * 2 + 1)
        self.hid_dim = hid_dim 
        self.z_dim = z_dim

        encoder, decoder = make_enc_dec(
            "stft",
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            sample_rate=sample_rate,
        )
        #fake masker to compy with asteroid BaseEncoderMaskerDecoder 
        masker = nn.Sequential(nn.Identity(54, unused_argument1=0.1, unused_argument2=False))
        
        self.activation = activation
        
        super().__init__(encoder, masker, decoder) 
        #real masker 
        self.enc1 = nn.Linear(prev_dim, self.hid_dim) 
        self.enc2 = nn.Linear(hid_dim, self.hid_dim) 
        self.enc3 = nn.Linear(hid_dim, self.n_freq) 
        self.latent_rep = nn.Linear(self.n_freq, self.z_dim)
        self.dec1 = nn.Linear(self.z_dim, self.hid_dim)
        self.dec2 = nn.Linear(self.hid_dim, self.n_freq)
        
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
        
        enc_out1 = torch.tanh(self.enc1(new_batch))
        enc_out2 = torch.tanh(self.enc2(enc_out1))
        enc_out3 = torch.tanh(self.enc3(enc_out2))
        z = self.latent_rep(enc_out3) 
        dec_1 = self.dec1(z)
        unrolled_masks = self.dec2(dec_1)

        masks = rearrange(unrolled_masks, '(n l) k -> n k l', n=batch_size)
        return masks 
    
    def apply_masks(self, tf_rep, est_masks):
        return apply_mag_mask(tf_rep, est_masks)

    def forward(self, wav):

        """Enc/Mask/Dec model forward
        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.
        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = _jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = _pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)

    def get_model_args(self):
        fb_config = self.encoder.filterbank.get_config()
        fb_config.pop('fb_name')
        model_args = {
            **fb_config,
            'activation': self.activation,
        }
        return model_args
