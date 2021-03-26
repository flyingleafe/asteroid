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

    
class VAE(BaseEncoderMaskerDecoder):
    # VAE encoder-latent_varitational-decoder inspired from : https://gitlab.inria.fr/smostafa/avse-vae/-/blob/master/train_VAE.py
    def __init__(
        self,
        input_dim=258, 
        latent_dim=32,
        hidden_dim_encoder=[2048],
        activation="tanh",
        n_filters=256,
        kernel_size=256,
        stride=128,
        sample_rate=8000
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim_encoder = hidden_dim_encoder 
        self.activation = activation # activation for audio layers
        
        stft, istft = make_enc_dec(
                        "stft",
                        n_filters=n_filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        sample_rate=sample_rate,
                )
        #fake masker to comply with asteroid BaseEncoderMaskerDecoder 
        masker = nn.Sequential(nn.Identity(1, unused_argument1=0.1, unused_argument2=False))
        super().__init__(stft, masker, istft) 

        self.decoder_layerZ = nn.Linear(self.latent_dim, self.hidden_dim_encoder[0])
        self.encoder_layerX = nn.Linear(self.input_dim, self.hidden_dim_encoder[0])
        self.output_layer = nn.Linear(hidden_dim_encoder[0], self.input_dim) 
    
        self.latent_mean_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        
        self.register_buffer('scaler_mean', torch.zeros(self.input_dim))
        self.register_buffer('scaler_std', torch.zeros(self.input_dim))
        self.has_scaler = False
        
    def compute_scaler(self, data_iter):
        count = 0
        total_sum = torch.zeros(self.input_dim)
        total_sum_2 = torch.zeros(self.input_dim)
        
        for batch in tqdm(data_iter, 'Computing scaler'):
            mix, _ = batch
            mix = _unsqueeze_to_3d(mix)
            tf_rep = self.forward_encoder(mix)
            tf_rep = torch.abs(tf_rep)**2
            
            total_sum   += torch.sum(tf_rep, dim=(0, 2))
            total_sum_2 += torch.sum(tf_rep.pow(2), dim=(0, 2))
            count       +=tf_rep.shape[0] *tf_rep.shape[2]

        mean = total_sum / count
        variance = (total_sum_2 / count - mean.pow(2)) * (count / (count - 1))
        std = torch.sqrt(variance)

        self.scaler_mean = mean
        self.scaler_std = std
        self.has_scaler = True
    
    def encode(self, tfrep):
        enc_out = self.encoder_layerX(tfrep)
        enc_out = torch.tanh(enc_out)
        return self.latent_mean_layer(enc_out), self.latent_logvar_layer(enc_out)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) #eps is normally distributed
        return eps.mul(std).add_(mu)

    def decode(self, z):
        out = self.decoder_layerZ(z)
        out =torch.tanh(out)
        return torch.exp(self.output_layer(out))
    
        return istft(pred_rep)
            
    def forward_masker(self, tf_rep):

        # power spec
        tf_rep = torch.abs(tf_rep)**2

        if self.has_scaler:
            l = tf_rep.shape[-1]
            mean = self.scaler_mean.view(-1, 1).expand(-1, l)
            std = self.scaler_std.view(-1, 1).expand(-1, l)
            tf_rep -= mean
            tf_rep /= std
        
        tf_rep = tf_rep.permute(0, 2, 1)
        mu, logvar = self.encode(tf_rep) 
        z = self.reparameterize(mu, logvar)

        pred_rep = self.decode(z)
        return pred_rep, mu, logvar 
    
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
        #import pdb; pdb.set_trace()

        # Real forward
        tf_rep = self.forward_encoder(wav)
        pred_rep, mu, logvar = self.forward_masker(tf_rep)
        pred_wav = self.forward_decoder(pred_rep.permute(0, 2, 1))
        reconstructed = _pad_x_to_y(pred_wav, wav)
        return _shape_reconstructed(reconstructed, shape)

    def get_model_args(self):
        fb_config = self.encoder.filterbank.get_config()
        fb_config.pop('fb_name')
        model_args = {
            **fb_config,
            'activation': self.activation,
        }
        return model_args
