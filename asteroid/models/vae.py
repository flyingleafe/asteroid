import torch
import torch.nn.functional as F
from torch import nn
from .base_models import BaseEncoderMaskerDecoder
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import mag, magreim, apply_mag_mask, angle, from_magphase
from ..masknn import norms, activations
from einops import rearrange
from tqdm import tqdm


class VAE_inner(nn.Module):
    
    
    def __init__(self, input_dim=513, latent_dim=64,
                 hidden_dim_encoder=128, activation='tanh'):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim_encoder = hidden_dim_encoder 
        self.activation = activations.get(activation)() # activation for audio layers
        
        self.decoder_layerZ = nn.Linear(self.latent_dim, self.hidden_dim_encoder)    
        self.encoder_layerX = nn.Linear(self.input_dim, self.hidden_dim_encoder)
        
        self.output_layer = nn.Linear(hidden_dim_encoder, self.input_dim) 
    
        
        #### Define bottleneck layer ####  
        
        self.latent_mean_layer = nn.Linear(self.hidden_dim_encoder, self.latent_dim)
        self.latent_logvar_layer = nn.Linear(self.hidden_dim_encoder, self.latent_dim)
        
                
    def encode(self, x):
        xv = self.encoder_layerX(x)
        he = self.activation(xv)
        return self.latent_mean_layer(he), self.latent_logvar_layer(he)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        zv = self.decoder_layerZ(z)
        hd = self.activation(zv)    
        return torch.exp(self.output_layer(hd))

    def forward(self, x):     
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE(BaseEncoderMaskerDecoder):
    def __init__(
        self,
        activation='tanh',
        latent_dim=64,
        hidden_dim_encoder=128,
        n_filters=1024,
        kernel_size=1024,
        stride=256,
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
        self.latent_dim = latent_dim
        self.hidden_dim_encoder = hidden_dim_encoder
        self.input_dim = n_filters // 2 + 1
        
        masker = VAE_inner(self.input_dim, self.latent_dim, self.hidden_dim_encoder, self.activation)
        super().__init__(encoder, masker, decoder)
            
    def forward_vae_mu_logvar(self, tf_rep_pow):
        tf_rep_pow = rearrange(tf_rep_pow, 'n k l -> n l k')
        output, mu, logvar = self.masker(tf_rep_pow)
        return rearrange(output, 'n l k -> n k l'), rearrange(mu, 'n l k -> n k l'), rearrange(logvar, 'n l k -> n k l')
    
    def forward_masker(self, tf_rep):
        output, _, _ = self.forward_vae_mu_logvar(torch.pow(mag(tf_rep), 2))
        return from_magphase(torch.sqrt(output), angle(tf_rep))
    
    def apply_masks(self, tf_rep, est_masks):
        # It's not a masker, so we just output the output
        return est_masks
        
    def get_model_args(self):
        fb_config = self.encoder.filterbank.get_config()
        fb_config.pop('fb_name')
        model_args = {
            **fb_config,
            'activation': self.activation,
            'latent_dim': self.latent_dim,
            'hidden_dim_encoder': self.hidden_dim_encoder,
        }
        return model_args
    
# class VAE(BaseEncoderMaskerDecoder):
#     # VAE encoder-latent_varitational-decoder inspired from : https://gitlab.inria.fr/smostafa/avse-vae/-/blob/master/train_VAE.py
#     def __init__(
#         self,
#         input_dim=258, 
#         latent_dim=32,
#         hidden_dim_encoder=[2048],
#         activation="tanh",
#         n_filters=256,
#         kernel_size=256,
#         stride=128,
#         sample_rate=8000
#     ):
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         self.hidden_dim_encoder = hidden_dim_encoder 
#         self.activation = activation # activation for audio layers
        
#         stft, istft = make_enc_dec(
#                         "stft",
#                         n_filters=n_filters,
#                         kernel_size=kernel_size,
#                         stride=stride,
#                         sample_rate=sample_rate,
#                 )
#         #fake masker to comply with asteroid BaseEncoderMaskerDecoder 
#         masker = nn.Sequential(nn.Identity(1, unused_argument1=0.1, unused_argument2=False))
#         super().__init__(stft, masker, istft) 

#         self.decoder_layerZ = nn.Linear(self.latent_dim, self.hidden_dim_encoder[0])
#         self.encoder_layerX = nn.Linear(self.input_dim, self.hidden_dim_encoder[0])
#         self.output_layer = nn.Linear(hidden_dim_encoder[0], self.input_dim) 
    
#         self.latent_mean_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
#         self.latent_logvar_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        
#         self.register_buffer('scaler_mean', torch.zeros(self.input_dim))
#         self.register_buffer('scaler_std', torch.zeros(self.input_dim))
#         self.has_scaler = False
        
#     def compute_scaler(self, data_iter):
#         count = 0
#         total_sum = torch.zeros(self.input_dim)
#         total_sum_2 = torch.zeros(self.input_dim)
        
#         for batch in tqdm(data_iter, 'Computing scaler'):
#             mix, _ = batch
#             mix = _unsqueeze_to_3d(mix)
#             tf_rep = self.forward_encoder(mix)
#             tf_rep = torch.abs(tf_rep)**2
            
#             total_sum   += torch.sum(tf_rep, dim=(0, 2))
#             total_sum_2 += torch.sum(tf_rep.pow(2), dim=(0, 2))
#             count       +=tf_rep.shape[0] *tf_rep.shape[2]

#         mean = total_sum / count
#         variance = (total_sum_2 / count - mean.pow(2)) * (count / (count - 1))
#         std = torch.sqrt(variance)

#         self.scaler_mean = mean
#         self.scaler_std = std
#         self.has_scaler = True
    
#     def encode(self, tfrep):
#         enc_out = self.encoder_layerX(tfrep)
#         enc_out = torch.tanh(enc_out)
#         return self.latent_mean_layer(enc_out), self.latent_logvar_layer(enc_out)
        
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std) #eps is normally distributed
#         return eps.mul(std).add_(mu)

#     def decode(self, z):
#         out = self.decoder_layerZ(z)
#         out =torch.tanh(out)
#         return torch.exp(self.output_layer(out))
    
#         return istft(pred_rep)
            
#     def forward_masker(self, tf_rep):

#         # power spec
#         tf_rep = torch.abs(tf_rep)**2

#         if self.has_scaler:
#             l = tf_rep.shape[-1]
#             mean = self.scaler_mean.view(-1, 1).expand(-1, l)
#             std = self.scaler_std.view(-1, 1).expand(-1, l)
#             tf_rep -= mean
#             tf_rep /= std
        
#         tf_rep = tf_rep.permute(0, 2, 1)
#         mu, logvar = self.encode(tf_rep) 
#         z = self.reparameterize(mu, logvar)

#         pred_rep = self.decode(z)
#         return pred_rep, mu, logvar 
    
#     def apply_masks(self, tf_rep, est_masks):
#         return apply_mag_mask(tf_rep, est_masks)

#     def forward(self, wav):

#         """Enc/Mask/Dec model forward
#         Args:
#             wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.
#         Returns:
#             torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
#         """
#         # Remember shape to shape reconstruction, cast to Tensor for torchscript
#         shape = _jitable_shape(wav)
#         # Reshape to (batch, n_mix, time)
#         wav = _unsqueeze_to_3d(wav)
#         #import pdb; pdb.set_trace()

#         # Real forward
#         tf_rep = self.forward_encoder(wav)
#         pred_rep, mu, logvar = self.forward_masker(tf_rep)
#         pred_wav = self.forward_decoder(pred_rep.permute(0, 2, 1))
#         reconstructed = _pad_x_to_y(pred_wav, wav)
#         return _shape_reconstructed(reconstructed, shape)

#     def get_model_args(self):
#         fb_config = self.encoder.filterbank.get_config()
#         fb_config.pop('fb_name')
#         model_args = {
#             **fb_config,
#             'activation': self.activation,
#         }
#         return model_args
