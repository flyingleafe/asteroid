import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from .base_models import BaseEncoderMaskerDecoder
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import mag, magreim, apply_mag_mask, angle, from_magphase, to_numpy, from_numpy, to_torch_complex, from_torch_complex
from ..masknn import norms, activations
from einops import rearrange
from tqdm import tqdm


class MCEM_algo:
    def __init__(self, X=None, W=None, H=None, Z=None, decoder=None,
                 niter_MCEM=100, niter_MH=40, burnin=30, var_MH=0.01):
        self.X = X # Mixture STFT, shape (F,N)
        self.W = W # NMF dictionary matrix, shape (F, K)
        self.H = H # NMF activation matrix, shape (K, N)
        self.V = self.W @ self.H # Noise variance, shape (F, N)
        self.Z = Z # Last draw of the latent variables, shape (D, N)
        self.decoder = decoder # VAE decoder, keras model
        self.niter_MCEM = niter_MCEM # Maximum number of MCEM iterations
        self.niter_MH = niter_MH # Number of iterations for the MH algorithm
        # of the E-step
        self.burnin = burnin # Burn-in period for the MH algorithm of the
        # E-step
        self.var_MH = var_MH # Variance of the proposal distribution of the MH
        # algorithm
        # output of the decoder with self.Z as input, shape (F, N)
        self.Z_mapped_decoder = self.decoder(self.Z.T).T
        self.a = torch.ones((1,self.X.shape[1])).type_as(self.Z_mapped_decoder) # gain parameters, shape (1,N)
        self.speech_var = self.Z_mapped_decoder*self.a # apply gain

    def num2torch(self, x):
        y = torch.from_numpy(x.astype(np.float32))
        return y
    def torch2num(self, x):
        y = x.detach().numpy()
        return y
    
    def metropolis_hastings(self, niter_MH=None, burnin=None):
        if niter_MH==None:
            niter_MH = self.niter_MH

        if burnin==None:
            burnin = self.burnin

        F, N = self.X.shape
        D = self.Z.shape[0]

        Z_sampled = torch.zeros((D, N, niter_MH - burnin), device=self.Z.device)

        cpt = 0
        for n in np.arange(niter_MH):

            Z_prime = self.Z + np.sqrt(self.var_MH)*torch.randn(D,N, device=self.Z.device)

            Z_prime_mapped_decoder = self.decoder(Z_prime.T).T
            # shape (F, N)
            speech_var_prime = Z_prime_mapped_decoder*self.a # apply gain

            acc_prob = ( torch.sum( torch.log(self.V + self.speech_var)
                                 - torch.log(self.V + speech_var_prime) 
                                 + ( 1/(self.V + self.speech_var) 
                                    - 1/(self.V + speech_var_prime) )
                                 * torch.abs(self.X)**2, axis=0) 
                        + .5*torch.sum( self.Z**2 - Z_prime**2 , axis=0) )

            is_acc = torch.log(torch.rand(1,N, device=acc_prob.device)) < acc_prob
            is_acc = is_acc.reshape((is_acc.shape[1],))

            self.Z[:,is_acc] = Z_prime[:,is_acc]
            self.Z_mapped_decoder = self.decoder(self.Z.T).T
            self.speech_var = self.Z_mapped_decoder*self.a

            if n > burnin - 1:
                Z_sampled[:,:,cpt] = self.Z
                cpt += 1

        return Z_sampled

    def run(self, hop, wlen, win, tol=1e-4):

        F, N = self.X.shape

        X_abs_2 = torch.abs(self.X)**2

        cost_after_M_step = torch.zeros((self.niter_MCEM, 1), device=X_abs_2.device)

        for n in np.arange(self.niter_MCEM):

            # MC-Step
            # print('Metropolis-Hastings')
            Z_sampled = self.metropolis_hastings(self.niter_MH, self.burnin)
            Z_sampled_mapped_decoder = torch.zeros((F, N, self.niter_MH-self.burnin), device=Z_sampled.device)
            
            for i in range(self.niter_MH-self.burnin):
                Z_sampled_mapped_decoder[:,:,i] = self.decoder(Z_sampled[:,:,i].T).T
                    
            speech_var_multi_samples = (Z_sampled_mapped_decoder*
                                        self.a[:,:,None]) # shape (F,N,R)

            # M-Step
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            # print('Update W')
            self.W = self.W*(
                    ((X_abs_2*torch.sum(V_plus_Z_mapped**-2,
                                     axis=-1)) @ self.H.T)
                    / (torch.sum(V_plus_Z_mapped**-1, axis=-1) @ self.H.T)
                    )**.5
            self.V = self.W @ self.H
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            # print('Update H')
            self.H = self.H*(
                    (self.W.T @ (X_abs_2 * torch.sum(V_plus_Z_mapped**-2,
                                                  axis=-1)))
                    / (self.W.T @ torch.sum(V_plus_Z_mapped**-1, axis=-1))
                    )**.5
            self.V = self.W @ self.H
            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            # print('Update gain')
            self.a = self.a*(
                    (torch.sum(X_abs_2 * torch.sum(
                            Z_sampled_mapped_decoder*(V_plus_Z_mapped**-2),
                            axis=-1), axis=0) )
                    /(torch.sum(torch.sum(
                            Z_sampled_mapped_decoder*(V_plus_Z_mapped**-1),
                            axis=-1), axis=0) ) )**.5

            speech_var_multi_samples = (Z_sampled_mapped_decoder*
                                        self.a[:,:,None]) # shape (F,N,R)

            V_plus_Z_mapped = self.V[:,:,None] + speech_var_multi_samples

            cost_after_M_step[n] = torch.mean(
                    torch.log(V_plus_Z_mapped)
                    + X_abs_2[:,:,None]/V_plus_Z_mapped )

            print("iter %d/%d - cost=%.4f\n" %
                  (n+1, self.niter_MCEM, cost_after_M_step[n]))

            if n>0 and torch.abs(cost_after_M_step[n-1] - cost_after_M_step[n]) < tol:
                print('tolerance achieved')
                break

        return cost_after_M_step, n

    def separate(self, niter_MH=None, burnin=None):

        if niter_MH==None:
            niter_MH = self.niter_MH

        if burnin==None:
            burnin = self.burnin

        F, N = self.X.shape

        Z_sampled = self.metropolis_hastings(niter_MH, burnin)
        
        Z_sampled_mapped_decoder = torch.zeros((F, N, self.niter_MH-self.burnin), device=Z_sampled.device)
        
        for i in range(self.niter_MH-self.burnin):
            Z_sampled_mapped_decoder[:,:,i] = self.decoder(Z_sampled[:,:,i].T).T        

        speech_var_multi_samples = (Z_sampled_mapped_decoder*
                                    self.a[:,:,None]) # shape (F,N,R)

        self.S_hat = torch.mean(
                (speech_var_multi_samples/(speech_var_multi_samples
                                           + self.V[:,:,None])),
                                           axis=-1) * self.X

        self.N_hat = torch.mean(
                (self.V[:,:,None]/(speech_var_multi_samples
                 + self.V[:,:,None])) , axis=-1) * self.X


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

    
class VAE_decoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.decoder_layerZ = vae.decoder_layerZ
        self.output_layer = vae.output_layer
        self.activation = vae.activation
        
    def forward(self, z):
        zv = self.decoder_layerZ(z)
        hd = self.activation(zv)    
        return torch.exp(self.output_layer(hd))

    
class VAE_MCEM(BaseEncoderMaskerDecoder):
    def __init__(self, vae_full):
        super().__init__(vae_full.encoder, vae_full.masker, vae_full.decoder)
    
    def forward_masker(self, tf_rep):
        return self.run_MCEM(mag(tf_rep))

    def apply_masks(self, tf_rep, est_masks):
        return from_magphase(est_masks, angle(tf_rep))
    
    def run_MCEM(self, tf_rep):
        niter_MCEM = 100 # number of iterations for the MCEM algorithm
        niter_MH = 40 # total number of samples for the Metropolis-Hastings algorithm
        burnin = 30 # number of initial samples to be discarded
        var_MH = 0.01 # variance of the proposal distribution
        tol = 1e-5 # tolerance for stopping the MCEM iterations
        K_b = 10 # NMF rank for noise model
        
        fbank_cfg = self.encoder.filterbank.get_config()
        wlen = fbank_cfg['n_filters']
        hop = fbank_cfg['stride']
        win = torch.sin(torch.arange(.5,wlen-.5+1)/wlen*np.pi).type_as(tf_rep) # sine analysis window
 
        batch_size, F_n, N = tf_rep.shape
        decoder = VAE_decoder(self.masker)
        
        for batch_index in range(batch_size):
            X = tf_rep[batch_index]
            
            # initialization of MCEM params
            eps = 2e-15
            torch.manual_seed(0)
            W0 = F.threshold(torch.rand(F_n,K_b), eps, eps).type_as(X)
            H0 = F.threshold(torch.rand(K_b,N), eps, eps).type_as(X)
            
            latent_mean, latent_log_var = self.masker.encode((torch.abs(X)**2).T)
            Z_init = latent_mean.T
        
            mcem_algo = MCEM_algo(X=X, W=W0, H=H0, Z=Z_init, decoder=decoder,
                                  niter_MCEM=niter_MCEM, niter_MH=niter_MH, burnin=burnin,
                                  var_MH=var_MH)
            
            mcem_algo.run(hop=hop, wlen=wlen, win=win, tol=tol)
            mcem_algo.separate(niter_MH=100, burnin=75)
            
            return mcem_algo.S_hat
        
    
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
