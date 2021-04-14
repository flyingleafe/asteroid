import torch.nn as nn
import torch 
import torch.nn.functional as F

from functools import partial
from .base_models import BaseGAN, BaseEncoderMaskerDecoder
from torch.nn.utils import spectral_norm
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import mag, magreim, apply_mag_mask

from pystoi import stoi
from pypesq import pesq


class MetricSTOI(nn.Module):
    '''range: [0, 1]'''
    def __init__(self, sr, extended=False):
        super().__init__()
        self.sr = sr
        self.extended = extended
        self._device = torch.device('cpu')
        
    def forward(self, clean, enhanced):
        assert len(clean) == len(enhanced)
        scores = []
        for c, e in zip(clean, enhanced):
            q = stoi(c.detach().cpu(), e.detach().cpu(), self.sr, extended=self.extended)
            scores.append(q)
        return torch.tensor(scores, device=self._device)
    
    def to_origin_range(v):
        return v

    def to(self, *args, **kwargs):
        r = super().to(*args, **kwargs)
        if isinstance(args[0], torch.device):
            self._device = args[0]
        return r
            

class MetricPESQ(nn.Module):
    '''range: [-0.5, 4.5]'''
    def __init__(self, sr):
        super().__init__()
        self.sr = sr
        self._device = torch.device('cpu')

    def forward(self, clean, enhanced):
        assert len(clean) == len(enhanced)
        clean, enhanced = np.array(clean), np.array(enhanced)
        scores = []
        for c, e in zip(clean, enhanced):
            q = pesq(c.detach().cpu(), e.detach().cpu(), self.sr)
            q = (q+0.5)/(4.5+0.5)  # (q-min)/(max-min)
            scores.append(q)
        return torch.tensor(scores, device=self._device)
    
    def to_origin_range(self, v):
        return v*(4.5+0.5)-0.5
    
    def to(self, *args, **kwargs):
        r = super().to(*args, **kwargs)
        if isinstance(args[0], torch.device):
            self._device = args[0]
        return r
    

class Generator_Sigmoid_LSTM_Masker(nn.Module):
    def __init__(self, n_dim=257):
        super().__init__()
        self.rnn = nn.LSTM(n_dim, 300, num_layers=2, bidirectional=True, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(300*2, 300),
            nn.LeakyReLU(0.2, True),
            nn.Linear(300, n_dim),
            nn.Sigmoid()
        )

    def forward(self, spec):
        spec = spec.transpose(1, 2)  # swap dims to match (batch, time, freq)
        out_rnn, h = self.rnn(spec)
        mask = self.fc_layers(out_rnn)
        mask = mask.transpose(2, 1)  # swap back
        return mask
    
    
class Discriminator_Stride1_SN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(2, 10, 5)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(10, 20, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(20),
            spectral_norm(nn.Conv2d(20, 40, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(40, 80, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(80),
            nn.AdaptiveAvgPool2d(1),  # (b, 80, 1, 1)
            nn.Flatten(),  # (b, 80)
            spectral_norm(nn.Linear(80, 40)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(40, 10)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(10, 1))
        )

    def forward(self, x, y):
        # conditional GAN
        xy = torch.stack([x,y], dim=1)  # to shape (batch, channel, H, W)
        return self.layers(xy)

    
class MetricGAN(BaseGAN, BaseEncoderMaskerDecoder):
    def __init__(
        self,
        target_metric="ESTOI",
        mask_threshold=0.05,
        n_filters=512,
        kernel_size=512,
        stride=256,
        sample_rate=8000,
    ):
        assert target_metric in ("PESQ", "STOI", "ESTOI")
        self.target_metric = target_metric
        self.mask_threshold = mask_threshold
        
        encoder, decoder = make_enc_dec(
            "stft",
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            sample_rate=sample_rate,
        )
        
        n_dim = n_filters // 2 + 1       
        generator = Generator_Sigmoid_LSTM_Masker(n_dim=n_dim)
        discriminator = Discriminator_Stride1_SN()
        
        BaseEncoderMaskerDecoder.__init__(self, encoder, generator, decoder)
        
        self.generator = generator
        self.discriminator = discriminator
        
        if self.target_metric == "STOI":
            self.metric_module = MetricSTOI(self.sample_rate, False)
        elif self.target_metric == "ESTOI":
            self.metric_module = MetricSTOI(self.sample_rate, True)
        else:
            self.metric_module = MetricPESQ(self.sample_rate)
    
    def forward_masker(self, tf_rep):
        return self.masker(mag(tf_rep))
    
    def apply_masks(self, tf_rep, est_masks):
        if self.mask_threshold > 0:
            est_masks = F.threshold(est_masks, self.mask_threshold, 0)
        
        return apply_mag_mask(tf_rep, est_masks)
    
    def forward_generator(self, tf_rep):
        est_masks = self.forward_masker(tf_rep)
        return self.apply_masks(tf_rep, est_masks)
    
    def fake_targets(self, mix, clean, enh):
        clean_wav = self.forward_decoder(clean)
        enh_wav = self.forward_decoder(enh)
        return self.metric_module(clean_wav.squeeze(1), enh_wav.squeeze(1))
    
    def forward(self, wav):
        return BaseEncoderMaskerDecoder.forward(self, wav)
        
    def get_model_args(self):
        fb_config = self.encoder.filterbank.get_config()
        fb_config.pop('fb_name')
        model_args = {
            **fb_config,
            'target_metric': self.target_metric,
            'mask_threshold': self.mask_threshold,
        }
        return model_args
        
        