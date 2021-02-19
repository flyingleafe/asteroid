import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid.models.base_models import BaseWavenetModel
from asteroid.utils.hub_utils import cached_download, SR_HASHTABLE
from asteroid.masknn.wavenet import Waveunet

class WaveUNet(BaseWavenetModel):
    def __init__(self, sample_rate=8000, input_length=16384, **wavenet_kwargs):
        wavenet = Waveunet(**wavenet_kwargs)
        super().__init__(wavenet, sample_rate=sample_rate)
        self.wavenet_kwargs = wavenet_kwargs
        self.input_length = input_length
    
    def apply_wavenet(self, wav):
        if wav.shape[-1] == self.input_length:
            return self.wavenet(wav)
        
        else:    
            if wav.shape[-1] % self.input_length != 0:
                padded_length = self.input_length - (wav.shape[-1] % self.input_length)
                wav = torch.cat([wav, torch.zeros(1, 1, padded_length).type_as(wav)], dim=-1)
            else:
                padded_length = 0

            assert wav.shape[-1] % self.input_length == 0 and wav.dim() == 3
            wav_chunks = list(torch.split(wav, self.input_length, dim=-1))

            enhanced_chunks = []
            for chunk in wav_chunks:
                enhanced_chunks.append(self.wavenet(chunk))

            enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]
            
            return enhanced
    
    def get_model_args(self):
        #return empty atm as configs are hardcoded for now
        return {
            **self.wavenet_kwargs,
            'sample_rate': self.sample_rate,
            'input_length': self.input_length
        }