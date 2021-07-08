from asteroid.models.base_models import BaseWavenetModel
from asteroid.masknn.wavenet import DemucsNet, apply_model_chunked

class Demucs(BaseWavenetModel):
    def __init__(self, sample_rate=8000, **wavenet_kwargs):
        wavenet = DemucsNet(**wavenet_kwargs)
        super().__init__(wavenet, sample_rate=sample_rate)
        self.wavenet_kwargs = wavenet_kwargs
    
    def apply_wavenet(self, wav):
        wav_length = wav.shape[-1]
        valid_length = self.wavenet.valid_length(wav_length)
        
        if wav_length == valid_length:
            return self.wavenet(wav)
        else:
            return apply_model_chunked(self.wavenet, wav, valid_length)
    
    def valid_length(self, length):
        return self.wavenet.valid_length(length)
    
    def get_model_args(self):
        #return empty atm as configs are hardcoded for now
        return {
            **self.wavenet_kwargs,
            'sample_rate': self.sample_rate,
        }