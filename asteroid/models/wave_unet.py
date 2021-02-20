from asteroid.models.base_models import BaseWavenetModel
from asteroid.masknn.wavenet import Waveunet, apply_model_chunked

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
            return apply_model_chunked(self.wavenet, wav, self.input_length)
    
    def get_model_args(self):
        #return empty atm as configs are hardcoded for now
        return {
            **self.wavenet_kwargs,
            'sample_rate': self.sample_rate,
            'input_length': self.input_length
        }