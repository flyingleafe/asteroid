

Noise aware training (NAT) -


To begin training -
```
Run ./run.sh
```
please change directory of datasets 

Noise Aware Training implementation for speech separation inspired from -  `Dynamic Noise Aware Training for Speech Enhancement Based on Deep Neural Networks` [Link](https://www.isca-speech.org/archive/archive_papers/interspeech_2014/i14_2670.pdf) 


* `base.py` - contains base class `BaseEncMaskDec` which takes encoder, masker and decoder funtions as input

```python
class BaseEncMaskDec(BaseModel):
    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation
        self.enc_activation = activations.get(encoder_activation or "linear")()
```
* encoder -
    * encoder encodes - mixture and noise using encoder to produce a combined encoded representation
    * Paper estimates noise - however here we directly feed noise get test if NAT works

```python
learnt_feature_mix = self.forward_encoder(wav)
learnt_feature_noise = self.forward_encoder(noise)
combined_feat = learnt_feature_mix + learnt_feature_noise
```
* The combined encoded feature (noisy speech + noise) is supposed to give complemetry information which can help the network to denoise speech effectively

```python
est_masks = self.forward_masker(combined_feat)
masked_tf_rep = self.apply_masks(combined_feat, est_masks)
decoded = self.forward_decoder(masked_tf_rep)
```
