import torch
from asteroid.models.base_models import BaseModel
from asteroid.masknn import activations
from asteroid.utils.torch_utils import pad_x_to_y


def _unsqueeze_to_3d(x):
    """Normalize shape of `x` to [batch, n_chan, time]."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x


class BaseEncMaskDec(BaseModel):
    def __init__(
        self, encoder, masker, decoder, encoder_activation=None, regular_training=False
    ):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation
        self.enc_activation = activations.get(encoder_activation or "linear")()
        self.regular_training = regular_training
        if self.regular_training:
            print("Trainig normally - w/ot NAT")
        else:
            print("Noise aware training")

    def forward(self, wav, noise):
        shape = torch.tensor(wav.shape)
        wav = _unsqueeze_to_3d(wav)
        noise = _unsqueeze_to_3d(noise)

        learnt_feature_mix = self.forward_encoder(wav)
        learnt_feature_noise = self.forward_encoder(noise)
        combined_feat = learnt_feature_mix + learnt_feature_noise

        if self.regular_training:
            combined_feat = self.forward_encoder(wav)
        else:
            # pass encoder through speech and  noise then add them
            learnt_feature_mix = self.forward_encoder(wav)
            learnt_feature_noise = self.forward_encoder(noise)
            combined_feat = learnt_feature_mix + learnt_feature_noise

        # estimate masks from the combined features
        est_masks = self.forward_masker(combined_feat)
        masked_tf_rep = self.apply_masks(combined_feat, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = pad_x_to_y(decoded, wav)
        return self._shape_reconstructed(reconstructed, shape)

    def apply_masks(
        self, tf_rep: torch.Tensor, est_masks: torch.Tensor
    ) -> torch.Tensor:
        return est_masks * tf_rep.unsqueeze(1)

    def forward_encoder(self, wav: torch.Tensor) -> torch.Tensor:
        tf_rep = self.encoder(wav)
        return self.enc_activation(tf_rep)

    def forward_masker(self, tf_rep: torch.Tensor) -> torch.Tensor:
        return self.masker(tf_rep)

    def forward_decoder(self, masked_tf_rep: torch.Tensor) -> torch.Tensor:
        return self.decoder(masked_tf_rep)

    def _shape_reconstructed(self, reconstructed, size):
        if len(size) == 1:
            return reconstructed.squeeze(0)
        return reconstructed
