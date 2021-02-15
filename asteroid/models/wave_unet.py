import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid.models.base_models import BaseModel
from asteroid.utils.hub_utils import cached_download, SR_HASHTABLE


class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)

class WaveUnet(BaseModel):
    def __init__(self, n_layers=12, channels_interval=24, n_src=1, sample_rate=8000.0):
        super(WaveUnet, self).__init__()

        self.n_src = n_src
        self.sample_rate=sample_rate
        self.n_layers = n_layers
        print('Number of layers: ', self.n_layers)
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        #          1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, self.n_src, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, _input):
        _input = _input.unsqueeze(1)
        '''
        _input = torch.randn(1, 1,  16384)
        16384 - should be the lenght of the mixture
        16384 - To get the correct dimension of convolution
        '''

        tmp = []
        o = _input

        # Up Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        o = self.middle(o)

        # Down Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            #import pdb; pdb.set_trace()
            # Skip Connection
            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)

        o = torch.cat([o, _input], dim=1)
        o = self.out(o)
        return o
    
    def get_model_args(self):
        #return empty atm as configs are hardcoded for now
        return {}

    def serialize(self):
        """Serialize model and output dictionary.
        Returns:
            dict, serialized model with keys `model_args` and `state_dict`.
        """
        import pytorch_lightning as pl  # Not used in torch.hub

        from .. import __version__ as asteroid_version  # Avoid circular imports

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(), #empty as of now (hardcoded)
        )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__,
            pytorch_lightning_version=pl.__version__,
            asteroid_version=asteroid_version,
        )
        model_conf["infos"] = infos
        return model_conf

    @classmethod
    def from_pretrained(cls, pretrained_model_conf_or_path, *args, **kwargs):
        """Instantiate separation model from a model config (file or dict).
        Args:
            pretrained_model_conf_or_path (Union[dict, str]): model conf as
                returned by `serialize`, or path to it. Need to contain
                `model_args` and `state_dict` keys.
            *args: Positional arguments to be passed to the model.
            **kwargs: Keyword arguments to be passed to the model.
                They overwrite the ones in the model package.
        Returns:
            nn.Module corresponding to the pretrained model conf/URL.
        Raises:
            ValueError if the input config file doesn't contain the keys
                `model_name`, `model_args` or `state_dict`.
        """
        from . import get  # Avoid circular imports

        if isinstance(pretrained_model_conf_or_path, str):
            cached_model = cached_download(pretrained_model_conf_or_path)
            conf = torch.load(cached_model, map_location="cpu")
        else:
            conf = pretrained_model_conf_or_path

        if "model_name" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "model_name`. Found only: {}".format(conf.keys())
            )
        if "state_dict" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "state_dict`. Found only: {}".format(conf.keys())
            )
        if "model_args" not in conf.keys():
            raise ValueError(
                "Expected config dictionary to have field "
                "model_args`. Found only: {}".format(conf.keys())
            )
        conf["model_args"].update(kwargs)  # kwargs overwrite config.
        if "sample_rate" not in conf["model_args"] and isinstance(
            pretrained_model_conf_or_path, str
        ):
            conf["model_args"]["sample_rate"] = SR_HASHTABLE.get(
                pretrained_model_conf_or_path, None
            )
        # Attempt to find the model and instantiate it.
        try:
            model_class = get(conf["model_name"])
        except ValueError:  # Couldn't get the model, maybe custom.
            model = cls(*args, **conf["model_args"])  # Child class.
        else:
            model = model_class(*args, **conf["model_args"])
        model.load_state_dict(conf["state_dict"])
        return model

    def get_state_dict(self):
        """ In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()
