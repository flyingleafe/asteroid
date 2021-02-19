import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseUNet

class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
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

    
class Waveunet(nn.Module):
    def __init__(self, n_layers=12, channels_interval=24, n_src=1,
                 middle_layer=[(15, 1)]):
        super().__init__()

        self.n_src = n_src
        self.n_layers = n_layers
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

        middle_layer_size = self.n_layers * self.channels_interval
        middle_layer = [
            DownSamplingLayer(middle_layer_size, middle_layer_size,
                              kernel_size=k, dilation=d, padding=(k // 2 * d))
            for (k, d) in middle_layer
        ]
        self.middle = nn.Sequential(*middle_layer)

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
        '''
        _input = torch.randn(1, 1, 16384)
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
        
        
class UNetGANGenerator(Waveunet):
    def __init__(self, middle_layer=[(3, 1), (3, 2), (3, 4)], **kwargs):
        super().__init__(middle_layer=middle_layer, **kwargs)


class UNetGANDiscriminator(nn.Module):
    """
    """
    def __init__(self, layers=[64, 128, 256]):
        super().__init__()
        prev_ch = 1
        main_layers = []
        for ch in layers:
            main_layers.append(DownSamplingLayer(prev_ch, ch, stride=2))
            prev_ch = ch
            
        main_layers.extend([
            nn.Conv1d(prev_ch, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        ])
        
        self.main = nn.Sequential(*main_layers)
        
    def forward(self, mixture, clean_or_enh):
        mixture = mixture.unsqueeze(1)
        inp = torch.cat([mixture, clean_or_enh], dim=-1)
        return self.main(inp).mean(dim=-1)