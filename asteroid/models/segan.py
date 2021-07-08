import torch.nn as nn
import torch 
import torch.nn.functional as F

from functools import partial
from asteroid.masknn.wavenet import apply_model_chunked
from .base_models import BaseGAN

def build_norm_layer(norm_type, param=None, num_feats=None):
    if norm_type == 'bnorm':
        return nn.BatchNorm1d(num_feats)
    # leave only batchnorm, not using anything fancy
#     elif norm_type == 'snorm':
#         spectral_norm(param)
#         return None
    elif norm_type is None:
        return None
    else:
        raise TypeError('Unrecognized norm type: ', norm_type)

class GConv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps,
                 kwidth, stride=1, 
                 bias=True, norm_type=None):
        super().__init__()
        self.conv = nn.Conv1d(ninp, fmaps, kwidth, stride=stride, bias=bias)
        self.norm = build_norm_layer(norm_type, self.conv, fmaps)
        self.act = nn.PReLU(fmaps, init=0)
        self.kwidth = kwidth
        self.stride = stride

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x, ret_linear=False):
        if self.stride > 1:
            P = (self.kwidth // 2 - 1,
                 self.kwidth // 2)
        else:
            P = (self.kwidth // 2,
                 self.kwidth // 2)
        x_p = F.pad(x, P, mode='reflect')
        a = self.conv(x_p)
        a = self.forward_norm(a, self.norm)
        h = self.act(a)
        if ret_linear:
            return h, a
        else:
            return h

class GDeconv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps,
                 kwidth, stride=4, 
                 bias=True,
                 norm_type=None,
                 act=None):
        super().__init__()
        pad = max(0, (stride - kwidth)//-2)
        self.deconv = nn.ConvTranspose1d(ninp, fmaps,
                                         kwidth, 
                                         stride=stride,
                                         padding=pad)
        self.norm = build_norm_layer(norm_type, self.deconv,
                                     fmaps)
        if act is not None:
            self.act = getattr(nn, act)()
        else:
            self.act = nn.PReLU(fmaps, init=0)
        self.kwidth = kwidth
        self.stride = stride

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x):
        h = self.deconv(x)
        if self.kwidth % 2 != 0:
            h = h[:, :, :-1]
        h = self.forward_norm(h, self.norm)
        h = self.act(h)
        return h
    

class GSkip(nn.Module):

    def __init__(self, skip_type, size, skip_init, skip_dropout=0,
                 merge_mode='sum', kwidth=11, bias=True):
        # skip_init only applies to alpha skips
        super().__init__()
        self.merge_mode = merge_mode
        if skip_type == 'alpha' or skip_type == 'constant':
            if skip_init == 'zero':
                alpha_ = torch.zeros(size)
            elif skip_init == 'randn':
                alpha_ = torch.randn(size)
            elif skip_init == 'one':
                alpha_ = torch.ones(size)
            else:
                raise TypeError('Unrecognized alpha init scheme: ', 
                                skip_init)
            #if cuda:
            #    alpha_ = alpha_.cuda()
            if skip_type == 'alpha':
                self.skip_k = nn.Parameter(alpha_.view(1, -1, 1))
            else:
                # constant, not learnable
                self.skip_k = nn.Parameter(alpha_.view(1, -1, 1))
                self.skip_k.requires_grad = False
        elif skip_type == 'conv':
            if kwidth > 1:
                pad = kwidth // 2
            else:
                pad = 0
            self.skip_k = nn.Conv1d(size, size, kwidth, stride=1,
                                    padding=pad, bias=bias)
        else:
            raise TypeError('Unrecognized GSkip scheme: ', skip_type)
        self.skip_type = skip_type
        if skip_dropout > 0:
            self.skip_dropout = nn.Dropout(skip_dropout)

    def __repr__(self):
        if self.skip_type == 'alpha':
            return self._get_name() + '(Alpha(1))'
        elif self.skip_type == 'constant':
            return self._get_name() + '(Constant(1))'
        else:
            return super().__repr__()

    def forward(self, hj, hi):
        if self.skip_type == 'conv':
            sk_h = self.skip_k(hj)
        else:
            skip_k = self.skip_k.repeat(hj.size(0), 1, hj.size(2)).type_as(hj)
            sk_h =  skip_k * hj
        if hasattr(self, 'skip_dropout'):
            sk_h = self.skip_dropout(sk_h)
        if self.merge_mode == 'sum':
            # merge with input hi on current layer
            return sk_h + hi
        elif self.merge_mode == 'concat':
            return torch.cat((hi, sk_h), dim=1)
        else:
            raise TypeError('Unrecognized skip merge mode: ', self.merge_mode)
            

class Generator(nn.Module):

    def __init__(self, ninputs, fmaps,
                 kwidth, poolings, 
                 dec_fmaps=None,
                 dec_kwidth=None,
                 dec_poolings=None,
                 z_dim=None,
                 no_z=False,
                 skip=True,
                 bias=False,
                 skip_init='one',
                 skip_dropout=0,
                 skip_type='alpha',
                 norm_type=None,
                 skip_merge='sum',
                 skip_kwidth=11):
        super().__init__()
        self.skip = skip
        self.bias = bias
        self.no_z = no_z
        
        self.enc_blocks = nn.ModuleList()
        assert isinstance(fmaps, list), type(fmaps)
        assert isinstance(poolings, list), type(poolings)
        if isinstance(kwidth, int): 
            kwidth = [kwidth] * len(fmaps)
        assert isinstance(kwidth, list), type(kwidth)
        skips = {}
        ninp = ninputs
        for pi, (fmap, pool, kw) in enumerate(zip(fmaps, poolings, kwidth),
                                              start=1):
            if skip and pi < len(fmaps):
                # Make a skip connection for all but last hidden layer
                gskip = GSkip(skip_type, fmap,
                              skip_init,
                              skip_dropout,
                              merge_mode=skip_merge,
                              kwidth=skip_kwidth,
                              bias=bias)
                l_i = pi - 1
                skips[l_i] = gskip
                setattr(self, 'alpha_{}'.format(l_i), skips[l_i])
            enc_block = GConv1DBlock(
                ninp, fmap, kw, stride=pool, bias=bias,
                norm_type=norm_type
            )
            self.enc_blocks.append(enc_block)
            ninp = fmap

        self.skips = skips
        if not no_z and z_dim is None:
            z_dim = fmaps[-1]
            
        self.z_dim = z_dim    
        
        if not no_z:
            ninp += z_dim
        # Ensure we have fmaps, poolings and kwidth ready to decode
        if dec_fmaps is None:
            dec_fmaps = fmaps[::-1][1:] + [1]
        else:
            assert isinstance(dec_fmaps, list), type(dec_fmaps)
        if dec_poolings is None:
            dec_poolings = poolings[:]
        else:
            assert isinstance(dec_poolings, list), type(dec_poolings)
        self.dec_poolings = dec_poolings
        if dec_kwidth is None:
            dec_kwidth = kwidth[:]
        else:
            if isinstance(dec_kwidth, int): 
                dec_kwidth = [dec_kwidth] * len(dec_fmaps)
        assert isinstance(dec_kwidth, list), type(dec_kwidth)
        # Build the decoder
        self.dec_blocks = nn.ModuleList()
        for pi, (fmap, pool, kw) in enumerate(zip(dec_fmaps, dec_poolings, 
                                                  dec_kwidth),
                                              start=1):
            if skip and pi > 1 and pool > 1:
                if skip_merge == 'concat':
                    ninp *= 2

            if pi >= len(dec_fmaps):
                act = 'Tanh'
            else:
                act = None
            if pool > 1:
                dec_block = GDeconv1DBlock(
                    ninp, fmap, kw, stride=pool,
                    norm_type=norm_type, bias=bias,
                    act=act
                )
            else:
                dec_block = GConv1DBlock(
                    ninp, fmap, kw, stride=1, 
                    bias=bias,
                    norm_type=norm_type
                )
            self.dec_blocks.append(dec_block)
            ninp = fmap

    def forward(self, x, z=None, ret_hid=False):
        hall = {}
        hi = x
        skips = self.skips
        
        skip_tensors = {}
        for l_i, enc_layer in enumerate(self.enc_blocks):
            hi, linear_hi = enc_layer(hi, True)
            #print('ENC {} hi size: {}'.format(l_i, hi.size()))
                    #print('Adding skip[{}]={}, alpha={}'.format(l_i,
                    #                                            hi.size(),
                    #                                            hi.size(1)))
            if self.skip and l_i < (len(self.enc_blocks) - 1):
                skip_tensors[l_i] = linear_hi
            if ret_hid:
                hall['enc_{}'.format(l_i)] = hi
        if not self.no_z:
            if z is None:
                # make z 
                z = torch.randn(hi.size(0), self.z_dim, *hi.size()[2:]).type_as(hi)
            if len(z.size()) != len(hi.size()):
                raise ValueError('len(z.size) {} != len(hi.size) {}'
                                 ''.format(len(z.size()), len(hi.size())))
            if not hasattr(self, 'z'):
                self.z = z
            hi = torch.cat((z, hi), dim=1)
            if ret_hid:
                hall['enc_zc'] = hi
        else:
            z = None
        enc_layer_idx = len(self.enc_blocks) - 1
        for l_i, dec_layer in enumerate(self.dec_blocks):
            if self.skip and enc_layer_idx in self.skips and \
            self.dec_poolings[l_i] > 1:
                skip_conn = skips[enc_layer_idx]
                skip_tensor = skip_tensors[enc_layer_idx]
                #hi = self.skip_merge(skip_conn, hi)
                #print('Merging  hi {} with skip {} of hj {}'.format(hi.size(),
                #                                                    l_i,
                #                                                    skip_conn['tensor'].size()))
                hi = skip_conn(skip_tensor, hi)
            #print('DEC in size after skip and z_all: ', hi.size())
            #print('decoding layer {} with input {}'.format(l_i, hi.size()))
            hi = dec_layer(hi)
            #print('decoding layer {} output {}'.format(l_i, hi.size()))
            enc_layer_idx -= 1
            if ret_hid:
                hall['dec_{}'.format(l_i)] = hi
        if ret_hid:
            return hi, hall
        else:
            return hi


class Discriminator(nn.Module):
    
    def __init__(self, ninputs, fmaps,
                 kwidth, poolings,
                 pool_type='none',
                 pool_slen=None,
                 norm_type='bnorm',
                 bias=True,
                 phase_shift=None):
        super().__init__()
        # phase_shift randomly occurs within D layers
        # as proposed in https://arxiv.org/pdf/1802.04208.pdf
        # phase shift has to be specified as an integer
        self.phase_shift = phase_shift
        if phase_shift is not None:
            assert isinstance(phase_shift, int), type(phase_shift)
            assert phase_shift > 1, phase_shift
        if pool_slen is None:
            raise ValueError('Please specify D network pool seq len '
                             '(pool_slen) in the end of the conv '
                             'stack: [inp_len // (total_pooling_factor)]')
        ninp = ninputs
        
        self.enc_blocks = nn.ModuleList()
        for pi, (fmap, pool) in enumerate(zip(fmaps,
                                              poolings),
                                          start=1):
            enc_block = GConv1DBlock(
                ninp, fmap, kwidth, stride=pool,
                bias=bias,
                norm_type=norm_type
            )
            self.enc_blocks.append(enc_block)
            ninp = fmap
        self.pool_type = pool_type
        if pool_type == 'none':
            # resize tensor to fit into FC directly
            pool_slen *= fmaps[-1]
            self.fc = nn.Sequential(
                nn.Linear(pool_slen, 256),
                nn.PReLU(256),
                nn.Linear(256, 128),
                nn.PReLU(128),
                nn.Linear(128, 1)
            )
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.fc[0])
                torch.nn.utils.spectral_norm(self.fc[2])
                torch.nn.utils.spectral_norm(self.fc[3])
        elif pool_type == 'conv':
            self.pool_conv = nn.Conv1d(fmaps[-1], 1, 1)
            self.fc = nn.Linear(pool_slen, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.pool_conv)
                torch.nn.utils.spectral_norm(self.fc)
        elif pool_type == 'gmax':
            self.gmax = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(fmaps[-1], 1, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.fc)
        elif pool_type == 'gavg':
            self.gavg = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(fmaps[-1], 1, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.fc)
        elif pool_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Conv1d(fmaps[-1], fmaps[-1], 1),
                nn.PReLU(fmaps[-1]),
                nn.Conv1d(fmaps[-1], 1, 1)
            )
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.mlp[0])
                torch.nn.utils.spectral_norm(self.mlp[1])
        else:
            raise TypeError('Unrecognized pool type: ', pool_type)
    
    def forward(self, x):
        h = x
        # store intermediate activations
        #int_act = {}
        for ii, layer in enumerate(self.enc_blocks):
            if self.phase_shift is not None:
                shift = random.randint(1, self.phase_shift)
                # 0.5 chance of shifting right or left
                right = random.random() > 0.5
                # split tensor in time dim (dim 2)
                if right:
                    sp1 = h[:, :, :-shift]
                    sp2 = h[:, :, -shift:]
                    h = torch.cat((sp2, sp1), dim=2)
                else:
                    sp1 = h[:, :, :shift]
                    sp2 = h[:, :, shift:]
                    h = torch.cat((sp2, sp1), dim=2)
            h = layer(h)
            #int_act['h_{}'.format(ii)] = h
        if self.pool_type == 'conv':
            h = self.pool_conv(h)
            h = h.view(h.size(0), -1)
            #int_act['avg_conv_h'] = h
            y = self.fc(h)
        elif self.pool_type == 'none':
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'gmax':
            h = self.gmax(h)
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'gavg':
            h = self.gavg(h)
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'mlp':
            y = self.mlp(h)
        #int_act['logit'] = y
        #return y, int_act
        return y
    
    
class SEGAN(BaseGAN):
    def __init__(self, sample_rate=8000, no_z=False):
        # original SEGAN parameters
        fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        poolings = [2]*len(fmaps)
        kwidth = 31
        
        gen = Generator(1,
            fmaps,
            kwidth=kwidth,
            poolings=poolings,
            skip_merge='concat',
            skip_type='constant',
            no_z = no_z,
            bias=True)
        
        disc = Discriminator(2,
            fmaps,
            kwidth=kwidth,
            poolings=poolings,
            pool_type='conv',
            pool_slen=8)
        
        super().__init__(gen, disc, sample_rate=sample_rate)
            
    def forward_discriminator(self, mix, clean_or_enh):
        return self.discriminator(torch.cat((mix, clean_or_enh), dim=1))
    
    def forward_generator(self, wav, z=None):
        if wav.shape[-1] == 16384:
            return self.generator(wav, z=z)
        else:
            return apply_model_chunked(partial(self.generator, z=z), wav, 16384)
    
    def valid_length(self, length):
        rem = length % 16384
        if rem == 0:
            return length
        return length + (16384 - rem)
    
    def get_model_args(self):
        return {
            'sample_rate': self.sample_rate,
            'no_z': self.generator.no_z,
        }