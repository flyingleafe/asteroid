import torch.nn as nn
import torch 
import torch.nn.functional as F

from .base_models import BaseEncoderMaskerDecoder
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import mag, reim, apply_mag_mask, angle, from_magphase
from ..masknn import norms, activations
from einops import rearrange

class FTB(nn.Module):

    def __init__(self, input_dim=257, in_channel=9, r_channel=5):
        super(FTB, self).__init__()
        
        self.in_channel = in_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, r_channel, kernel_size=[1,1]),
            nn.BatchNorm2d(r_channel),
            nn.ReLU()
        )
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(r_channel*input_dim, in_channel, kernel_size=9,padding=4),
            nn.BatchNorm1d(in_channel),
            nn.ReLU()
        )
        self.freq_fc = nn.Linear(input_dim, input_dim, bias=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel*2, in_channel, kernel_size=[1,1]),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, inputs):
        '''
        inputs should be [Batch, Ca, Dim, Time]
        '''
        # T-F attention        
        conv1_out = self.conv1(inputs)
        B, C, D, T= conv1_out.size()
        reshape1_out = torch.reshape(conv1_out,[B, C*D, T])
        conv1d_out = self.conv1d(reshape1_out)
        conv1d_out = torch.reshape(conv1d_out, [B, self.in_channel,1,T])
        
        # now is also [B,C,D,T]
        att_out = conv1d_out*inputs
        
        # tranpose to [B,C,T,D]
        att_out = torch.transpose(att_out, 2, 3)
        freqfc_out = self.freq_fc(att_out)
        att_out = torch.transpose(freqfc_out, 2, 3)

        cat_out = torch.cat([att_out, inputs], 1)
        outputs = self.conv2(cat_out)
        return outputs

    
class InforComu(nn.Module):
        
        def __init__(self, src_channel, tgt_channel):
            super(InforComu, self).__init__()
            self.comu_conv = nn.Conv2d(src_channel, tgt_channel, kernel_size=(1,1))
        
        def forward(self, src, tgt):
            outputs=tgt*torch.tanh(self.comu_conv(src))
            return outputs


class GLayerNorm2d(nn.Module):
    
    def __init__(self, in_channel, eps=1e-12):
        super(GLayerNorm2d, self).__init__()
        self.eps = eps 
        self.beta = nn.Parameter(torch.ones([1, in_channel,1,1]))
        self.gamma = nn.Parameter(torch.zeros([1, in_channel,1,1]))
    
    def forward(self,inputs):
        mean = torch.mean(inputs,[1,2,3], keepdim=True)
        var = torch.var(inputs,[1,2,3], keepdim=True)
        outputs = (inputs - mean)/ torch.sqrt(var+self.eps)*self.beta+self.gamma
        return outputs


class TSB(nn.Module):

    def __init__(self, input_dim=257, channel_amp=9, channel_phase=8):
        super(TSB, self).__init__()
        
        self.ftb1 = FTB(input_dim=input_dim, in_channel=channel_amp)
        self.amp_conv1 = nn.Sequential(
            nn.Conv2d(channel_amp, channel_amp, kernel_size=(5,5), padding=(2,2)),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU()
        )
        self.amp_conv2 = nn.Sequential(
            nn.Conv2d(channel_amp, channel_amp, kernel_size=(1,25), padding=(0,12)),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU()
        )
        self.amp_conv3 = nn.Sequential(
            nn.Conv2d(channel_amp, channel_amp, kernel_size=(5,5), padding=(2,2)),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU()
        )
        
        self.ftb2 = FTB(input_dim=input_dim, in_channel=channel_amp)

        self.phase_conv1 = nn.Sequential(
            nn.Conv2d(channel_phase, channel_phase, kernel_size=(5,5), padding=(2,2)),
            GLayerNorm2d(channel_phase),
        )
        self.phase_conv2 = nn.Sequential(
            nn.Conv2d(channel_phase, channel_phase, kernel_size=(1,25), padding=(0,12)),
            GLayerNorm2d(channel_phase),
        )

        self.p2a_comu = InforComu(channel_phase, channel_amp)
        self.a2p_comu = InforComu(channel_amp, channel_phase)

    def forward(self, amp, phase):
        '''
        amp should be [Batch, Ca, Dim, Time]
        amp should be [Batch, Cr, Dim, Time]
        
        '''
        
        amp_out1 = self.ftb1(amp)
        amp_out2 = self.amp_conv1(amp_out1)
        amp_out3 = self.amp_conv2(amp_out2)
        amp_out4 = self.amp_conv3(amp_out3)
        amp_out5 = self.ftb2(amp_out4)
        
        phase_out1 = self.phase_conv1(phase)
        phase_out2 = self.phase_conv2(phase_out1)
        
        amp_out = self.p2a_comu(phase_out2, amp_out5)
        phase_out = self.a2p_comu(amp_out5, phase_out2)
        
        return amp_out, phase_out
    
    
class PhasenMasker(nn.Module):

    def __init__(
        self,
        fft_len=512,
        num_blocks=3,
        channel_amp=24,
        channel_phase=12,
        rnn_nums=300
    ):
        super().__init__() 
        self.num_blocks = 3
        self.feat_dim = fft_len // 2 + 1 
       
 #        self.win_len = win_len
#         self.win_inc = win_inc
#         self.fft_len = fft_len 
#         self.win_type = win_type 

#         fix = True
#         self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='complex', fix=fix)
#         self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='complex', fix=fix)
        
        self.amp_conv1 = nn.Sequential(
            nn.Conv2d(2, channel_amp, 
                kernel_size=[7,1],
                padding=(3,0)
            ),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU(),
            nn.Conv2d(channel_amp, channel_amp, 
                kernel_size=[1,7],
                padding=(0,3)
            ),
            nn.BatchNorm2d(channel_amp),
            nn.ReLU(),
        )
        self.phase_conv1 = nn.Sequential(
            nn.Conv2d(2, channel_phase, 
                kernel_size=[3,5],
                padding=(1,2)
            ),
            nn.Conv2d(channel_phase, channel_phase, 
                kernel_size=[3,25],
                padding=(1, 12)
            ),
        )

        self.tsbs = nn.ModuleList()
        for idx in range(self.num_blocks):
            self.tsbs.append(
                TSB(input_dim=self.feat_dim,
                    channel_amp=channel_amp,
                    channel_phase=channel_phase
                )
            )
   
        self.amp_conv2 = nn.Sequential(
            nn.Conv2d(channel_amp, 8, kernel_size=[1, 1]),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.phase_conv2 = nn.Conv1d(channel_phase,2,kernel_size=[1,1])
        
        self.rnn = nn.LSTM(
            self.feat_dim * 8,
            rnn_nums,
            bidirectional=True,
            batch_first=True,
        )
        
        self.fcs = nn.Sequential(
            nn.Linear(rnn_nums*2,600),
            nn.ReLU(),
            nn.Linear(600,600),
            nn.ReLU(),
            nn.Linear(600,self.feat_dim),
            nn.Sigmoid()
        )

    def forward(self, cmp_spec):
#         # [B, D*2, T]
#         cmp_spec = self.stft(inputs)
#         cmp_spec = torch.unsqueeze(cmp_spec, 1)

#         # to [B, 2, D, T]
#         cmp_spec = torch.cat([
#             cmp_spec[:,:,:self.feat_dim,:],
#             cmp_spec[:,:,self.feat_dim:,:],
#             ],
#             1)

#         # to [B, 1, D, T]
#         amp_spec = torch.sqrt(
#                             torch.abs(cmp_spec[:,0])**2+
#                             torch.abs(cmp_spec[:,1])**2,
#                         )
#         amp_spec = torch.unsqueeze(amp_spec, 1)
        
        spec = self.amp_conv1(cmp_spec)
        phase = self.phase_conv1(cmp_spec)
#         s_spec = spec
#         s_phase = phase
        for idx, layer in enumerate(self.tsbs):
#             if idx != 0:
#                 spec += s_spec
#                 phase += s_phase
            spec, phase = layer(spec, phase)
        spec = self.amp_conv2(spec)

        spec = torch.transpose(spec, 1,3)
        B, T, D, C = spec.size()
        spec = torch.reshape(spec, [B, T, D*C])
        spec = self.rnn(spec)[0]
        spec = self.fcs(spec)
        
        spec = torch.reshape(spec, [B,T,D,1]) 
        spec = torch.transpose(spec, 1,3)
        
        phase = self.phase_conv2(phase)
        # norm to 1
        phase = phase / (torch.sqrt(
            torch.abs(phase[:,0])**2+
            torch.abs(phase[:,1])**2)
        +1e-8).unsqueeze(1)
        
        return spec, phase
#         est_spec = amp_spec * spec * phase 
#         est_spec = torch.cat([est_spec[:,0], est_spec[:,1]], 1)
#         est_wav = self.istft(est_spec)
#         est_wav = torch.squeeze(est_wav, 1)
        #t = amp_spec 
#         return est_spec, est_wav# , [t[0], spec[0], phase[0]
    

class Phasen(BaseEncoderMaskerDecoder):
    def __init__(
        self,
        n_filters=512,
        kernel_size=400,
        stride=100,
        sample_rate=8000,
        **masker_kwargs
    ):
        encoder, decoder = make_enc_dec(
            "stft",
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
            sample_rate=sample_rate,
        )
        
        masker = PhasenMasker(fft_len=n_filters, **masker_kwargs)
        super().__init__(encoder, masker, decoder)
        

    def forward_masker(self, tf_rep):
        tf_rep = tf_rep.unsqueeze(1)
        input_rep = torch.cat(reim(tf_rep), dim=1)
        mag_mask, phase = self.masker(input_rep)
        
        # reshape back to Asteroid format
        phase = torch.cat([phase[:,0], phase[:,1]], dim=-2)
        return mag_mask.squeeze(1), phase.squeeze(1)
    
    
    def apply_masks(self, tf_rep, est_masks):
        orig_mag = mag(tf_rep)
        mag_mask, phase = est_masks
        return apply_mag_mask(phase, orig_mag * mag_mask)