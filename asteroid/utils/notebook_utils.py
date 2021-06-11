import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
import librosa.display as lrd
import IPython.display as ipd


def show_wav(wav, sr=8000, figsize=(10,4), specgram_lib='librosa', save_to=None):
    if type(wav) == str:
        wav, sr = lr.load(wav)
    elif type(wav) == torch.Tensor:
        wav = wav.detach().numpy()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    plt.sca(axes[0])
    lrd.waveplot(wav, sr=sr, ax=axes[0])
    plt.ylabel('Amplitude')
    
    tticks, tlabels = plt.xticks()
    
    plt.sca(axes[1])
    if specgram_lib == 'librosa':
        S_db = lr.amplitude_to_db(np.abs(lr.stft(wav)))
        img = lrd.specshow(S_db, sr=sr, ax=axes[1])
        fig.colorbar(img, ax=axes[1])
    elif specgram_lib == 'matplotlib':
        plt.specgram(wav, Fs=sr, mode='magnitude')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
    else:
        raise ValueError(f'Invalid `specgram_lib={spegram_lib}`, should be one of (`librosa`, `matplotlib`)')
    
    plt.xticks(tticks)

    fig.tight_layout()
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')
    
    plt.show()
    
    ipd.display(ipd.Audio(wav, rate=sr))