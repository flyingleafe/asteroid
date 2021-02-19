import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
import librosa.display as lrd
import IPython.display as ipd


def show_wav(wav, sr=8000, figsize=(10,4), specgram_lib='librosa'):
    if type(wav) == str:
        wav, sr = lr.load(wav)
    elif type(wav) == torch.Tensor:
        wav = wav.detach().numpy()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    lrd.waveplot(wav, sr=sr, ax=axes[0])
    
    if specgram_lib == 'librosa':
        S_db = lr.amplitude_to_db(np.abs(lr.stft(wav)))
        img = lrd.specshow(S_db, sr=sr, ax=axes[1])
        fig.colorbar(img, ax=axes[1])
    elif specgram_lib == 'matplotlib':
        plt.specgram(wav)
    else:
        raise ValueError(f'Invalid `specgram_lib={spegram_lib}`, should be one of (`librosa`, `matplotlib`)')

    fig.tight_layout()
    plt.show()
    ipd.display(ipd.Audio(wav, rate=sr))