import torch
import torch.nn.functional as F
import os
import os.path
import numpy as np
import soundfile as sf
import librosa as lr
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from collections.abc import Iterable

from pathlib import Path, PurePath

def online_mixing_collate(batch):
    """Mix target sources to create new mixtures.
    Output of the default collate function is expected to return two objects:
    inputs and targets.
    """
    # Inputs (batch, time) / targets (batch, n_src, time)
    inputs, targets = default_collate(batch)
    batch, n_src, _ = targets.shape

    energies = torch.sum(targets ** 2, dim=-1, keepdim=True)
    new_src = []
    for i in range(targets.shape[1]):
        new_s = targets[torch.randperm(batch), i, :]
        new_s = new_s * torch.sqrt(energies[:, i] / (new_s ** 2).sum(-1, keepdims=True))
        new_src.append(new_s)

    targets = torch.stack(new_src, dim=1)
    inputs = targets.sum(1)
    return inputs, targets


def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10 ** a)
    return noise_rms

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def add_noise_with_snr(clean, noise, snr):
    clean_rms = cal_rms(clean)
    noise_rms = cal_rms(noise)
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
    
    noise = noise * (adjusted_noise_rms / noise_rms)
    
    mixed = clean + noise
    alpha = 1.0
    
    if mixed.max(axis=0) > 1 or mixed.min(axis=0) < -1:
        if mixed.max(axis=0) >= abs(mixed.min(axis=0)):
            alpha = 1. / mixed.max(axis=0)
        else:
            alpha = -1. / mixed.min(axis=0)
        mixed = mixed * alpha

    return mixed


def cut_or_pad(tensor, length):
    if tensor.shape[0] >= length:
        return tensor[:length]
    else:
        if isinstance(tensor, torch.Tensor):
            return F.pad(tensor, (0, length - tensor.shape[0]))
        else:
            return np.pad(tensor, ((0, length - tensor.shape[0]),))


def find_audio_files(path, exts=[".wav"]):
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
                
    return audio_files


class CachedWavSet(data.Dataset):
    """
    Class for a small dataset which fits into the memory.
    All the files in the dataset should be in one directory.
    
    Args:
        data_dir (Union[str, os.PathLike]): directory containing the wav files
        sample_rate (int, optional): sample rate to use. Default: 16000
        with_path (bool, optional): return path to files together with files themselves
    """
    def __init__(self, data_dir, sample_rate=16000, with_path=False):
        self.with_path = with_path
        self.sample_rate = sample_rate
        
        all_files = os.listdir(data_dir)
        self.wav_paths = [os.path.join(data_dir, p) for p in all_files if p.endswith(".wav")]
        self.cache = [None]*len(self.wav_paths)
        
    def __len__(self):
        return len(self.wav_paths)
    
    def __getitem__(self, idx):
        cached = self.cache[idx]
        path = self.wav_paths[idx]
        if cached is None:
            cached, _ = lr.load(path, sr=self.sample_rate, mono=True, dtype='float32')
            self.cache[idx] = cached

        if self.with_path:
            return cached, PurePath(path)
        return cached
    

class TrimmedSet(data.Dataset):
    """
    Utility wrapper which trims/pads clips in the dataset to a given length    
    """
    def __init__(self, ds, length):
        self.ds = ds
        self.length = length
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        batches = self.ds[idx]
        if isinstance(batches, Iterable):
            return tuple(cut_or_pad(b, self.length) for b in batches)
        else:
            return cut_or_pad(batches, self.length)
        

class SlicedSet(data.Dataset):
    """
    Utility wrapper which allows seamless slicing of any dataset
    """
    
    def __init__(self, ds, slc):
        if type(slc) != slice:
            raise ValueError('Index argument should be a slice')
        
        start = slc.start if slc.start is not None else 0
        stop = slc.stop if slc.stop is not None else len(ds)
        step = slc.step if slc.step is not None else 1
        
        self.idxs = list(range(start, stop, step))
        self.ds = ds
        
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, idx):
        if type(idx) == slice:
            return SlicedSet(self, idx)
        
        return self.ds[self.idxs[idx]]
        