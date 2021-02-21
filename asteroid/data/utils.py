import torch
import torch.nn.functional as F
import os
import os.path
import numpy as np
import soundfile as sf
import librosa as lr
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from collections.abc import Iterable
from tqdm import tqdm

from typing import Optional, Union, List
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


def crop_or_wrap(wav, crop_len, offset):
    if len(wav) < crop_len:
        n_repeat = int(np.ceil(float(crop_len) / float(len(wav))))
        wav_ex = np.tile(wav, n_repeat)
        wav = wav_ex[0 : crop_len]
    else:
        offset = offset % (len(wav) - crop_len + 1)
        wav = wav[offset : offset+crop_len]
    return wav
        

def find_audio_files(path, exts=[".wav"]):
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
                
    return audio_files


def _batch_head(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1:]
    return batch, None

def _batch_cons(head, tail):
    if tail is not None:
        if not isinstance(head, (list, tuple)):
            head = [head]
        return tuple(list(head) + list(tail))
    return head


class CachedWavSet(Dataset):
    """
    Class for a small dataset which fits into the memory.
    All the files in the dataset should be in one directory.
    
    Args:
        data_dir (Union[str, os.PathLike]): directory containing the wav files
        sample_rate (int, optional): sample rate to use. Default: 16000
        with_path (bool, optional): return path to files together with files themselves
    """
    def __init__(self, data_dir, sample_rate=16000, with_path=False, precache=False):
        self.with_path = with_path
        self.sample_rate = sample_rate
        
        all_files = os.listdir(data_dir)
        self.wav_paths = [os.path.join(data_dir, p) for p in all_files if p.endswith(".wav")]
        self.cache = [None]*len(self.wav_paths)
        
        if precache:
            for i in tqdm(range(len(self)), 'Precaching audio'):
                _ = self[i]
        
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


class FixedMixtureSet(Dataset):
    
    MAX_CLIP_LEN = 100000000  # arbitrary large length; should be larger than any reasonable
                              # clip length for any reasonable sample rate
    
    """
    Wrapper dataset which combines a given set of clean audio clips and set of noise clips
    in a random, but fixed manner (determined by a random seed), with a fixed SNR.
    
    Args:
        clean (Dataset): dataset with clean data
        noises (Dataset): dataset with noises data, should be with the same sample rate as clean
        snrs (float, list of float): SNRs with which to combine noises
        random_seed (int): random seed which determines the mixtures
        mixtures_per_clean (int or None): with how many noises from the noise dataset to mix each utterance.
            If `None`, mix with every noise.
        with_snr (bool): Whether or not to return SNR of mixture together with the mixture and clean sample
        crop_length (int or None): if not `None`, randomly crop/wrap each clean sample to the fixed length.
    """
    def __init__(self, clean: Dataset, noises: Dataset, snrs: Union[float, List[float]],
                 random_seed: int = 42, mixtures_per_clean: Optional[int] = None,
                 with_snr: bool = False, crop_length: Optional[int] = None):
        if mixtures_per_clean is None:
            mixtures_per_clean = len(noises)
                    
        self.clean = clean
        self.noises = noises
        self.snrs = snrs if isinstance(snrs, Iterable) else [snrs]
        self.random_state = np.random.RandomState(random_seed)
        self.with_snr = with_snr
        self.crop_length = crop_length
        
        one_snr_len = len(clean)*mixtures_per_clean
        self.mapping = np.zeros((one_snr_len*len(self.snrs), 5), dtype=int)
        
        # clean and noise indices
        offset = 0
        for snr in self.snrs:
            clean_ix = 0
            for ix in range(offset, offset+one_snr_len, mixtures_per_clean):
                self.mapping[ix:ix+mixtures_per_clean, 0] = clean_ix
                self.mapping[ix:ix+mixtures_per_clean, 1] = self.random_state.choice(
                    np.arange(len(self.noises)), size=mixtures_per_clean, replace=False)
                clean_ix += 1
            
            self.mapping[offset:offset+one_snr_len, 2] = snr
            offset += one_snr_len
            
        # noise offsets
        self.mapping[:, 3] = self.random_state.randint(0, self.MAX_CLIP_LEN, len(self))
        self.mapping[:, 4] = self.random_state.randint(0, self.MAX_CLIP_LEN, len(self))
    
    def __len__(self):
        return self.mapping.shape[0]

    def __getitem__(self, idx):        
        clean_ix, noise_ix, snr, clean_offset, noise_offset = self.mapping[idx]
        clean, tail = _batch_head(self.clean[clean_ix])
        noise = self.noises[noise_ix]
        
        if self.crop_length is not None:
            clean = crop_or_wrap(clean, self.crop_length, clean_offset)
        
        noise = crop_or_wrap(noise, len(clean), noise_offset)
        mix = add_noise_with_snr(clean, noise, snr)
        
        mix = torch.from_numpy(mix)
        clean = torch.from_numpy(clean)
        
        if self.with_snr:
            tail = [snr] if tail is None else [snr] ++ tail
        
        return _batch_cons([mix, clean], tail)
    
    
class RandomMixtureSet(Dataset):
    """
    Continuously applies random mixtures to TIMIT dataset
    """
    def __init__(self, clean: Dataset, noises: Dataset, snr_range=(-25, -5), repeat_factor=1,
                 random_seed=42, crop_length=None, with_snr=False):
        
        self.init_random_seed = random_seed
        
        self.low_snr, self.high_snr = snr_range
        self.crop_length = crop_length
        self.repeat_factor = repeat_factor
        self.with_snr = with_snr
        
        self.clean_set = clean
        self.noises = noises
        
        self.reset_random_state()
        
    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.init_random_seed)
        
    def _random_crop(self, wav, crop_length):
        if len(wav) > crop_length:
            offset = self.random_state.randint(0, len(wav) - crop_length)
        else:
            offset = 0
        
        return crop_or_wrap(wav, crop_length, offset)
        
    def __len__(self):
        #16 for testing 
        #return 16 
        return len(self.clean_set)*self.repeat_factor
    
    def __getitem__(self, idx):
        idx = idx % len(self.clean_set)
        
        clean, tail = _batch_head(self.clean_set[idx])
        noise_idx = self.random_state.randint(0, len(self.noises))
        noise = self.noises[noise_idx]
        
        if self.crop_length is not None:
            clean = self._random_crop(clean, self.crop_length)
            
        noise = self._random_crop(noise, len(clean))
        
        snr = self.random_state.uniform(self.low_snr, self.high_snr)
        mix = add_noise_with_snr(clean, noise, snr)
        
        mix = torch.from_numpy(mix)
        clean = torch.from_numpy(clean)
        
        return _batch_cons([mix, clean], tail)
    

class TrimmedSet(Dataset):
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
        

class SlicedSet(Dataset):
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
        
