import os
import torch
import zipfile
import os.path
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import pandas as pd
import numpy as np
import librosa as lr
import soundfile as sf

from torch import hub
from tqdm import tqdm
from pathlib import PurePath

from .utils import CachedWavSet, TrimmedSet, SlicedSet, add_noise_with_snr, cut_or_pad


def _load_audio(path, sr=16000):
    audio, _ = lr.load(path, sr=sr, mono=True, dtype='float32')
    return audio


class TimitDataset(Dataset):
    """
    TIMIT dataset
    """
    
    dataset_name = "TIMIT"
    download_url = "https://data.deepai.org/timit.zip"
    
    def __init__(self, timit_dir, subset='train', sample_rate=16000, with_path=True):
        if subset not in ('test', 'train'):
            raise ValueError(f'Invalid subset type {subset} (should be \'test\' or \'train\')')
            
        self.data_dir = os.path.join(timit_dir, 'data')
        self.sample_rate = sample_rate
        self.with_path = with_path
        
        df = pd.read_csv(os.path.join(timit_dir, f'{subset}_data.csv'))
        self.df = df[df['is_converted_audio'] == True]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        clean_rel_path = self.df.iloc[idx]['path_from_data_dir']
        path = os.path.join(self.data_dir, clean_rel_path)
        audio = _load_audio(path, self.sample_rate)
        
        if self.with_path:
            return audio, path
        return audio
    
    @classmethod
    def download(cls, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        exists_cond = all([
            os.path.isdir(os.path.join(out_dir, 'data')),
            os.path.isfile(os.path.join(out_dir, 'train_data.csv')),
            os.path.isfile(os.path.join(out_dir, 'test_data.csv'))
        ])
        if exists_cond:
            print('Dataset seems to be already downloaded and extracted')
            return
        
        zip_path = os.path.join(out_dir, 'timit.zip')
        hub.download_url_to_file(cls.download_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(out_dir)
                
    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "enhancement"
        infos["licenses"] = [timit_license]
        return infos
        

timit_license = dict(
    title="The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus",
    title_link="https://data.deepai.org/timit.zip",
    author="DARPA, Texas Instruments, MIT",
    author_link="https://www.darpa.mil/",
    non_commercial=False,
)


class TimitLegacyDataset(Dataset):
    """
    TIMIT dataset
    """

    dataset_name = "TIMIT"

    def __init__(self, timit, noises, subset='train', snr=0, random_seed=42,
                 ignore_saved=False, mixtures_per_clean=2, cache_dir=None,
                 crop_length=None, dset_name=None, with_path=False, sample_rate=16000):
        
        if subset not in ('test', 'train'):
            raise ValueError(f'Invalid subset type {subset} (should be \'test\' or \'train\')')
        
        self.cache_dir = os.path.join(cache_dir, f'{snr}db') if cache_dir is not None else None
        self.sample_rate = sample_rate
        self.snr = snr
        self.with_path = with_path
        self.crop_length = crop_length
        
        if isinstance(timit, Dataset):
            self.clean_set = timit
            
            # hacking around
            if isinstance(timit, Subset):
                self.data_dir = timit.dataset.data_dir
            elif isinstance(timit, TimitCleanDataset):
                self.data_dir = timit.data_dir
        else:
            self.data_dir = timit
            self.clean_set = TimitDataset(timit, subset, sample_rate=self.sample_rate)
            
        # Assuming that the noise set is small and easily fits in the memory as a whole
        if isinstance(noises, Dataset):
            self.noises = noises
        else:
            self.noises = CachedWavSet(noises, sample_rate=self.sample_rate, with_path=True)
        
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self.random_state = np.random.RandomState(random_seed)
        
        if self.cache_dir is not None and dset_name is not None and not ignore_saved:
            mix_csv_path = os.path.join(self.cache_dir, f'{dset_name}_{random_seed}.csv')
            if os.path.isfile(mix_csv_path):
                self.mix_df = pd.read_csv(mix_csv_path)
                return
        
        self.mix_df = pd.DataFrame(columns=['mix_path', 'clean_idx', 'noise_idx', 'clean_offset', 'noise_offset'])
        
        for clean_idx in range(len(self.clean_set)):
            _, clean_path = self.clean_set[clean_idx]
            clean_rel_path = PurePath(clean_path).relative_to(self.data_dir)
            selected_noise_idxs = self.random_state.choice(range(len(self.noises)),
                                                           size=mixtures_per_clean, replace=False)
            
            for noise_idx in selected_noise_idxs:
                noise_path = self.noises.wav_paths[noise_idx]
                noise_name = PurePath(noise_path).stem
                
                if self.cache_dir is not None:
                    mix_rel_path = clean_rel_path.with_suffix(f'.{noise_name}.wav')
                    mix_path = self.cache_dir / mix_rel_path
                else:
                    mix_path = None
                
                clean_offset = self.random_state.randint(0, 10000000)
                noise_offset = self.random_state.randint(0, 10000000)
                self.mix_df.loc[len(self.mix_df)] = [mix_path, clean_idx, noise_idx, clean_offset, noise_offset]
                
        if self.cache_dir is not None and dset_name is not None:
            mix_csv_path = os.path.join(self.cache_dir, f'{dset_name}_{random_seed}.csv')
            self.mix_df.to_csv(mix_csv_path, index=False)
    
    def __len__(self):
        return len(self.mix_df)

    def __getitem__(self, idx):
        
        if type(idx) == slice:
            return SlicedSet(self, idx)
        
        row = self.mix_df.iloc[idx]
        mix_path = row['mix_path']
        clean_idx = row['clean_idx']
        noise_idx = row['noise_idx']
        
        clean, clean_path = self.clean_set[clean_idx]
        
        if self.crop_length is not None:
            if self.crop_length > len(clean):
                clean = np.pad(clean, ((0, self.crop_length - len(clean)),))
            else:
                offset = row['clean_offset'] % (len(clean) - self.crop_length)
                clean = clean[offset:offset+self.crop_length]
        
        if mix_path is not None and os.path.isfile(mix_path):
            mix = _load_audio(mix_path, self.sample_rate)
        else:
            noise, noise_path = self.noises[noise_idx]

            if len(noise) < len(clean):
                n_repeat = int(np.ceil(float(len(clean)) / float(len(noise))))
                noise_ex = np.tile(noise, n_repeat)
                noise = noise_ex[0 : len(clean)]
            elif len(noise) > len(clean):
                offset = row['noise_offset'] % (len(noise) - len(clean) + 1)
                noise = noise[offset : offset+len(clean)]
            
            assert len(noise) == len(clean)
            
            mix = add_noise_with_snr(clean, noise, self.snr)
            if mix_path is not None:
                os.makedirs(PurePath(mix_path).parent, exist_ok=True)
                sf.write(file=mix_path, data=mix, samplerate=self.sample_rate)
        
        mix = torch.from_numpy(mix)
        clean = torch.from_numpy(clean)
        
        if self.with_path:
            return mix, clean, mix_path
            
        return mix, clean

    @classmethod
    def load_with_cache(cls, timit_dir, noise_dir, cache_dir, snrs,
                        root_seed=42, prefetch_mixtures=True, **kwargs):
        sets = []
        
        for i in tqdm(range(len(snrs)), 'Preparing datasets'):
            snr = snrs[i]
            seed = root_seed + i
            ds = TimitDataset(timit_dir, noise_dir, snr=snr,
                              random_seed=seed, cache_dir=cache_dir, **kwargs)
            sets.append(ds)

        total_ds = ConcatDataset(sets)
        
        if prefetch_mixtures:
            preloader = DataLoader(total_ds, num_workers=10)

            total_len = 0
            lens = []
            for batch in tqdm(preloader, 'Load samples'):
                mix = batch[0]
                total_len += mix.shape[1]
                lens.append(mix.shape[1])

            total_len = np.sum(lens)
            mean_len = np.mean(lens)
            median_len = np.median(lens)
            min_len = np.min(lens)
            max_len = np.max(lens)

            print(f'Track lengths stats: total {total_len}, mean {mean_len}, median {median_len}, min {min_len}, max {max_len}')

            total_duration = total_len // 16000
            secs = total_duration % 60
            mins = (total_duration // 60) % 60
            hours = total_duration // 3600

            print(f'Tracks in total: {len(total_ds)}')
            print(f'Total audio duration: {hours}:{mins}:{secs}')
            
        return total_ds

    def get_infos(self):
        """Get dataset infos (for publishing models).
        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "enhancement"
        #infos["licenses"] = [timit_license]
        return infos