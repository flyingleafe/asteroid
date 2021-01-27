import os
import torch
import os.path
from torch.utils import data
import pandas as pd
import numpy as np
import librosa as lr
import soundfile as sf

from tqdm import tqdm
from pathlib import PurePath

from .utils import WavSet, TrimmedSet, add_noise_with_snr, cut_or_pad


class TimitDataset(data.Dataset):
    
    """TIMIT dataset
    """

    dataset_name = "TIMIT"

    def __init__(self, timit_dir, noise_dir, subset='train', snr=0, random_seed=42,
                 ignore_saved=False, mixtures_per_clean=2, cache_dir=None,
                 track_duration=None, dset_name=None, with_path=False, sample_rate=16000):
        if subset not in ('test', 'train'):
            raise ValueError(f'Invalid subset type {subset} (should be \'test\' or \'train\')')
        
        self.data_dir = os.path.join(timit_dir, 'data')
        self.noise_dir = noise_dir
        self.cache_dir = os.path.join(cache_dir, f'{snr}db') if cache_dir is not None else None
        self.sample_rate = sample_rate
        self.snr = snr
        self.with_path = with_path
        self.track_duration = track_duration
        
        df = pd.read_csv(os.path.join(timit_dir, f'{subset}_data.csv'))
        self.clean_df = df[df['is_converted_audio'] == True]
        
        # Assuming that the noise set is small and easily fits in the memory as a whole
        noise_set = WavSet(noise_dir, with_path=True)
        self.noises = [noise_set[i] for i in range(len(noise_set))]
        
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self.random_state = np.random.RandomState(random_seed)
        
        if self.cache_dir is not None and dset_name is not None and not ignore_saved:
            mix_csv_path = os.path.join(self.cache_dir, f'{dset_name}_{random_seed}.csv')
            if os.path.isfile(mix_csv_path):
                self.mix_df = pd.read_csv(mix_csv_path)
                return
        
        self.mix_df = pd.DataFrame(columns=['mix_path', 'clean_idx', 'noise_idx', 'random_offset'])
        
        for clean_idx in range(len(self.clean_df)):
            clean_path = PurePath(self.clean_df.iloc[clean_idx]['path_from_data_dir'])
            selected_noise_idxs = self.random_state.choice(range(len(self.noises)),
                                                           size=mixtures_per_clean, replace=False)
            for noise_idx in selected_noise_idxs:
                _, noise_path = self.noises[noise_idx]
                noise_name = PurePath(noise_path).stem
                
                if self.cache_dir is not None:
                    mix_rel_path = clean_path.with_suffix(f'.{noise_name}.wav')
                    mix_path = self.cache_dir / mix_rel_path
                else:
                    mix_path = None
                
                random_offset = self.random_state.randint(0, 10000000)
                self.mix_df.loc[len(self.mix_df)] = [mix_path, clean_idx, noise_idx, random_offset]
                
        if self.cache_dir is not None and dset_name is not None:
            mix_csv_path = os.path.join(self.cache_dir, f'{dset_name}_{random_seed}.csv')
            self.mix_df.to_csv(mix_csv_path, index=False)
                
    def _load_audio(self, path):
        audio, _ = lr.load(path, sr=self.sample_rate, mono=True, dtype='float32')
        return audio
    
    def __len__(self):
        return len(self.mix_df)

    def __getitem__(self, idx):
        row = self.mix_df.iloc[idx]
        mix_path = row['mix_path']
        clean_idx = row['clean_idx']
        noise_idx = row['noise_idx']
        
        clean_rel_path = self.clean_df.iloc[clean_idx]['path_from_data_dir']
        clean = self._load_audio(os.path.join(self.data_dir, clean_rel_path))
        noise, _ = self.noises[noise_idx]
        
        if self.track_duration is not None:
            clean = cut_or_pad(clean, self.track_duration)
        
        if mix_path is not None and os.path.isfile(mix_path):
            mix = self._load_audio(mix_path)
        else:  
            if len(noise) < len(clean):
                n_repeat = int(np.ceil(float(len(clean)) / float(len(noise))))
                noise_ex = np.tile(noise, n_repeat)
                noise = noise_ex[0 : len(clean)]
            elif len(noise) > len(clean):
                offset = row['random_offset'] % (len(noise) - len(clean) + 1)
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

        total_ds = data.ConcatDataset(sets)
        
        if prefetch_mixtures:
            preloader = data.DataLoader(total_ds, num_workers=10)

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
    
timit_license = None