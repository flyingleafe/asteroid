import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid.models.base_models import BaseModel
import numpy as np
import pandas as pd
import librosa as lr
import soundfile as sf
import matplotlib.pyplot as plt
import time
import os.path
import sys

from librosa import display as lrd
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
from functools import partial
from torch.multiprocessing import Process, Queue, cpu_count
from queue import Empty

from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from asteroid.data import TimitDataset
from asteroid.data.utils import CachedWavSet, FixedMixtureSet
from tqdm import trange, tqdm
from asteroid import DCUNet, DCCRNet, DPRNNTasNet, ConvTasNet, RegressionFCNN, WaveUNet, DPTNet, Demucs, SMoLnet
#from asteroid import DCUNet, DCCRNet, DPRNNTasNet, ConvTasNet, RegressionFCNN, WaveUNet, DPTNet
from asteroid.engine.system import UNetGAN

from asteroid.masknn import UNetGANGenerator, UNetGANDiscriminator
from asteroid.utils.notebook_utils import show_wav

sys.path.append('../egs')

from timit_drones.evaluate import evaluate_model

TIMIT_DIR_8kHZ ='/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/TIMIT_8kHZ' 
TEST_NOISE_DIR = '/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/noises-test-drones'
SAMPLE_RATE    = 8000
TEST_SNRS      = [-30, -25, -20, -15, -10, -5, 0]
SEED           = 42


timit_test_clean = TimitDataset(TIMIT_DIR_8kHZ, subset='test', sample_rate=SAMPLE_RATE, with_path=False)
timit_small = Subset(timit_test_clean, np.arange(len(timit_test_clean)//20))
noises_test = CachedWavSet(TEST_NOISE_DIR, sample_rate=SAMPLE_RATE, precache=True)


timit_test_small = FixedMixtureSet(timit_small, noises_test, snrs=TEST_SNRS, random_seed=SEED, with_snr=True)
timit_test = FixedMixtureSet(timit_test_clean, noises_test, snrs=TEST_SNRS, random_seed=SEED, with_snr=True)

torch.multiprocessing.set_sharing_strategy('file_system')

metrics_names = {
    'pesq': 'PESQ',
    'stoi': 'STOI',
    'si_sdr': 'SI-SDR',
}

#mapping for plotting
model_labels = {
    'input': 'Input',
    'baseline': 'Baseline DNN',
    'baseline_v2': 'Baseline DNN (L1 loss)',
    'baseline_proper_mse': 'Baseline DNN (fixed test set)',
    'waveunet_v1': 'Wave-U-Net',
    'dcunet_20': 'DCUNet-20',
    'dccrn': 'DCCRN',
    'smolnet': 'SMoLnet',
    'dprnn': 'DPRNN',
    'conv_tasnet': 'Conv-TasNet',
    'dptnet': 'DPTNet',
    'demucs': 'Demucs',
}

#model_labels = {
#    'input': 'Input',
#    'baseline': 'Baseline DNN',
#    'baseline_v2': 'Baseline DNN (L1 loss)',
#    'baseline_proper_mse': 'Baseline DNN (fixed test set)',
#    'waveunet_v1': 'Wave-U-Net',
#    'dcunet_20': 'DCUNet-20',
#    'dccrn': 'DCCRN',
#    'dprnn': 'DPRNN',
#    'conv_tasnet': 'Conv-TasNet',
#    'dptnet': 'DPTNet'
#}

def plot_results(dfs, figsize=(15, 5), metrics=['pesq', 'stoi', 'si_sdr'],
                 plot_name=None):

    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=figsize)

    for model_name, df in dfs.items():
        scores = df.groupby('snr').agg({
            metric: ['mean', 'std', 'count'] for metric in metrics
        })

        line_kwargs = {'marker': 'o', 'alpha': 0.8}
        fill_kwargs = {}
        if model_name == 'input':
            line_kwargs = {'c': 'black', 'ls': '--'}
            fill_kwargs = {'color': 'black'}

        for i, metric in enumerate(metrics):
            plt.sca(axes[i])
            means = scores[metric]['mean']
            stds = scores[metric]['std'].values / np.sqrt(scores[metric]['count'].values) * 3
            xs = means.index
            plt.plot(xs, means, label=model_labels[model_name], **line_kwargs)
            plt.fill_between(xs, means - stds, means + stds, alpha=0.2, **fill_kwargs)

    for i, metric in enumerate(metrics):
        plt.sca(axes[i])
        plt.grid(which='both')
        plt.title(metrics_names[metric])
        plt.xlabel('SNR, dB')
        if i == 0:
            plt.legend()

    if plot_name is not None:
        plt.savefig(plot_name, bbox_inches='tight')

    #plt.show()

def highlight_max(s): 
    if s.dtype == np.object: 
        is_max = [False for _ in range(s.shape[0])] 
    else: 
        is_max = s == s.max() 
    return ['font-weight: bold' if cell else '' for cell in is_max] 

def avg_results_table(dfs, models, metrics=['pesq', 'stoi', 'si_sdr']):
    total_df = pd.DataFrame(columns=['Model', 'N. of params'] + [metrics_names[m] for m in metrics])
    for model_name, df in dfs.items():
        model = models[model_name]
        if model is not None:
            param_count = sum(p.numel() for p in model.parameters())
            param_approx = np.around(param_count / 1000000, decimals=2)
        else:
            param_approx = None

        total_df.loc[len(total_df)] = [model_labels[model_name], f'{param_approx}M'] + list(df[metrics].mean(axis=0))

    #total_df = total_df.style.apply(highlight_max)
    #import pdb; pdb.set_trace()
    total_df.sort_values(['SI-SDR', 'PESQ', 'STOI'], inplace=True)
    total_df = total_df.round({'SI-SDR': 3, 'PESQ': 3, 'STOI': 3})
    print(total_df)
    return total_df
#
#models = {
#    'input': None,
#    #'baseline': RegressionFCNN.from_pretrained('../../../workspace/models/baseline_model_v1.pt'),
#    #'baseline_v2': RegressionFCNN.from_pretrained('../../../workspace/models/baseline_model_v2.pt'),
#    #'baseline_proper_mse': RegressionFCNN.from_pretrained('../../../workspace/models/baseline_model_fixed_mse.pt'),
#    #'waveunet_v1': WaveUNet.from_pretrained('../../../workspace/models/waveunet_model_adapt.pt'),
#    #'dcunet_20': DCUNet.from_pretrained('../../../workspace/models/dcunet_20_random_v2.pt'),
#    #'dccrn': DCCRNet.from_pretrained('../../../workspace/models/dccrn_random_v1.pt'),
#    'dprnn': DPRNNTasNet.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/DRONE_project/asteroid/notebooks/dprnn_model.pt'),
#    'conv_tasnet': ConvTasNet.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/DRONE_project/asteroid/notebooks/convtasnet_model.pt'),
#    'dptnet': DPTNet.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/DRONE_project/asteroid/notebooks/dptnet_model.pt'),
#}
#
models = {
    'input': None,
    'baseline': RegressionFCNN.from_pretrained('models/baseline_model_v1.pt'),
    'waveunet_v1': WaveUNet.from_pretrained('models/waveunet_model_adapt.pt'),
    'dcunet_20': DCUNet.from_pretrained('models/dcunet_20_random_v2.pt'),
    'dccrn': DCCRNet.from_pretrained('models/dccrn_random_v1.pt'),
    'smolnet': SMoLnet.from_pretrained('models/SMoLnet.pt'),
    'dprnn': DPRNNTasNet.from_pretrained('models/dprnn_model.pt'),
    'conv_tasnet': ConvTasNet.from_pretrained('models/convtasnet_model.pt'),
    'dptnet': DPTNet.from_pretrained('models/dptnet_model.pt'),
    'demucs': Demucs.from_pretrained('models/Demucs.pt'),
}

def eval_all_and_plot(models, test_set, directory, plot_name):
    results_dfs = {}

    for model_name, model in models.items():
        print(f'Evaluating {model_labels[model_name]}')
        csv_path = f'/jmain01/home/JAD007/txk02/aaa18-txk02/DRONE_project/asteroid/notebooks/{directory}/{model_name}.csv'

        if os.path.isfile(csv_path):
            print('Results already available')
            df = pd.read_csv(csv_path)
        else:
            df = evaluate_model(model, test_set)
            df.to_csv(csv_path, index=False)

        results_dfs[model_name] = df

    plot_results(results_dfs, figsize=(14, 8), plot_name=plot_name)
    return avg_results_table(results_dfs, models)

#eval_all_and_plot(models, timit_test, 'eval_results_new', 'results_new_v2.pdf')
eval_all_and_plot(models, timit_test_small, 'eval_results_new', 'results_small_v2.pdf')
