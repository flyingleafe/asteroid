import torch
import numpy as np
import pandas as pd
import librosa as lr
import soundfile as sf
import matplotlib.pyplot as plt



from torch.utils.data import DataLoader, ConcatDataset, random_split
from asteroid.data import TimitDataset, TimitLegacyDataset
from asteroid.data.utils import CachedWavSet
from tqdm import tqdm

from torch import optim
from pytorch_lightning import Trainer, loggers as pl_loggers
from asteroid_filterbanks.transforms import mag

from asteroid.losses import singlesrc_neg_sisdr
from asteroid.data.utils import find_audio_files


from asteroid.metrics import get_metrics

TIMIT_CACHE_DIR = '/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/test_noisy_drone_cache'
TIMIT_DIR_8kHZ  = '/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/TIMIT_8kHZ'
SAMPLE_RATE     = 8000

noises = CachedWavSet('/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/noises-test-drones', sample_rate=SAMPLE_RATE, precache=True, with_path=True)

test_snrs = [-30, -25, -20, -15, -10, -5, 0]
test_sets = {}

i = 0
for snr in tqdm(test_snrs, 'Load datasets'):
    test_sets[snr] = TimitLegacyDataset(
        TIMIT_DIR_8kHZ, noises, sample_rate=SAMPLE_RATE,
        cache_dir=TIMIT_CACHE_DIR, snr=snr, dset_name='test-drones',
        subset='test', random_seed=68 + i, with_path=True)
    i += 1

from asteroid import ConvTasNet, DPRNNTasNet
dprnn_model = DPRNNTasNet.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/DRONE_project/asteroid/notebooks/dprnn_model.pt')
convtasnet_model = ConvTasNet.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/DRONE_project/asteroid/notebooks/convtasnet_model.pt')

def get_all_metrics_from_model(model, test_sets):    
    series_list = []
    torch.no_grad().__enter__()
    model = model.cuda()
    for snr, test_set in test_sets.items():
            print(f'SNR: {snr}db')
            loader = DataLoader(test_set, num_workers=0)

            for i, (mix, clean, path) in tqdm(enumerate(loader)):
                mix = mix.cuda()
                estimate = model(mix).detach().flatten().cpu().numpy()
                metrics_dict = get_metrics(mix.cpu().numpy(), clean.numpy(), estimate, sample_rate=SAMPLE_RATE, metrics_list=["pesq"])
                metrics_dict["mix_path"] = path
                metrics_dict["snr"] = snr
                series_list.append(pd.Series(metrics_dict))
                all_metrics_df = pd.DataFrame(series_list)             
                if i == 500: break
    return all_metrics_df

print('get metrics for ConvTasNet')
ct_metrics = get_all_metrics_from_model(model=convtasnet_model, test_sets=test_sets)
ct_metrics.to_csv('evaluation/pesq_convtasnet.csv')
print('get metrics for DPRNN')
dprnn_metrics = get_all_metrics_from_model(model=dprnn_model, test_sets=test_sets)
dprnn_metrics.to_csv('evaluation/pesq_dprnn.csv')

pesqs = {}
pesqs['convtasnet'] = ct_metrics.groupby('snr').agg({'pesq': ['mean']})['pesq']['mean']
pesqs['dprnn'] = dprnn_metrics.groupby('snr').agg({'pesq': ['mean']})['pesq']['mean']
pesqs['input'] = dprnn_metrics.groupby('snr').agg({'input_pesq': ['mean']})['input_pesq']['mean']

print('pesq: Input, {}'.format(pesqs['input']))
print('pesq: ConvTasNet, {}'.format(pesqs['convtasnet']))
print('pesq: DPRNN, {}'.format(pesqs['dprnn']))


for model_name, series in pesqs.items():
    line_kwargs = {'marker': 'o'}
    if model_name == 'input':
        line_kwargs = {'c': 'black', 'ls': '--'}
    plt.plot(series.index, series, label=model_name, **line_kwargs)
    plt.legend()
    plt.grid(which='both')
    plt.savefig('results.png')




