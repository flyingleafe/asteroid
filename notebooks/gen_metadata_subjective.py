import os
import glob
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

csv_path_dict = {
 '-30': '/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/test_noisy_drone_cache/-30db/test-drones_68.csv',
 '-25': '/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/test_noisy_drone_cache/-25db/test-drones_69.csv',
 '-20': '/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/test_noisy_drone_cache/-20db/test-drones_70.csv',
 '-15': '/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/test_noisy_drone_cache/-15db/test-drones_71.csv',
 '-10': '/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/test_noisy_drone_cache/-10db/test-drones_72.csv',
 '-5': '/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/test_noisy_drone_cache/-5db/test-drones_73.csv',
  '0': '/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/test_noisy_drone_cache/0db/test-drones_74.csv',
 }

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


#directories to save denoised audio in
save_enhanced_dir = "/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/drone_noise_out/"

# get csvfilenames for metadata information 

def get_all_metrics_from_model(model, test_sets, model_name=None):    
    series_list = []
    torch.no_grad().__enter__()
    model = model.cuda()
    for snr, test_set in test_sets.items():
            # makde dirs for each models and separate dir for each snr
            os.makedirs(f'{save_enhanced_dir}/{str(model_name)}/{snr}dB/data/', exist_ok=True)
            denoised_file_paths = []
            print(f'SNR: {snr}db')
            loader = DataLoader(test_set, num_workers=0)

            for i, (mix, clean, path) in tqdm(enumerate(loader)):
                mix = mix.cuda()
                estimate = model(mix).detach().flatten().cpu().numpy()

                denoised_file_name = path[0].split('/')[-1]
                #add a "_" in front of the denoised fie
                denoised_file_path = f'{save_enhanced_dir}/{str(model_name)}/{snr}dB/data/{model_name}_{denoised_file_name}'
                denoised_file_paths.append(denoised_file_path)
                sf.write(denoised_file_path, estimate, samplerate=SAMPLE_RATE)

                #Dont calculate metric just save separated plus, meta data
                #metrics_dict = get_metrics(mix.cpu().numpy(), clean.numpy(), estimate, sample_rate=SAMPLE_RATE, metrics_list=["pesq"])
                #metrics_dict["mix_path"] = path
                #metrics_dict["snr"] = snr
                #series_list.append(pd.Series(metrics_dict))
                #all_metrics_df = pd.DataFrame(series_list)             
                if i == 5 : break

            csv_path_tmp = csv_path_dict[str(snr)]
            df = pd.read_csv(csv_path_tmp)
            denoised_file_paths = pd.Series(denoised_file_paths)
            df['denoised_path'] = denoised_file_paths
            df_csv_path = f'{save_enhanced_dir}/{str(model_name)}/{snr}dB/{model_name}_snr{snr}dB.csv'
            df.to_csv(df_csv_path)
    return None


#directory to store evaluation results in
#os.makedirs('evaluation', exist_ok=True)

from asteroid import DPTNet, SMoLnet, RegressionFCNN, DCUNet, WaveUNet
baseline_model = RegressionFCNN.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/Drone_Models_selected/baseline_model_v1.pt')
smolnet_model = SMoLnet.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/Drone_Models_selected/SMoLnet.pt')
dcunet_model = DCUNet.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/Drone_Models_selected/dcunet_20_random_v2.pt')
dptnet_model = DPTNet.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/Drone_Models_selected/dptnet_model.pt')
waveunet_model = WaveUNet.from_pretrained('/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/Drone_Models_selected/waveunet_model_adapt.pt')

print('get metrics for DPTNet')
get_all_metrics_from_model(model=dptnet_model, test_sets=test_sets, model_name='DPTNet')

print('get metrics for Regression model')
get_all_metrics_from_model(model=baseline_model, test_sets=test_sets, model_name='RegressionFCNN')

print('get metrics for SMoLnet model')
get_all_metrics_from_model(model=smolnet_model, test_sets=test_sets, model_name='SMoLnet')

print('get metrics for DCUNet model')
get_all_metrics_from_model(model=dcunet_model, test_sets=test_sets, model_name='DCUNet')

print('get metrics for WaveUNet model')
get_all_metrics_from_model(model=waveunet_model, test_sets=test_sets, model_name='WaveUNet')


'''
#print('get metrics for DPRNN')
#dprnn_metrics = get_all_metrics_from_model(model=dprnn_model, test_sets=test_sets)
#dprnn_metrics.to_csv('evaluation/pesq_dprnn.csv')

pesqs = {}
pesqs['dptnet'] = dpt_metrics.groupby('snr').agg({'pesq': ['mean']})['pesq']['mean']
pesqs['input'] = dpt_metrics.groupby('snr').agg({'input_pesq': ['mean']})['input_pesq']['mean']

breakpoint()
print('pesq: Input, {}'.format(pesqs['input']))
print('pesq: ConvTasNet, {}'.format(pesqs['dptnet']))

for model_name, series in pesqs.items():
    line_kwargs = {'marker': 'o'}
    if model_name == 'input':
        line_kwargs = {'c': 'black', 'ls': '--'}
    plt.plot(series.index, series, label=model_name, **line_kwargs)
    plt.legend()
    plt.grid(which='both')
    plt.savefig('results.png')
'''



