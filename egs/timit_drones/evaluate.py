import hydra
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.multiprocessing import Process, Queue, cpu_count
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import DataLoader
from functools import partial
from queue import Empty
from tqdm import trange, tqdm
from pypesq import pesq
from pystoi import stoi

from asteroid.models import BaseModel
from asteroid.metrics import get_metrics

def _eval(batch, metrics, including='output', sample_rate=8000, use_pypesq=False):
    mix, clean, estimate, snr = batch
    if use_pypesq:
        metrics = [m for m in metrics if m != 'pesq']
        
    has_estoi = False
    if 'estoi' in metrics:
        metrics = [m for m in metrics if m != 'estoi']
        has_estoi = True
    
    res = get_metrics(mix.numpy(), clean.numpy(), estimate.numpy(),
                      sample_rate=sample_rate, metrics_list=metrics, including=including)
    
    if use_pypesq:
        res['pesq'] = pesq(clean.flatten(), estimate.flatten(), sample_rate)
        
    if has_estoi:
        res['estoi'] = stoi(clean.flatten(), estimate.flatten(), sample_rate, extended=True)
    
    if including == 'input':
        for m in metrics:
            res[m] = res['input_'+m]
            del res['input_'+m]
            
    res['snr'] = snr[0].item()
    return res


def data_feed_process(queue, signal_queue, model, test_set):
    loader = DataLoader(test_set, num_workers=2)
    with_cuda = torch.cuda.is_available()
    
    if model is not None:
        if with_cuda:
            model = model.cuda()
        model.eval()
    
    def model_run(mix):
        if model is None:
            return mix
        else:
            return model(mix.cuda()).squeeze(1).detach().cpu()
        
    with torch.no_grad():
        for ix, (mix, clean, snr) in enumerate(loader):
            enh = model_run(mix)
            queue.put((ix, (mix, clean, enh, snr)))

    if model is not None and with_cuda:
        model = model.cpu()
            
    # wait for a signal to end the process
    signal_queue.get()

    
def eval_process(proc_idx, input_queue, output_queue, **kwargs):
    while True:
        try:
            inp = input_queue.get()
            if inp is None:
                break
            else:
                ix, batch = inp
                res = _eval(batch, **kwargs)
                output_queue.put((ix, res))
        except Empty:
            time.sleep(0.1)
            

def evaluate_model(model, test_set, num_workers=None, metrics=['pesq', 'stoi', 'si_sdr'],
                   sample_rate=8000, max_queue_size=100, use_pypesq=False, use_file_sharing=True):
    
    if use_file_sharing:
        torch.multiprocessing.set_sharing_strategy('file_system')
    
    if num_workers is None:
        num_workers = cpu_count()
    
    ds_len = len(test_set)
    df = pd.DataFrame(columns=['snr']+metrics, index=np.arange(ds_len))
    including = 'input' if model is None else 'output'
    
    signal_queue = Queue()
    input_queue = Queue(maxsize=max_queue_size)
    output_queue = Queue(maxsize=max_queue_size)
    
    try:
        feed_pr = Process(target=data_feed_process, args=(input_queue, signal_queue, model, test_set))
        feed_pr.start()

        eval_prs = []
        for i in range(num_workers-1):
            pr = Process(target=eval_process, args=(i, input_queue, output_queue), kwargs={
                'metrics': metrics,
                'including': including,
                'sample_rate': sample_rate,
                'use_pypesq': use_pypesq,
            })
            pr.start()
            eval_prs.append(pr)

        for i in tqdm(range(ds_len), 'Evaluating and calculating scores'):
            ix, res = output_queue.get()
            row = pd.Series(res)
            df.loc[ix] = row
    
    except Exception as err:
        raise err
        
    finally:    
        signal_queue.put(None)
        for pr in eval_prs:
            input_queue.put(None)

        feed_pr.join()
        for pr in eval_prs:
            pr.join()

    return df
    

def evaluate_input(*args, **kwargs):
    return evaluate_model(None, *args, **kwargs)

    
metrics_names = {
    'pesq': 'PESQ',
    'stoi': 'STOI',
    'estoi': 'ESTOI',
    'si_sdr': 'SI-SDR',
}

model_labels = {
    'input': 'Input',
    'baseline': 'Baseline DNN',
    'baseline_1024': 'Baseline DNN v2',
    'baseline_v2': 'Baseline DNN (L1 loss)',
    'baseline_proper_mse': 'Baseline DNN (fixed test set)',
    'baseline_sigmoid': 'Baseline DNN (sigm)',
    'baseline_sigmoid_1024': 'Baseline DNN v2 (sigm)',
    'waveunet_v1': 'Wave-U-Net',
    'dcunet_20': 'DCUNet-20',
    'dccrn': 'DCCRN',
    'dccrn_1024': 'DCCRN v2',
    'smolnet': 'SMoLnet',
    'smolnet_tms': 'SMoLnet-TMS',
    'smolnet_cirm': 'SMoLnet-cIRM',
    'smolnet_1024': 'SMoLnet v2',
    'dprnn': 'DPRNN',
    'conv_tasnet': 'Conv-TasNet',
    'dptnet': 'DPTNet',
    'demucs': 'Demucs',
    'demucs_full': 'Demucs (full)',
    'unetgan': 'UNetGAN',
    'unetgan-nogan': 'UNetGAN generator (MSE loss only)',
    'phasen': 'PHASEN',
}

def aggregate_results(dfs, metrics=['pesq', 'stoi', 'si_sdr']):
    return {
        name: df.groupby('snr').agg({
            metric: ['mean', 'std', 'count'] for metric in metrics
        })
        for name, df in dfs.items()
    }
    

def plot_results(dfs, figsize=(15, 5), metrics=['pesq', 'stoi', 'si_sdr'],
                 plot_name=None, legend='right', labels=None, lines={}, **kwargs): 
    
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=figsize)
    all_scores = aggregate_results(dfs, metrics)
    
    if labels is None:
        labels = model_labels
    
    for model_name, scores in all_scores.items():
        line_kwargs = lines.get(model_name, {'marker': '.', 'alpha': 0.8})
        fill_kwargs = {}
        if model_name == 'input':
            line_kwargs = {'c': 'black', 'ls': 'dotted'}
            fill_kwargs = {'color': 'black'}
        
        for i, metric in enumerate(metrics):
            plt.sca(axes[i])
            means = scores[metric]['mean']
            stds = scores[metric]['std'].values / np.sqrt(scores[metric]['count'].values) * 3
            xs = means.index
            plt.plot(xs, means, label=labels[model_name], **line_kwargs)
            plt.fill_between(xs, means - stds, means + stds, alpha=0.2, **fill_kwargs)
    
    for i, metric in enumerate(metrics):
        plt.sca(axes[i])
        plt.grid(which='both')
        plt.title(metrics_names[metric])
        plt.xlabel('SNR, dB')
        if legend == 'right' and i == len(metrics) - 1:
            plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        elif legend == 'top' and i == 0:
            plt.legend(bbox_to_anchor=(0, 1.05, 1, 0), loc="lower left", ncol=3)
    
    if plot_name is not None:
        plt.savefig(plot_name, bbox_inches='tight')
    
    plt.show()
    
    
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
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_approx = np.around(param_count / 1000000, decimals=2)
        else:
            param_approx = None

        total_df.loc[len(total_df)] = [model_labels[model_name], f'{param_approx}M'] + list(df[metrics].mean(axis=0))

    total_df = total_df.style.apply(highlight_max)
    return total_df

def avg_improvements_table(dfs, models, metrics=['pesq', 'stoi', 'si_sdr']):
    total_df = pd.DataFrame(columns=['Model', 'N. of params'] + [metrics_names[m] for m in metrics])
    
    input_avgs = dfs['input'][metrics].mean(axis=0)
    for model_name, df in dfs.items():
        model = models[model_name]
        if model is None:
            continue
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_approx = np.around(param_count / 1000000, decimals=2)
        
        avgs = df[metrics].mean(axis=0)
        improvements = np.round(((avgs - input_avgs) / np.abs(input_avgs)) * 100, 1)
        
        total_df.loc[len(total_df)] = [model_labels[model_name], f'{param_approx}M'] + list(improvements)
        
    total_df = total_df.style.apply(highlight_max)
    return total_df


def eval_all(models, test_set, directory, metrics=['pesq', 'stoi', 'si_sdr'], subset_ixs=None, **kwargs):
    results_dfs = {}
    os.makedirs(directory, exist_ok=True)
    
    orig_metrics = metrics
    
    for model_name, model in models.items():
        print(f'Evaluating {model_labels[model_name]}')
        csv_path = f'{directory}/{model_name}.csv'

        existing_df = None
        metrics = orig_metrics
        
        if os.path.isfile(csv_path):
            existing_df = pd.read_csv(csv_path)
            existing_metrics = [m for m in metrics if m in existing_df.columns]
            metrics = [m for m in metrics if m not in existing_df.columns]
            print('Metrics already calculated: ', existing_metrics)
            
        if len(metrics) == 0:
            df = existing_df
        else:
            df = evaluate_model(model, test_set, metrics=metrics, **kwargs)
            if existing_df is not None:
                df = df.merge(existing_df, how='outer')
                
            df.to_csv(csv_path, index=False)

        if subset_ixs is not None:
            df = df[subset_ixs]
            
        results_dfs[model_name] = df[['snr'] + orig_metrics]
        
    return results_dfs

def eval_all_and_plot(models, test_set, directory, plot_name=None, figsize=(14,8),
                      metrics=['pesq', 'stoi', 'si_sdr'], **kwargs):
    results_dfs = eval_all(models, test_set, directory, metrics=metrics, **kwargs)
    plot_results(results_dfs, metrics=metrics, figsize=figsize, plot_name=plot_name, **kwargs)
    return avg_improvements_table(results_dfs, models)


@hydra.main(config_path='conf', config_name='config')
def main(args):
    pass


if __name__ == "__main__":
    main()