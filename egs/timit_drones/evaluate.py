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
from asteroid.data import TimitDataset
from torch.utils.data import DataLoader, random_split, Subset
from asteroid.data.utils import CachedWavSet, FixedMixtureSet

from asteroid.models import BaseModel
from asteroid.metrics import get_metrics
from  asteroid.masknn import MCEM_algo, VAE_Decoder_Eval
import torch.nn as nn

def _jitable_shape(tensor):
    """Gets shape of ``tensor`` as ``torch.Tensor`` type for jit compiler
    .. note::
        Returning ``tensor.shape`` of ``tensor.size()`` directly is not torchscript
        compatible as return type would not be supported.
    Args:
        tensor (torch.Tensor): Tensor
    Returns:
        torch.Tensor: Shape of ``tensor``
    """
    return torch.tensor(tensor.shape)

def _pad_x_to_y(x: torch.Tensor, y: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Right-pad or right-trim first argument to have same size as second argument
    Args:
        x (torch.Tensor): Tensor to be padded.
        y (torch.Tensor): Tensor to pad `x` to.
        axis (int): Axis to pad on.
    Returns:
        torch.Tensor, `x` padded to match `y`'s shape.
    """
    if axis != -1:
        raise NotImplementedError
    inp_len = y.shape[axis]
    output_len = x.shape[axis]
    return nn.functional.pad(x, [0, inp_len - output_len])

def _shape_reconstructed(reconstructed, size):
    """Reshape `reconstructed` to have same size as `size`
    Args:
        reconstructed (torch.Tensor): Reconstructed waveform
        size (torch.Tensor): Size of desired waveform
    Returns:
        torch.Tensor: Reshaped waveform
    """
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed

#%% MCEM algorithm parameters
niter_MCEM = 10 # number of iterations for the MCEM algorithm - CHNAGE THIS TO 100
niter_MH = 40 # total number of samples for the Metropolis-Hastings algorithm
burnin = 30 # number of initial samples to be discarded
var_MH = 0.01 # variance of the proposal distribution
tol = 1e-5 # tolerance for stopping the MCEM iterations

def evaluate_vae(vae_model, decoder, mix):
    shape = _jitable_shape(mix)
    X = vae_model.encoder(mix.cuda()).cpu().numpy()
    X = X[0]

    F, N = X.shape
    g0 = np.ones((1,N))
    g_tensor0 = torch.from_numpy(g0.astype(np.float32))


    #Intermediate representaion
    _, z, _ = vae_model.forward_masker(torch.Tensor(X[None]).cuda())
    z = z[0].cuda().T

    Z_init = z.cpu().numpy()

    eps = np.finfo(float).eps
    K_b = 10 # NMF rank for noise model
    W0 = np.maximum(np.random.rand(F,K_b), eps)
    H0 = np.maximum(np.random.rand(K_b,N), eps)
    V_b0 = W0@H0
    V_b_tensor0 = torch.from_numpy(V_b0.astype(np.float32))
    # All-ones initialization of the gain parameters
    g0 = np.ones((1,N))
    g_tensor0 = torch.from_numpy(g0.astype(np.float32))


    mcem_algo = MCEM_algo(X=X, W=W0, H=H0, Z=Z_init, decoder=decoder,
              niter_MCEM=niter_MCEM, niter_MH=niter_MH, burnin=burnin,
              var_MH=var_MH)

    wlen_sec=64e-3
    fs=8000
    wlen = int(wlen_sec*fs) # window length of 64 ms

    hop = np.int(wlen//2) # hop size
    win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

    cost, niter_final = mcem_algo.run(hop=hop, wlen=wlen, win=win)
    # Separate the sources from the estimated parameters
    mcem_algo.separate(niter_MH=100, burnin=75)
    estimate = vae_model.forward_decoder(torch.Tensor(mcem_algo.S_hat).cuda())
    estimate= _pad_x_to_y(estimate, mix)
    estimate = _shape_reconstructed(estimate, shape)
    return estimate 
    

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
        if model.__class__.__name__ == 'VAE':
            decoder = VAE_Decoder_Eval(model)
            return evaluate_vae(model, decoder, mix).detach().cpu()
        else:
            return model(mix.cuda()).squeeze(1).detach().cpu()
        
    with torch.no_grad():
        for ix, (mix, clean, snr) in enumerate(loader):
            print('MODEL NAME: ', model.__class__.__name__)
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
            

def evaluate_model(model, test_set, num_workers=None, metrics=['pesq', 'estoi', 'si_sdr'],
                   sample_rate=8000, max_queue_size=100, use_pypesq=False, use_file_sharing=True, **kwargs):
    
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
    'vae': 'VAE',
    'waveunet_v1': 'Wave-U-Net',
    'dcunet_20': 'DCUNet-20',
    'dccrn': 'DCCRN',
    'dccrn_1024': 'DCCRN v2',
    'smolnet': 'SMoLnet-TCS',
    'smolnet_tms': 'SMoLnet-TMS',
    'smolnet_cirm': 'SMoLnet-cIRM',
    'smolnet_1024': 'SMoLnet v2',
    'dprnn': 'DPRNN',
    'conv_tasnet': 'Conv-TasNet',
    'dptnet': 'DPTNet',
    'demucs': 'Demucs',
    'demucs_full': 'Demucs (full)',
    'segan': 'SEGAN',
    'segan-nogan': 'SEGAN (L1 loss only)',
    'unetgan': 'UNetGAN',
    'unetgan-nogan': 'UNetGAN (MSE loss only)',
    'phasen': 'PHASEN',
    'phasen_1024': 'PHASEN v2',
}


main_models = ['baseline', 'vae', 'smolnet', 'smolnet_tms', 'smolnet_cirm',
               'dcunet_20', 'dccrn', 'phasen', 'waveunet_v1', 'demucs',
               'conv_tasnet', 'dprnn', 'dptnet', 'segan', 'unetgan']

kelly_colors = dict(
    vivid_yellow=(255, 179, 0),
    strong_purple=(128, 62, 117),
    vivid_orange=(255, 104, 0),
    very_light_blue=(166, 189, 215),
    vivid_red=(193, 0, 32),
    grayish_yellow=(206, 162, 98),
    #medium_gray=(129, 112, 102),

    # these aren't good for people with defective color vision:
    vivid_green=(0, 125, 52),
    strong_purplish_pink=(246, 118, 142),
    strong_blue=(0, 83, 138),
    strong_yellowish_pink=(255, 122, 92),
    strong_violet=(83, 55, 122),
    vivid_orange_yellow=(255, 142, 0),
    strong_purplish_red=(179, 40, 81),
    vivid_greenish_yellow=(244, 200, 0),
    strong_reddish_brown=(127, 24, 13),
    vivid_yellowish_green=(147, 170, 0),
    deep_yellowish_brown=(89, 51, 21),
    vivid_reddish_orange=(241, 58, 19),
    dark_olive_green=(35, 44, 22))

model_colors = {m: (r/255, g/255, b/255) for m, (r, g, b) in zip(main_models, kelly_colors.values())}

def aggregate_results(dfs, metrics=['pesq', 'estoi', 'si_sdr']):
    return {
        name: df.groupby('snr').agg({
            metric: ['mean', 'std', 'count'] for metric in metrics
        })
        for name, df in dfs.items()
    }
    

def plot_results(dfs, figsize=(15, 5), metrics=['pesq', 'estoi', 'si_sdr'],
                 plot_name=None, legend='right', legend_ncol=3, legend_pad=0.08, legend_pad_left=0.0,
                 ax_bgcol=None, labels=None, lines={}, **kwargs): 
    
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=figsize)
    all_scores = aggregate_results(dfs, metrics)
    
    if labels is None:
        labels = model_labels
    
    for model_name, scores in all_scores.items():
        line_kwargs = lines.get(model_name, {'marker': '.', 'alpha': 0.8})
        fill_kwargs = {}
        if model_name in model_colors:
            line_kwargs['color'] = model_colors[model_name]
            fill_kwargs['color'] = model_colors[model_name]
            
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
        if ax_bgcol is not None:
            axes[i].set_facecolor(ax_bgcol)
        
        plt.sca(axes[i])
        plt.grid(which='both')
        plt.title(metrics_names[metric])
        plt.xlabel('SNR, dB')
        
    fig.tight_layout()
        
    legend_kwargs = {}
    if ax_bgcol is not None:
        legend_kwargs['facecolor'] = ax_bgcol
        
    if legend == 'right':
        plt.sca(axes[len(metrics) - 1])
        plt.legend(bbox_to_anchor=(1,1), loc="upper left", **legend_kwargs)
    elif legend == 'top':
        plt.sca(axes[0])
        plt.legend(bbox_to_anchor=(legend_pad_left, 1 + legend_pad, 1, 0), loc="lower left",
                   ncol=legend_ncol, **legend_kwargs)
                    
    if plot_name is not None:
        plt.savefig(plot_name, bbox_inches='tight')
    
    plt.show()
    
    
def highlight_max(s): 
    if s.dtype == np.object: 
        is_max = [False for _ in range(s.shape[0])] 
    else: 
        is_max = s == s.max() 
    return ['font-weight: bold' if cell else '' for cell in is_max] 


def avg_results_table(dfs, models, metrics=['pesq', 'estoi', 'si_sdr']):
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

def avg_improvements_table(dfs, models, metrics=['pesq', 'estoi', 'si_sdr']):
    total_df = pd.DataFrame(columns=['Model', 'N. of params'] + [metrics_names[m] for m in metrics])
    
    input_avgs = dfs['input'][metrics].mean(axis=0)
    for model_name, df in dfs.items():
        model = models[model_name]
        if model is None:
            continue
        
        if hasattr(model, 'generator'):
            params = model.generator.parameters()
        else:
            params = model.parameters()
            
        param_count = sum(p.numel() for p in params if p.requires_grad)
        param_approx = np.around(param_count / 1000000, decimals=2)
        
        avgs = df[metrics].mean(axis=0)
        improvements = np.round(((avgs - input_avgs) / np.abs(input_avgs)) * 100, 1)
        
        total_df.loc[len(total_df)] = [model_labels[model_name], f'{param_approx}M'] + list(improvements)
        
    total_df = total_df.style.apply(highlight_max)
    return total_df


def eval_all(models, test_set, directory, metrics=['pesq', 'estoi', 'si_sdr'], subset_ixs=None, **kwargs):
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
                df = pd.concat([existing_df, df[metrics]], axis=1)
                
            df.to_csv(csv_path, index=False)

        if subset_ixs is not None:
            df = df[subset_ixs]
            
        results_dfs[model_name] = df[['snr'] + orig_metrics]
        
    return results_dfs


def eval_all_and_plot(models, test_set, directory, plot_name=None, figsize=(14,8),
                      metrics=['pesq', 'estoi', 'si_sdr'], **kwargs):
    results_dfs = eval_all(models, test_set, directory, metrics=metrics, **kwargs)
    plot_results(results_dfs, metrics=metrics, figsize=figsize, plot_name=plot_name, **kwargs)
    return avg_improvements_table(results_dfs, models, metrics=metrics)


def time_evaluate_cpu_all(models, number=1, stft_only=False):
    orig_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.no_grad():
        inp = torch.rand(8192)
        times = {}
        for name, model in models.items():
            if model is None:
                continue
            
            print(f'Running {name}')
            model.eval()
            if not stft_only:
                times[name] = timeit.timeit('_ = model(inp)', number=number, globals=locals()) / number
            else:
                times[name] = timeit.timeit('_ = model.decoder(model.encoder(inp))',
                                            number=number, globals=locals()) / number

    torch.set_num_threads(orig_threads)
    return times

# def time_eval_cpu()


@hydra.main(config_path='conf', config_name='config')
def main(args):
    pass
    
if __name__ == "__main__":
    main()
