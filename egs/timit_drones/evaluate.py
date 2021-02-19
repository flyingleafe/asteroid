import torch
import numpy as np
import pandas as pd

from torch.multiprocessing import Process, Queue, cpu_count
from torch.utils.data import DataLoader
from functools import partial
from queue import Empty
from tqdm import trange, tqdm

from asteroid.metrics import get_metrics

def _eval(batch, metrics, including='output', sample_rate=8000):
    mix, clean, estimate, snr = batch
    res = get_metrics(mix.numpy(), clean.numpy(), estimate.numpy(),
                          sample_rate=sample_rate, metrics_list=metrics, including=including)
    
    if including == 'input':
        for m in metrics:
            res[m] = res['input_'+m]
            del res['input_'+m]
            
    res['snr'] = snr[0].item()
    return res


def data_feed_process(queue, signal_queue, model, test_set):
    loader = DataLoader(test_set, num_workers=2)
    
    if model is not None:
        model = model.cuda()
        model.eval()
    
    def model_run(mix):
        if model is None:
            return mix
        else:
            return model(mix.cuda()).squeeze(1).detach().cpu()
        
    for mix, clean, snr in loader:
        enh = model_run(mix)
        queue.put((mix, clean, enh, snr))
        
    # wait for a signal to end the process
    signal_queue.get()

    
def eval_process(proc_idx, input_queue, output_queue, **kwargs):
    while True:
        try:
            batch = input_queue.get()
            if batch is None:
                break
            else:
                res = _eval(batch, **kwargs)
                output_queue.put(res)
        except Empty:
            time.sleep(0.1)
            

def evaluate_model(model, test_set, num_workers=None, metrics=['pesq', 'stoi', 'si_sdr'],
                   sample_rate=8000, max_queue_size=100):
    
    if num_workers is None:
        num_workers = cpu_count()
    
    df = pd.DataFrame(columns=['snr']+metrics)
    ds_len = len(test_set)
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
                'sample_rate': sample_rate
            })
            pr.start()
            eval_prs.append(pr)

        for i in tqdm(range(ds_len), 'Evaluating and calculating scores'):
            res = output_queue.get()
            df = df.append(res, ignore_index=True)
    
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