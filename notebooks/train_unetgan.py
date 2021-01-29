import torch
import numpy as np
import pandas as pd
import librosa as lr
import matplotlib.pyplot as plt

from librosa import display as lrd
import IPython.display as ipd

from torch.utils.data import DataLoader, ConcatDataset, random_split
from asteroid.data import TimitDataset
from tqdm import tqdm

from asteroid.data.utils import find_audio_files, cut_or_pad

from torch import optim
from pytorch_lightning import Trainer, loggers as pl_loggers
from asteroid.engine import System
from asteroid.losses import singlesrc_neg_sisdr

from asteroid.engine.system import UNetGAN

def sisdr_loss_wrapper(est_target, target):
    return singlesrc_neg_sisdr(est_target.squeeze(1), target).mean()

def train_val_split(ds, val_fraction=0.1, random_seed=42):
    assert val_fraction > 0 and val_fraction < 0.5
    len_train = int(len(ds) * (1 - val_fraction))
    len_val = len(ds) - len_train
    return random_split(ds, [len_train, len_val], generator=torch.Generator().manual_seed(random_seed))


def main():
    TIMIT_CACHE_DIR = '/import/vision-eddydata/dm005_tmp/mixed_wavs_asteroid'
    train_snrs = [-25, -20, -15, -10, -5, 0, 5, 10, 15]
    test_snrs = [-30, -25, -20, -15, -10, -5, 0]
    
    timit_train_misc = TimitDataset.load_with_cache(
        '../../../datasets/TIMIT', '../../../datasets/noises-train',
        cache_dir=TIMIT_CACHE_DIR, snrs=train_snrs, root_seed=42, prefetch_mixtures=False,
        dset_name='train-misc', subset='train', track_duration=48000)
    
    train_set, val_set = train_val_split(timit_train_misc)
    
    BATCH_SIZE = 64
    NUM_WORKERS = 10

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    
    unetgan_module = UNetGAN(lr_g=2e-4, lr_d=2e-4)
    
    trainer = Trainer(max_epochs=102, gpus=-1, accelerator='dp',
                      resume_from_checkpoint='logs/lightning_logs/version_0/checkpoints/epoch=99-step=395190.ckpt')
    trainer.fit(unetgan_module, train_loader, val_loader)
    
    
if __name__ == "__main__":
    main()