import torch
import os
import os.path
import shutil
import numpy as np
import soundfile as sf

from pathlib import PurePath
from torch.utils.data import DataLoader, random_split
from asteroid.data import TimitDataset, TimitCleanDataset, RandomMixtureDataset
from tqdm import tqdm

from torch import optim
from pytorch_lightning import Trainer, seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from asteroid_filterbanks.transforms import mag
from asteroid.engine import System
from asteroid.losses import singlesrc_neg_sisdr

from asteroid import DCUNet, DCCRNet


BATCH_SIZE       = 64     # could be more on cluster, test if larger one work
SAMPLE_RATE      = 8000   # as agreed upon
#CROP_LEN         = 8192   # slightly more than a second, guaranteed to be less than the shortest clip in TIMIT
CROP_LEN         = 24000  # average track len in TIMIT
SEED             = 42     # magic number :)

# directory to cache fixed mixtures
TIMIT_CACHE_DIR = '/import/vision-eddydata/dm005_tmp/mixed_wavs_asteroid2'
# directory with train noises (n116-n120)
DRONE_NOISE_DIR = '../../../datasets/noises-train-drones'
# fixed SNRs for validation set
TRAIN_SNRS = [-25, -20, -15, -10, -5]

def sisdr_loss_wrapper(est_target, target):
    return singlesrc_neg_sisdr(est_target.squeeze(1), target).mean()

def train_val_split(ds, val_fraction=0.1, random_seed=SEED):
    assert val_fraction > 0 and val_fraction < 0.5
    len_train = int(len(ds) * (1 - val_fraction))
    len_val = len(ds) - len_train
    return random_split(ds, [len_train, len_val], generator=torch.Generator().manual_seed(random_seed))


#Resample TIMIT dataset
TIMIT_DIR = PurePath('../../../datasets/TIMIT')
TIMIT_DIR_8kHZ = PurePath('/import/vision-eddydata/dm005_tmp/TIMIT_8kHZ')


os.makedirs(TIMIT_DIR_8kHZ, exist_ok=True)
shutil.copyfile(TIMIT_DIR / 'train_data.csv', TIMIT_DIR_8kHZ / 'train_data.csv')
shutil.copyfile(TIMIT_DIR / 'test_data.csv', TIMIT_DIR_8kHZ / 'test_data.csv')

data_dir_in = TIMIT_DIR / 'data'
data_dir_out = TIMIT_DIR_8kHZ / 'data'

def resample(ds, dir_in, dir_out, message='Resampling'):
    dl = DataLoader(ds, num_workers=10)
    for wav, path in tqdm(dl, message):
        path = PurePath(path[0])
        out_path = dir_out / path.relative_to(dir_in)
        os.makedirs(out_path.parent, exist_ok=True)
        sf.write(file=out_path, data=wav[0].numpy(), samplerate=SAMPLE_RATE)

timit_train = TimitCleanDataset(TIMIT_DIR, subset='train', sample_rate=SAMPLE_RATE)
resample(timit_train, data_dir_in, data_dir_out, 'Resampling training data')

timit_test = TimitCleanDataset(TIMIT_DIR, subset='test', sample_rate=SAMPLE_RATE)
resample(timit_test, data_dir_in, data_dir_out, 'Resampling test data')


# Reproducibility - fix all random seeds
seed_everything(SEED)

timit_train_drones = TimitDataset.load_with_cache(
    TIMIT_DIR_8kHZ, DRONE_NOISE_DIR,
    cache_dir=TIMIT_CACHE_DIR, snrs=TRAIN_SNRS, root_seed=SEED,
    mixtures_per_clean=5, dset_name='train-drones', sample_rate=SAMPLE_RATE,
    subset='train', crop_length=CROP_LEN)

timit_train, timit_val = train_val_split(timit_train_drones, val_fraction=0.1, random_seed=SEED)

NUM_WORKERS = 5
train_loader = DataLoader(timit_train, shuffle=True, batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS, drop_last=True)
val_loader = DataLoader(timit_val, batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS, drop_last=True)

# some random parameters, does it look sensible?
LR = 1e-3
REDUCE_LR_PATIENCE = 3
EARLY_STOP_PATIENCE = 10
MAX_EPOCHS = 500

# the model here should be constructed in the script accordingly to the passed config (including the model type)
# most of the models accept `sample_rate` parameter for encoders, which is important (default is 16000, override)
model = DCUNet("DCUNet-20", fix_length_mode="trim", sample_rate=SAMPLE_RATE)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=REDUCE_LR_PATIENCE)
early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)

# Probably we also need to subclass `System`, in order to log the target metrics on the validation set (PESQ/STOI)
system = System(model, optimizer, sisdr_loss_wrapper, train_loader, val_loader, scheduler)


# log dir and model name are also part of the config, of course
LOG_DIR = 'logs'
logger = pl_loggers.TensorBoardLogger(LOG_DIR, name='TIMIT-drones-DCUNET-20-proper', version=1)

# choose the proper accelerator for JADE, probably `ddp` (also, `auto_select_gpus=True` might be useful)
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=-1, accelerator='dp',
                  logger=logger, callbacks=[early_stopping], deterministic=True)

trainer.fit(system)

torch.save(model.serialize(), 'dcunet_20_proper_v1.pt')






