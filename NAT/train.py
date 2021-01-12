import os
from torch.utils.data import DataLoader
import torch
from librimix_dataset_trip import LibriMixTrip
import pytorch_lightning as pl
from nat_model import Model
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from convtasnet_base import ConvTasNet

# Train config
# regular training = False -> Noise aware training
regular_training = False
batch_size = 8


model = ConvTasNet(n_src=2, regular_training=regular_training)

# Load dataset object -> o/p should be - (mix, sources and noise)
train_data = LibriMixTrip(
    csv_dir="/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/asteroid/egs/librimix/ConvTasNet/data/wav8k/min/train-100",
    task="sep_noisy",
)
valid_data = LibriMixTrip(
    csv_dir="/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/asteroid/egs/librimix/ConvTasNet/data/wav8k/min/dev",
    task="sep_noisy",
)

## make datalaoder
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

# sample training data for overfit test
mix, target, noise = next(iter(train_loader))
sample = [mix, target, noise]
sample = False  # sample=sample to overfit

# callbacks
callbacks = []
checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=5, verbose=True)
callbacks.append(checkpoint)
callbacks.append(
    EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True)
)


# gpu if any
gpus = -1 if torch.cuda.is_available() else None
distributed_backend = "ddp" if torch.cuda.is_available() else "dp"

# define lightning model and innit trainer and train
model = Model(model=model, sample=sample, plot=False)
trainer = pl.Trainer(
    gpus=gpus,
    gradient_clip_val=5.0,
    callbacks=callbacks,
    distributed_backend=distributed_backend,
)
trainer.fit(model, train_loader, valid_loader)
