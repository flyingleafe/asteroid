import numpy as np
import matplotlib
import os
import soundfile as sf
import torch.nn as nn
import pytorch_lightning as pl
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
import torch
import matplotlib.pyplot as plt

loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
normalize = matplotlib.colors.Normalize(vmin=-1, vmax=1)

fs = 8000
device = "cuda"


class Model(pl.LightningModule):
    def __init__(self, model, sample=None, plot=False):
        super().__init__()
        self.model = model
        self.sample = sample
        self.plot = plot
        self.dev = device

    def forward(self, x):
        mix, _, noise = x
        mix = mix.to(device=self.dev)
        noise = noise.to(device=self.dev)

        estimates = self.model(mix, noise)

        if self.plot:
            print("plotting")
            if self.global_step % 50 == 0:
                example = estimates.cpu().detach().numpy()
                self.save_audio_and_plots(example)
        return estimates

    def training_step(self, x, idx):
        if self.sample:
            # print('overfitting test')
            output = self(self.sample)
            self.sample[1] = self.sample[1].to(device=self.dev)
            loss = loss_func(output, self.sample[1])
            print("loss: ", loss.item())
        else:
            mix, target, noise = x
            output = self(x)
            loss = loss_func(output, target)
            # print("loss: ", loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def save_audio_and_plots(self, x):
        os.makedirs("waveplots/", exist_ok=True)
        os.makedirs("save_audio/", exist_ok=True)
        s1 = x[0][0]
        s2 = x[0][1]

        # plots
        plt.clf()
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(s1)
        plt.subplot(1, 2, 2)
        plt.plot(s2)
        plt.savefig("waveplots/output_{}.png".format(self.global_step))

        # save audio
        sf.write("save_audio/s1_{}.wav".format(self.global_step), s1, samplerate=fs)
        sf.write("save_audio/s2_{}.wav".format(self.global_step), s2, samplerate=fs)
