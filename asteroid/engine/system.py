import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import Namespace
from torch.optim.lr_scheduler import ReduceLROnPlateau
from asteroid.masknn import UNetGANGenerator, UNetGANDiscriminator
from asteroid.masknn.wavenet import apply_model_chunked
from collections import OrderedDict

from ..utils import flatten_dict, unsqueeze_to_3d


class System(pl.LightningModule):
    """Base class for deep learning systems.
    Contains a model, an optimizer, a loss function, training and validation
    dataloaders and learning rate scheduler.

    Args:
        model (torch.nn.Module): Instance of model.
        optimizer (torch.optim.Optimizer): Instance or list of optimizers.
        loss_func (callable): Loss function with signature
            (est_targets, targets).
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Instance, or list
            of learning rate schedulers. Also supports dict or list of dict as
            ``{"interval": "step", "scheduler": sched}`` where ``interval=="step"``
            for step-wise schedulers and ``interval=="epoch"`` for classical ones.
        config: Anything to be saved with the checkpoints during training.
            The config dictionary to re-instantiate the run for example.

    .. note:: By default, ``training_step`` (used by ``pytorch-lightning`` in the
        training loop) and ``validation_step`` (used for the validation loop)
        share ``common_step``. If you want different behavior for the training
        loop and the validation loop, overwrite both ``training_step`` and
        ``validation_step`` instead.

    For more info on its methods, properties and hooks, have a look at lightning's docs:
    https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#lightningmodule-api
    """

    default_monitor: str = "val_loss"

    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        # hparams will be logged to Tensorboard as text variables.
        # summary writer doesn't support None for now, convert to strings.
        # See https://github.com/pytorch/pytorch/issues/33140
        self.hparams = Namespace(**self.config_to_hparams(self.config))

    def forward(self, *args, **kwargs):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_nb, train=True):
        """Common forward step between training and validation.

        The function of this method is to unpack pool_slenthe data given by the loader,
        forward the batch through the model and compute the loss.
        Pytorch-lightning handles all the rest.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
            train (bool): Whether in training mode. Needed only if the training
                and validation steps are fundamentally different, otherwise,
                pytorch-lightning handles the usual differences.

        Returns:
            :class:`torch.Tensor` : The loss value on this batch.

        .. note::
            This is typically the method to overwrite when subclassing
            ``System``. If the training and validation steps are somehow
            different (except for ``loss.backward()`` and ``optimzer.step()``),
            the argument ``train`` can be used to switch behavior.
            Otherwise, ``training_step`` and ``validation_step`` can be overwriten.
        """
        inputs, targets = batch
        est_targets = self(inputs)
        loss = self.loss_func(est_targets, targets)
        return loss

    def training_step(self, batch, batch_nb):
        """Pass data through the model and compute the loss.

        Backprop is **not** performed (meaning PL will do it for you).

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            torch.Tensor, the value of the loss.
        """
        loss = self.common_step(batch, batch_nb, train=True)
        self.log("loss", loss, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        """Need to overwrite PL validation_step to do validation.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
        """
        loss = self.common_step(batch, batch_nb, train=False)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        """Log hp_metric to tensorboard for hparams selection."""
        hp_metric = self.trainer.callback_metrics.get("val_loss", None)
        if hp_metric is not None:
            self.trainer.logger.log_metrics({"hp_metric": hp_metric}, step=self.trainer.global_step)

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.Tensor(v)
        return dic

    
# class GANSystem()    

    
class UNetGAN(pl.LightningModule):

    def __init__(
        self,
        mse_weight: float = 20,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        training_phase = 0,
        good_d_loss = 0.0001,
        good_adv_loss = 0.001,
        d_loss_accum_steps = 20,
        disc_concat_type = 'channels',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.generator = UNetGANGenerator()
        self.discriminator = UNetGANDiscriminator(concat_type=disc_concat_type)
        self.training_phase = training_phase
        self.automatic_optimization = False
        self.good_d_loss = good_d_loss
        self.good_adv_loss = good_adv_loss
        self.d_loss_accum_steps = d_loss_accum_steps
        self.last_d_losses = []

    def forward(self, wav):
        wav = unsqueeze_to_3d(wav)
        if wav.shape[-1] == 16384:
            return self.generator(wav)
        else:
            return apply_model_chunked(self.generator, wav, 16384)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def compute_generator_loss(self, batch, batch_idx):
        mix, clean = batch
        mix = unsqueeze_to_3d(mix)
        clean = unsqueeze_to_3d(clean)
        
        batch_size = mix.shape[0]
        real = torch.ones((batch_size, 1, 1)).type_as(mix)
        enh = self.generator(mix)
    
        disc_vals = self.discriminator(mix, enh)
            
        mse_loss = F.mse_loss(clean, enh)
        adv_loss = self.adversarial_loss(disc_vals, real)
        lm = self.hparams.mse_weight
        
        return adv_loss + lm * mse_loss, adv_loss
    
    def compute_discriminator_loss(self, batch, batch_idx):
        mix, clean = batch
        
        mix = unsqueeze_to_3d(mix)
        clean = unsqueeze_to_3d(clean)
        
        batch_size = mix.shape[0]        
        fake = torch.zeros((batch_size, 1, 1)).type_as(mix)
        real = torch.ones((batch_size, 1, 1)).type_as(mix)
        enh = self.generator(mix).detach()
        
        # Put everything through discriminator in one pass
        double_mix = torch.cat([mix, mix], dim=0)
        clean_enh = torch.cat([clean, enh], dim=0)
        disc_vals = self.discriminator(double_mix, clean_enh)
        
        real_fake = torch.cat([real, fake], dim=0)
        return self.adversarial_loss(disc_vals, real_fake)

    def set_training_phase(self, phase):
        assert phase in (0, 1)
        self.training_phase = phase
        self.log("training_phase", phase, logger=True)
        
    def accumulate_d_loss(self, loss):
        self.last_d_losses.append(loss)
        if len(self.last_d_losses) > self.d_loss_accum_steps:
            self.last_d_losses = self.last_d_losses[-self.d_loss_accum_steps:]
        
        if len(self.last_d_losses) == self.d_loss_accum_steps:
            return np.mean(self.last_d_losses)
        return None
    
    def reset_d_loss(self):
        self.last_d_losses = []
    
    def training_step(self, batch, batch_idx, optimizer_idx):        
        opt_gen, opt_dis = self.optimizers()

        # generator phase
        if self.training_phase == 0:
            with opt_gen.toggle_model():
                g_loss, adv_loss = self.compute_generator_loss(batch, batch_idx)
                self.log("g_loss", g_loss, logger=True, prog_bar=True)
                opt_gen.zero_grad()
                self.manual_backward(g_loss)
                opt_gen.step()

                acc_loss = self.accumulate_d_loss(adv_loss.item())
                if acc_loss is not None and acc_loss <= self.good_adv_loss:
                    self.reset_d_loss()
                    self.set_training_phase(1)
            
            return g_loss
                
        elif self.training_phase == 1:
            with opt_dis.toggle_model():
                d_loss = self.compute_discriminator_loss(batch, batch_idx)
                self.log("d_loss", d_loss, logger=True, prog_bar=True)
                opt_dis.zero_grad()
                self.manual_backward(d_loss)
                opt_dis.step()

                acc_loss = self.accumulate_d_loss(d_loss.item())
                if acc_loss is not None and acc_loss <= self.good_d_loss:
                    self.reset_d_loss()
                    self.set_training_phase(0)

            return d_loss
                
        else:
            raise ValueError('training phase bad')
    
    def validation_step(self, batch, batch_idx):
        mix, clean = batch
        mix = unsqueeze_to_3d(mix)
        clean = unsqueeze_to_3d(clean)
        
        batch_size = mix.shape[0]
        fake = torch.zeros((batch_size, 1, 1)).type_as(mix)
        real = torch.ones((batch_size, 1, 1)).type_as(mix)
        enh = self.generator(mix)
        
        clean_disc_vals = self.discriminator(mix, clean)
        enh_disc_vals = self.discriminator(mix, enh)
        
        real_loss = self.adversarial_loss(clean_disc_vals, real)
        fake_loss = self.adversarial_loss(enh_disc_vals, fake)
        val_d_loss = (real_loss + fake_loss) / 2
        
        mse_loss = F.mse_loss(clean, enh)
        adv_loss = self.adversarial_loss(enh_disc_vals, real)
        lm = self.hparams.mse_weight
        val_g_loss = adv_loss + lm * mse_loss
        
        self.log("val_g_loss", val_g_loss, on_epoch=True, prog_bar=True)
        self.log("val_d_loss", val_d_loss, on_epoch=True, prog_bar=True)
        self.log("val_mse_loss", mse_loss, on_epoch=True)
        
        return val_g_loss, val_d_loss

    def configure_optimizers(self):
        lr_g = self.hparams.lr_g
        lr_d = self.hparams.lr_d

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)
        return [opt_g, opt_d], []

#     def on_epoch_end(self):
#         z = self.validation_z.type_as(self.generator.model[0].weight)

#         # log sampled images
#         sample_imgs = self(z)
#         grid = torchvision.utils.make_grid(sample_imgs)
#         self.logger.experiment.add_image('generated_images', grid, self.current_epoch)