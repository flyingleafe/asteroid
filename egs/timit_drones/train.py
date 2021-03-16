import hydra
import logging
import os
import torch
import torch.nn.functional as F
import asteroid

from pathlib import PurePath
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from asteroid.data import TimitDataset
from asteroid.data.utils import CachedWavSet, RandomMixtureSet, FixedMixtureSet
from asteroid_filterbanks.transforms import mag

from pytorch_lightning import Trainer, seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from asteroid.engine import System
from asteroid.utils import unsqueeze_to_3d
from asteroid.losses import singlesrc_neg_sisdr, singlesrc_mse

from load_dataset import download_datasets


logger = logging.getLogger(__name__)


class MagnitudeIRMSystem(System):
    def common_step(self, batch, batch_nb, train=True):
        mix, clean = batch
        mix = unsqueeze_to_3d(mix)
        clean = unsqueeze_to_3d(clean)
        
        mix_tf = self.model.forward_encoder(mix)
        clean_tf = self.model.forward_encoder(clean)
        
        true_irm = torch.minimum(mag(clean_tf) / mag(mix_tf), torch.tensor(1).type_as(mix_tf))
        est_irm = self.model.forward_masker(mix_tf)
        loss = self.loss_func(est_irm, true_irm)
        return loss

class MagnitudeVAESystem(System):
    def common_step(self, batch, batch_nb, train=True):
        mix, clean = batch
        mix = unsqueeze_to_3d(mix)
        clean = unsqueeze_to_3d(clean)
        
        mix_tf = self.model.forward_encoder(mix)
        clean_tf = self.model.forward_encoder(clean)
        
        true_irm = torch.minimum(mag(clean_tf) / mag(mix_tf), torch.tensor(1).type_as(mix_tf))
        est_irm, mu, logvar = self.model.forward_masker(mix_tf)
        
        #loss = self.loss_func(est_irm, true_irm, mu, logvar)
        loss = self.loss_func(est_irm, true_irm)

        #loss_mse = self.loss_func(est_irm, true_irm)
        # loss kld taken from - https://gitlab.inria.fr/smostafa/avse-vae/-/blob/master/train_VAE.py
        #loss_kld = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp())

        #pytorch kl - issue - goes to nan - log 0 ?
        #loss_kld = F.kl_div(mu, logvar)
        #loss = loss_mse + loss_kld

        #import pdb; pdb.set_trace()
        return loss

    
class SMoLNetSystem(System):
    def common_step(self, batch, batch_nb, train=True):
        mix, clean = batch
        mix = unsqueeze_to_3d(mix)
        clean = unsqueeze_to_3d(clean)
        
        mix_tf = self.model.forward_encoder(mix)
        clean_tf = self.model.forward_encoder(clean)
        
        model_output = self.model.forward_masker(mix_tf)
        
        if self.model.target == "cIRM":
            raise NotImplementedError('Too lazy to fully implement cIRM now!')
            
        elif self.model.target == "TMS":
            loss = self.loss_func(model_output, mag(clean_tf))
        else:
            loss = self.loss_func(model_output, clean_tf)
        
        return loss


def sisdr_loss_wrapper(est_target, target):
    return singlesrc_neg_sisdr(est_target.squeeze(1), target).mean()

def mse_loss_wrapper(est_target, target):
    return F.mse_loss(unsqueeze_to_3d(est_target), unsqueeze_to_3d(target))

def vae_loss_wrapper(est_target, target, mu, logvar):
    recon = torch.sum( torch.log(est_target) + target/(est_target) )
    KLD = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp())
    return recon + KLD

def l1_loss_wrapper(est_target, target):
    return F.l1_loss(unsqueeze_to_3d(est_target), unsqueeze_to_3d(target))

def prepare_mixture_set(clean, noises, params, **kwargs):
    mix_type = params.pop('type')
    if mix_type == 'random':
        cls = RandomMixtureSet
    elif mix_type == 'fixed':
        cls = FixedMixtureSet
        
    return cls(clean, noises, **params, **kwargs)


def prepare_dataloaders(args):
    train_noise_dir = PurePath(args.dset.noises) / 'noises-train-drones'
    noises = CachedWavSet(train_noise_dir, sample_rate=args.sample_rate, precache=True)

    # Load clean data and split it into train and val
    timit = TimitDataset(args.dset.timit, subset='train', sample_rate=args.sample_rate, with_path=False)
    
    len_train = int(len(timit) * (1 - args.dset.val_fraction))
    len_val = len(timit) - len_train
    timit_train, timit_val = random_split(timit, [len_train, len_val],
                                          generator=torch.Generator().manual_seed(args.random_seed))
    
    timit_train = prepare_mixture_set(timit_train, noises, dict(args.dset.mixture_train),
                                      random_seed=args.random_seed, crop_length=args.crop_length)
    timit_val   = prepare_mixture_set(timit_val, noises, dict(args.dset.mixture_val),
                                      random_seed=args.random_seed, crop_length=args.crop_length)    
    
    train_loader = DataLoader(timit_train, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.dl_workers, drop_last=True)
    val_loader   = DataLoader(timit_val, batch_size=args.batch_size,
                              num_workers=args.dl_workers, drop_last=True)
    
    return train_loader, val_loader

def prepare_system(args, model, train_loader, val_loader):
    # TBD: completely different preparation for GANs
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.reduce_lr_patience)
    
    if args.loss == "sisdr":
        loss = sisdr_loss_wrapper
    elif args.loss == "l1":
        loss = l1_loss_wrapper
    elif args.loss in ("l2", "mse"):
        loss = mse_loss_wrapper
    elif args.loss == "vae_loss":
        loss = vae_loss_wrapper
    else:
        raise ValueError(f'Unsupported loss type `{args.loss}`')
    
    if isinstance(model, asteroid.RegressionFCNN):
        model.compute_scaler(train_loader)
        cls = MagnitudeIRMSystem
    elif isinstance(model, asteroid.VAE):
        model.compute_scaler(train_loader)
        cls = MagnitudeVAESystem
    elif isinstance(model, asteroid.SMoLnet):
        cls = SMoLNetSystem
    else:
        cls = System
        
    return cls(model, optimizer, loss, train_loader, val_loader, scheduler)


@hydra.main(config_path='conf', config_name='config')
def main(args):
    download_datasets(args)
    
    seed_everything(args.random_seed)
    train_loader, val_loader = prepare_dataloaders(args)
    
    model = hydra.utils.instantiate(args.model)
    model_version = args.get('model_version', None)
    model_name = model.__class__.__name__
    logger.info(f'Training model: {model_name}')
    
    system = prepare_system(args, model, train_loader, val_loader)
    
    workspace_dir = PurePath(args.workspace_dir)
    log_dir = workspace_dir / 'logs'
    models_dir = workspace_dir / 'models'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    tb_logger = pl_loggers.TensorBoardLogger(str(log_dir), name=model_name, version=model_version)
    
    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStopping(**args.early_stopping))
    
    if args.model_checkpoint:
        callbacks.append(ModelCheckpoint(**args.model_checkpoint))
    
    if args.get('fast_dev_run', False):
        trainer = Trainer(max_epochs=args.max_epochs, fast_dev_run=True,
                          logger=tb_logger, callbacks=callbacks, deterministic=True)
    else:
        gpus = args.get('gpus', -1)
        trainer = Trainer(max_epochs=args.max_epochs, gpus=gpus, accelerator='dp',
                          logger=tb_logger, callbacks=callbacks, deterministic=True)
    trainer.fit(system)
    
    if model_version:
        model_path = models_dir / f'{model_name}_{model_version}.pt'
    else:
        model_path = models_dir / f'{model_name}.pt'
        
    torch.save(model.serialize(), str(model_path))
    logging.info(f'Saved the trained model to {model_path}')

    
if __name__ == "__main__":
    main()
