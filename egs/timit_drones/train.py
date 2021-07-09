import hydra
import logging
import os
import sys
import torch
import torch.nn.functional as F
import asteroid

from omegaconf import OmegaConf
from parse import *
from pathlib import PurePath
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from asteroid.data import TimitDataset
from asteroid.data.utils import CachedWavSet, RandomMixtureSet, FixedMixtureSet
from asteroid_filterbanks.transforms import mag, reim
from functools import partial

from pytorch_lightning import Trainer, seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from asteroid.engine import System
from asteroid.utils import unsqueeze_to_3d
from asteroid.losses import singlesrc_neg_sisdr, singlesrc_mse

from load_dataset import download_datasets


logger = logging.getLogger(__name__)

def perfect_cirm(noisy, clean, K=10, c=0.1):
    Xr, Xi = reim(noisy)
    Sr, Si = reim(clean)
    
    norm = Xr**2 + Xi**2
    Cr = (Xr*Sr + Xi*Si) / norm
    Ci = (Xr*Si + Xi*Sr) / norm
    
    Cr_masked = K*torch.tanh(c*Cr)
    Ci_masked = K*torch.tanh(c*Ci)
    return torch.cat((Cr_masked, Ci_masked), dim=-2)

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
        
        clean_pow = torch.pow(mag(clean_tf), 2)
        
        # a HACK to not fiddle with datasets changing - training on clean data!
        est_pow, mu, logvar = self.model.forward_vae_mu_logvar(clean_pow)
        
        loss, rec_loss, kl_loss = self.loss_func(est_pow, clean_pow, mu, logvar)
        self.log("rec_loss", rec_loss, logger=True)
        self.log("kl_loss", kl_loss, logger=True)
        return loss
    

class MagnitudeSimpleVAESystem(System):
    def common_step(self, batch, batch_nb, train=True):
        mix, clean = batch
        mix = unsqueeze_to_3d(mix)
        clean = unsqueeze_to_3d(clean)
        
        mix_tf = self.model.forward_encoder(mix)
        clean_tf = self.model.forward_encoder(clean)
        
        clean_pow = torch.pow(mag(clean_tf), 2)
        mix_pow = torch.pow(mag(mix_tf), 2)
        
        est_pow, mu, logvar = self.model.forward_vae_mu_logvar(mix_pow)
        
        loss, rec_loss, kl_loss = self.loss_func(est_pow, clean_pow, mu, logvar)
        self.log("rec_loss", rec_loss, logger=True)
        self.log("kl_loss", kl_loss, logger=True)
        return loss


class MagnitudeAESystem(System):
    def common_step(self, batch, batch_nb, train=True):
        mix, clean = batch
        mix = unsqueeze_to_3d(mix)
        clean = unsqueeze_to_3d(clean)
        
        mix_tf = self.model.forward_encoder(mix)
        clean_tf = self.model.forward_encoder(clean)
        
        true_irm = torch.minimum(mag(clean_tf) / mag(mix_tf), torch.tensor(1).type_as(mix_tf))
        est_irm = self.model.forward_masker(mix_tf)
        
        mse_loss = self.loss_func(est_irm, true_irm)

        return mse_loss
 
    
class SMoLNetSystem(System):
    def common_step(self, batch, batch_nb, train=True):
        mix, clean = batch
        mix = unsqueeze_to_3d(mix)
        clean = unsqueeze_to_3d(clean)
        
        mix_tf = self.model.forward_encoder(mix)
        clean_tf = self.model.forward_encoder(clean)
        
        model_output = self.model.forward_masker(mix_tf)
        
        if self.model.target == "cIRM":
            target_mask = perfect_cirm(mix_tf, clean_tf)
            loss = self.loss_func(model_output, target_mask)
            
        elif self.model.target == "TMS":
            loss = self.loss_func(model_output, mag(clean_tf))
        else:
            loss = self.loss_func(model_output, clean_tf)
        
        return loss

class PhasenSystem(System):
    def common_step(self, batch, batch_nb, train=True):
        mix, clean = batch
        mix = unsqueeze_to_3d(mix)
        clean = unsqueeze_to_3d(clean)
        
        mix_tf = self.model.forward_encoder(mix)
        clean_tf = self.model.forward_encoder(clean)
        est_masks = self.model.forward_masker(mix_tf)
        est_tf = self.model.apply_masks(mix_tf, est_masks)
        
        return self.loss_func(est_tf, clean_tf)

    
def sisdr_loss_wrapper(est_target, target):
    return singlesrc_neg_sisdr(est_target.squeeze(1), target.squeeze(1)).mean()

def mse_loss_wrapper(est_target, target):
    return F.mse_loss(unsqueeze_to_3d(est_target), unsqueeze_to_3d(target))

def vae_loss_wrapper(est_target, target, mu, logvar):
    ratio = target/est_target
    recon = torch.sum(ratio - torch.log(ratio) - 1 )
    KLD = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp())
    return recon + KLD, recon, KLD

def vae_simple_loss_wrapper(est_target, target, mu, logvar):
    recon = F.mse_loss(unsqueeze_to_3d(est_target), unsqueeze_to_3d(target))
    KLD = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp())
    return recon + 1e-5 * KLD, recon, KLD

def l1_loss_wrapper(est_target, target):
    return F.l1_loss(unsqueeze_to_3d(est_target), unsqueeze_to_3d(target))

def phasen_loss_wrapper(est_target, target):
    est_mag = mag(est_target)
    true_mag = mag(target)
    est_mag_comp = est_mag**0.3
    true_mag_comp = true_mag**0.3
    
    mag_loss = F.mse_loss(est_mag_comp, true_mag_comp)

    # scale the complex spectrograms' magniture to the power 0.3 as well
    true_comp_coeffs = (true_mag_comp/(1e-8+true_mag)).repeat(1,2,1)
    est_comp_coeffs = (est_mag_comp/(1e-8+est_mag)).repeat(1,2,1)
    phase_loss = F.mse_loss(est_target * est_comp_coeffs, target * true_comp_coeffs)
    
    return (mag_loss + phase_loss) / 2 

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


def init_weights(m, gain):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=gain)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    
def prepare_system(args, model, train_loader, val_loader):
    # TBD: completely different preparation for GANs
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.reduce_lr_patience)
    
    if args.loss == "sisdr":
        loss = sisdr_loss_wrapper
    elif args.loss == "phasen":
        loss = phasen_loss_wrapper
    elif args.loss == "l1":
        loss = l1_loss_wrapper
    elif args.loss in ("l2", "mse"):
        loss = mse_loss_wrapper
    elif args.loss == "vae_loss":
        loss = vae_loss_wrapper
    elif args.loss == "simple_vae_loss":
        loss = vae_simple_loss_wrapper
    else:
        raise ValueError(f'Unsupported loss type `{args.loss}`')
    
    if isinstance(model, asteroid.RegressionFCNN):
        model.compute_scaler(train_loader)
        cls = MagnitudeIRMSystem
    elif isinstance(model, asteroid.VAE):
        if args.get('model_version', None) == 'simple':
            print('Training simple VAE (on noisy data)')
            cls = MagnitudeSimpleVAESystem
        else:
            cls = MagnitudeVAESystem
    elif isinstance(model, asteroid.AutoEncoder):
        model.compute_scaler(train_loader)
        cls = MagnitudeAESystem
    elif isinstance(model, asteroid.SMoLnet):
        cls = SMoLNetSystem
    elif isinstance(model, asteroid.Phasen):
        cls = PhasenSystem
    else:
        if isinstance(model, (asteroid.DCUNet, asteroid.DCCRNet)):
            print('Applying Glorot initialization to the model weights')
            lrelu_gain = nn.init.calculate_gain('leaky_relu', 0.01)
            model.apply(partial(init_weights, gain=lrelu_gain))
            
        cls = System
        
    return cls(model, optimizer, loss, train_loader, val_loader, scheduler)


@hydra.main(config_path='conf', config_name='config')
def main(args):
    model_params = dict(args.model)
    model_params['sample_rate'] = args.sample_rate
    
    dict_args = dict(args)
    del dict_args['model']
    args = OmegaConf.create(dict_args)
    args = OmegaConf.merge(args, model_params)

    download_datasets(args)
    
    seed_everything(args.random_seed)
    train_loader, val_loader = prepare_dataloaders(args)
    
    model_conf = hydra.utils.instantiate(model_params)
    model = model_conf['model']
    
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
    
    is_fast_dev_run = args.get('fast_dev_run', False)
    
    if is_fast_dev_run:
        trainer = Trainer(max_epochs=args.max_epochs, fast_dev_run=True,
                          logger=tb_logger, callbacks=callbacks, deterministic=True)
        
        trainer.fit(system)
        
    else:
        trainer_kwargs = dict(
            max_epochs=args.max_epochs,
            gpus=args.get('gpus', -1),
            accelerator='dp',
            logger=tb_logger,
            callbacks=callbacks,
            deterministic=True
        )
        
        if args.get('resume_from_last_checkpoint', False):
            version = 'version_0' if model_version is None else model_version
            ckpt_dir = log_dir / model_name / version / 'checkpoints' 
            
            try:
                ckpts = os.listdir(ckpt_dir)
            except FileNotFoundError:
                ckpts = []
                
            if len(ckpts) == 0:
                print('No checkpoints to resume from found')
            else:
                # resuming from last epoch always
                epochs = [search('epoch={:02d}', c).fixed[0] for c in ckpts]
                max_epoch_idx = epochs.index(max(epochs))
                last_ckpt = ckpts[max_epoch_idx]
                ckpt_path = ckpt_dir / last_ckpt
                trainer_kwargs['resume_from_checkpoint'] = str(ckpt_path)
        
        trainer = Trainer(**trainer_kwargs)
        trainer.fit(system)
    
        if model_version:
            model_path = models_dir / f'{model_name}_{model_version}.pt'
        else:
            model_path = models_dir / f'{model_name}.pt'

        torch.save(model.serialize(), str(model_path))
        logging.info(f'Saved the trained model to {model_path}')

    
if __name__ == "__main__":
    main()
