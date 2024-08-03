from src.model.cvae import CVAE

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer

import numpy as np
import torch
import os


class CVAEAnonymizer:
    def __init__(self, args):
        """
        :param args: Namespace containing parameters and hyperparameters of the current run.
        """

        self.args = args
        self.train_logs = {}
        
        # these hyperparameters are fixed.
        self.device = 'cuda'
        self.hidden_dim = 256
        self.latent_dim = 50  # this will be multiplied by two (to account for mean & std)

        # early stopping
        self.best_loss = np.inf
        self.best_epoch = 0
        self.best_model = None
        self.epochs_no_improve = 0
        self.n_epochs_stop = 15

        self.method = args.anonymizer # it could be cvae or cvae-online
        assert self.method in ['cvae','cvae-online']


    def fit(self, data, labels, val_data, val_labels, g):
        """
        Fit the CVAE.
        
        :param data: Training data.
        :param labels: Training labels.
        :param val_data: Validation data.
        :param val_labels: Validation labels.
        :param g: Random generator.
        """

        self.num_classes = len(np.unique(labels))

        train_loader, val_loader = self.prepare_data(data, labels, val_data, val_labels, g)
            
        ckpt_path = os.path.join(self.args.ckpt_root,self.args.dataset,'cvae-online',f'cvae-online-s{self.args.seed}')
        best_ckpt = os.path.join(ckpt_path,'best.ckpt')
        
        checkpoint_callback = ModelCheckpoint(dirpath=ckpt_path, filename="{epoch}")
        checkpoint_callback_best = ModelCheckpoint(dirpath=ckpt_path, monitor="val_loss", mode="min", filename="best")
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=15)
        
        trainer = Trainer(accelerator="gpu",
            devices=[0],
            max_epochs=1000, 
            logger=False,
            check_val_every_n_epoch=1, 
            callbacks=[checkpoint_callback,checkpoint_callback_best,early_stopping]
        )

        if not os.path.isfile(best_ckpt): 
            self.model = CVAE(
                input_dim=len(data[0]),
                hidden_dim = self.hidden_dim,
                latent_dim = self.latent_dim,
                n_classes = self.num_classes,
                device = self.device
            )
            trainer.fit(self.model, train_dataloaders = train_loader, val_dataloaders = val_loader)
        
        
        self.model = CVAE.load_from_checkpoint(best_ckpt) 
        

    def generate_one(self, label, variance):
        """
        Generate one single sample.
        
        :param label: label of the sample to generate.
        :param variance: sampling variance.
        :return generated sample.
        """
        return self.model.generate_one(label, self.train_min, self.train_max, variance)


    def prepare_data(self, data, labels, val_data, val_labels, g):
        """
        Preprocessing step.
        
        :param data: Training data.
        :param labels: Training labels.
        :param val_data: Validation data.
        :param val_labels: Validation labels.
        :param g: Random generator.
        :return train_loader, val_loader.
        """
        data, labels = torch.Tensor(data).to(self.device), torch.Tensor(labels).to(self.device).to(int)
        val_data, val_labels = torch.Tensor(val_data).to(self.device), torch.Tensor(val_labels).to(self.device).to(int)

        data, train_min, train_max = self._normalize(data)
        val_data, _, _ = self._normalize(val_data, train_min, train_max)

        self.train_min = train_min
        self.train_max = train_max

        train_dataset = TensorDataset(data, labels)
        val_dataset = TensorDataset(val_data, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size = 512, shuffle = True, generator=g)
        val_loader = DataLoader(val_dataset, batch_size = 512, shuffle = False, generator=g)
        return train_loader, val_loader


    def _normalize(self, tensor, _min = None, _max = None):
        """
        Normalize tensor.

        :param tensor: embedding to normalize.
        :param _min: minimum value for normalization.
        :param _max: minimum value for normalization.
        :return normalized tensor.
        """
        if (_min == None) and (_max == None):
            _min = tensor.min(dim=0).values
            _max = tensor.max(dim=0).values
        return (tensor - _min) / (_max - _min), _min, _max


    def _un_normalize(self, tensor, _min, _max):
        """
        Un-normalize tensor.

        :param tensor: embedding to normalize.
        :param _min: minimum value for un-normalizing.
        :param _max: minimum value for un-normalizing.
        :return un-normalized tensor.
        """
        return (tensor) * (_max - _min) + _min
    

    def _get_label_distribution(self, labels):
        """
        Un-normalize data.

        :param tensor: embedding to normalize.
        :param _min: minimum value for un-normalizing.
        :param _max: minimum value for un-normalizing.
        :return label distribution.
        """
        num_labels = np.max(labels) + 1
        label_counts = np.zeros(num_labels)
        for label in labels:
            label_counts[label] += 1
        label_distribution = label_counts / len(labels)
        return label_distribution