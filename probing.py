from src.utils.data import get_databases, get_dataloaders_from_embeddings, normalize_embeddings
from src.utils.utils import seed_everything, get_anonymizer_flag
from src.utils.metrics import average_mindistance

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from src.model.linear import LinearModule
from argparse import ArgumentParser

import pytorch_lightning as pl
import numpy as np
import json
import os

import random
import torch


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(args):

    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)
    g = torch.Generator()
    g.manual_seed(args.seed)


    # LOAD DATA & PREPROCESSING
    anonymized_flag = args.anonymizer in ['cvae','ksame']
    x_train, y_train, x_val, y_val, x_test, y_test = get_databases(args, anonymized=anonymized_flag)
    x_train, x_val, x_test = normalize_embeddings(x_train, x_val, x_test) 
    train_loader, val_loader, test_loader = get_dataloaders_from_embeddings(args, x_train, y_train, x_val, y_val, x_test, y_test, g) 
     
    # TRAINING PIPELINE
    
    # get ckpt 
    ckpt_path = os.path.join(args.ckpt_root,args.dataset,'probing',get_anonymizer_flag(args))
    best_ckpt = os.path.join(ckpt_path,'best.ckpt')
    # callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_path, filename="{epoch}")
    checkpoint_callback_best = ModelCheckpoint(dirpath=ckpt_path, monitor="Val/Loss", mode="min", filename="best")
    early_stopping = EarlyStopping(monitor="Val/Loss", mode="min", patience=5)


    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=args.num_epochs,
        logger=False, 
        callbacks=[checkpoint_callback,checkpoint_callback_best,early_stopping],
        check_val_every_n_epoch=1,
    )

    # train or load ckpt
    if not os.path.isfile(best_ckpt):
        # setup model
        model = LinearModule(num_classes = len(np.unique(y_train)), embedding_dimension = len(x_train[0]), lr = args.learning_rate)
        # train for max_epochs or until early stopping
        trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)

    
    model = LinearModule.load_from_checkpoint(best_ckpt)    #####

    # INFERENCE
    trainer.test(model, test_loader)
    auc = model.test_auroc * 100
    avg_mind = average_mindistance(args, x_train)
    
    
    logs_folder = os.path.join(args.output_root,args.dataset)
    filepath = os.path.join(logs_folder,f'{args.dataset}_{get_anonymizer_flag(args)}.json')
    os.makedirs(logs_folder, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump({'auc' : auc, 'average_mindistance' : avg_mind}, f, indent=4)


if __name__ == '__main__':

    parser = ArgumentParser()

    # GENERAL
    parser.add_argument('--database_root', type=str, default="assets/database", help='define the database root')
    parser.add_argument('--ckpt_root', type=str, default='assets/ckpts/', help='define the checkpoint root')
    parser.add_argument('--output_root', type=str, default='assets/logs/', help='define the output root')

    # DATASET & HYPERPARAMS
    parser.add_argument('--dataset', type=str, required=True, help='define the dataset name')
    parser.add_argument('--backbone', type=str, default='vit_base_patch14_dinov2.lvd142m', help='define the feature extractor')
    parser.add_argument('--seed', type=int, default=42, help='define the random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='define the running device')

    # ANONYMIZATION
    parser.add_argument('--anonymizer', type=str, default='identity', help='define the anonymizer',
                                                  choices=['ksame','cvae','identity'])
    parser.add_argument('--k', type=int, default=None, help='k value defined for k-SAME')
    
    # TRAINING
    parser.add_argument('-l','--learning_rate', type=float, default=0.001, help='define the learning rate')
    parser.add_argument('-e','--num_epochs', type=int, default=100, help='define the maximum number of epochs')
    parser.add_argument('-b','--batch_size', type=int, default=128, help='define the batch size')
    

    args = parser.parse_args()

    # seed
    seed_everything(args.seed)

    # run
    main(args)