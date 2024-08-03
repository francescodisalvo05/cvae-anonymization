from src.utils.utils import init_db_folders, seed_everything
from src.utils.data import get_dataloaders, extract_embeddings

from argparse import ArgumentParser

import numpy as np
import torch
import timm
import os

def main(args):

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get model from timm
    model = timm.create_model(args.backbone, pretrained=True, num_classes=0).to(device)
    model.requires_grad_(False)
    model = model.eval()

    # get the required transform function for the given feature extractor
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # get dataloaders
    dataloaders  = get_dataloaders(args, transforms)

    # create database folders, if necessary
    init_db_folders(args)

    print(f"Creating DBs for {args.dataset}...")

    for split in dataloaders.keys():

        # get database of embeddings in the form
        #   db = {'embeddings' : [...], 'labels' : [...]
        db = extract_embeddings( model = model, 
                                 device = device,
                                 dataloader = dataloaders[split])
        
        # store database
        # database_root / dataset / backbone / train|test.npz
        np.savez_compressed(os.path.join(args.database_root,args.dataset,f'{split}.npz'), **db)


if __name__ == '__main__':

    parser = ArgumentParser()

    # GENERAL
    parser.add_argument('--dataset_root', type=str, default="assets/data", help='define the dataset root')
    parser.add_argument('--database_root', type=str, default="assets/database", help='define the database root')

    # DATASET & HYPERPARAMS
    parser.add_argument('--dataset', type=str, required=True, help='define the dataset name')
    parser.add_argument('--backbone', type=str, default='vit_base_patch14_dinov2.lvd142m', help='define the feature extractor')
    parser.add_argument('--batch_size', type=int, default=128, help='define the batch size')
    parser.add_argument('--seed', type=int, default=42, help='define the random seed')

    args = parser.parse_args()

    main(args)