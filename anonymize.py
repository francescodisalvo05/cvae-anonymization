from src.utils.utils import seed_everything, get_anonymizer_ds_flag
from src.utils.data import get_databases

from src.anonymizer.cvae import CVAEAnonymizer
from src.anonymizer.ksame import kSAME

from argparse import ArgumentParser
import numpy as np
import time
import torch
import os





def main(args):

    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    # LOAD TENSORS & PREPROCESSING
    x_train, y_train, x_val, y_val, _, _ = get_databases(args, anonymized=False)

    # EXECUTE
    print(f"Starting anonymization of `{args.dataset}` with `{args.anonymizer}`")
    start = time.time()
    if args.anonymizer == "cvae":
        anonymizer = CVAEAnonymizer(args, g)
        x_train_anonymized, y_train = anonymizer.apply(x_train, y_train, x_val, y_val)
    elif args.anonymizer == "ksame":
        assert args.k, f"You need to set a proper value of `k`, you have {args.k}"
        anonymizer = kSAME(k=args.k, device=args.device)
        x_train_anonymized, y_train = anonymizer.apply(x_train, y_train)
    print(f"\tElapsed time = {(time.time() - start):.2f}s")

    # STORE ANONYMIZED DBs >> database_root / dataset / train_[anonymizer].npz
    train_data = {'embeddings': x_train_anonymized, 'labels': y_train}
    output_path = os.path.join(args.database_root,args.dataset,f'train_{get_anonymizer_ds_flag(args)}.npz')
    np.savez_compressed(output_path, **train_data)


if __name__ == '__main__':

    parser = ArgumentParser()

    # GENERAL
    parser.add_argument('--database_root', type=str, default="assets/database", help='define the database root')
    parser.add_argument('--ckpt_root', type=str, default='assets/ckpts/', help='define the checkpoint root')

    # DATASET & HYPERPARAMS
    parser.add_argument('--dataset', type=str, required=True, help='define the dataset name')
    parser.add_argument('--batch_size', type=int, default=128, help='define the batch size')
    parser.add_argument('--seed', type=int, default=42, help='define the random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='define the running device')

    # ANONYMIZATION
    parser.add_argument('--anonymizer', type=str, default='ksame', help='define the anonymizer', choices=['ksame','cvae'])
    parser.add_argument('--k', type=int, default=None, help='define the k value used for k-SAME')


    args = parser.parse_args()

    # run
    main(args)