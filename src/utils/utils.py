import numpy as np
import random, os
import torch


def seed_everything(seed=42):
    """
    Ensure reproducibility.
    :param seed: Integer defining the seed number.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_db_folders(args):
    """
    Create database's folder, if necessary.
    The folder structure is as follows:
        database_root (from args) / dataset-name
    :param args: Namespace containing parameters and hyperparameters of the current run.
    """
    # database root
    if not os.path.exists(args.database_root):
        os.makedirs(args.database_root)

    # database root / dataset
    dataset_path = os.path.join(args.database_root,args.dataset)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)


def get_anonymizer_flag(args):
    """
    Get the `flag` (i.e., id) of the current anonymization.
    This is used for logging purposes.
    :param args: Namespace containing parameters and hyperparameters of the current run.
    """
    if args.anonymizer == "ksame":
        return f'{args.anonymizer}-{args.k}-s{args.seed}'
        
    elif args.anonymizer in ["cvae","identity"]:
        return f'{args.anonymizer}-s{args.seed}'


def get_anonymizer_ds_flag(args):
    """
    Get the `flag` (i.e., identifier) of the anonymized dataset.
    :param args: Namespace containing parameters and hyperparameters of the current run.
    """
    if args.anonymizer == "ksame":
        return f'{args.anonymizer}-{args.k}' # kSame is deterministic
        
    elif args.anonymizer == "cvae":
        return f'{args.anonymizer}-s{args.seed}' # then use lambda!
