from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from medimeta import MedIMeta
from medmnist import INFO
from tqdm import tqdm

import torchvision.transforms as T
import torchvision
import numpy as np
import medmnist
import torch
import os

from src.utils.utils import get_anonymizer_ds_flag


def extract_embeddings(model, device, dataloader):
    """
    Extract features from the selected backbone.
    :param model: Selected backbone.
    :param device: Running device.
    :param dataloader: Current dataloader (train|test).
    :return data: Dictionary containing the extracted data.
    """

    embeddings_db, labels_db = [], []
    for extracted in tqdm(dataloader):
        images, labels = extracted
        images = images.to(device)
        output = model.forward_features(images)
        output = model.forward_head(output, pre_logits=True)
        labels_db.extend(labels)
        embeddings_db.extend(output.detach().cpu().numpy())

    data = {
        'embeddings': embeddings_db,
        'labels': labels_db
    }

    return data


def get_dataloaders(args, transforms):
    """
    Return the dataloaders for the selected dataset and backbone.
    :param args: Namespace containing parameters and hyperparameters of the current run.
    :param transforms: Transform function (determined via timm for the selected backbone). 
    :return data: Dictionary containing both train and test dataloaders.
    """
    dataloaders = {}
    data_path = os.path.join(args.dataset_root,args.dataset)

    if args.dataset == "dtd":
        trainset = torchvision.datasets.DTD(root=data_path, split='train', download=True, transform=transforms)
        valset = torchvision.datasets.DTD(root=data_path, split='val', download=True, transform=transforms)
        testset = torchvision.datasets.DTD(root=data_path, split='test', download=True, transform=transforms)
        dataloaders['train'] = DataLoader(trainset, batch_size = args.batch_size, shuffle = False)
        dataloaders['val'] = DataLoader(valset, batch_size = args.batch_size, shuffle = False)
        dataloaders['test'] = DataLoader(testset, batch_size = args.batch_size, shuffle = False)
        return dataloaders
    
    elif args.dataset == "octdl":
        trainset = ImageFolder(root=os.path.join(data_path,'dataset_1','train'), transform=transforms)
        valset = ImageFolder(root=os.path.join(data_path,'dataset_1','val'), transform=transforms)
        testset = ImageFolder(root=os.path.join(data_path,'dataset_1','test'), transform=transforms)
        dataloaders['train'] = DataLoader(trainset, batch_size = args.batch_size, shuffle = False)
        dataloaders['val'] = DataLoader(valset, batch_size = args.batch_size, shuffle = False)
        dataloaders['test'] = DataLoader(testset, batch_size = args.batch_size, shuffle = False)
        return dataloaders

    elif args.dataset == "aircraft":
        trainset = torchvision.datasets.FGVCAircraft(root=data_path, split='train', download=True, transform=transforms)
        valset = torchvision.datasets.FGVCAircraft(root=data_path, split='val', download=True, transform=transforms)
        testset = torchvision.datasets.FGVCAircraft(root=data_path, split='test', download=True, transform=transforms)
        dataloaders['train'] = DataLoader(trainset, batch_size = args.batch_size, shuffle = False)
        dataloaders['val'] = DataLoader(valset, batch_size = args.batch_size, shuffle = False)
        dataloaders['test'] = DataLoader(testset, batch_size = args.batch_size, shuffle = False)
        return dataloaders

    elif args.dataset == "skinl_derm":
        root_data_path = os.path.join(args.dataset_root,'medimeta')
        trainset = MedIMeta(root_data_path, args.dataset, "Diagnosis grouped", split="train", transform=transforms) 
        valset = MedIMeta(root_data_path, args.dataset, "Diagnosis grouped", split="val", transform=transforms)
        testset = MedIMeta(root_data_path, args.dataset, "Diagnosis grouped", split="test", transform=transforms)
        dataloaders['train'] = DataLoader(trainset, batch_size = args.batch_size, shuffle = False)
        dataloaders['val'] = DataLoader(valset, batch_size = args.batch_size, shuffle = False)
        dataloaders['test'] = DataLoader(testset, batch_size = args.batch_size, shuffle = False)
        return dataloaders
    
    
    elif args.dataset == "organs_axial":
        # convert img to RGB & append after ToTensor()
        to_rgb = T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        transforms.transforms.insert(-1, to_rgb)
        root_data_path = os.path.join(args.dataset_root,'medimeta')
        trainset = MedIMeta(root_data_path, args.dataset, "organ label", split="train", transform=transforms) 
        valset = MedIMeta(root_data_path, args.dataset, "organ label", split="val", transform=transforms)
        testset = MedIMeta(root_data_path, args.dataset, "organ label", split="test", transform=transforms)
        dataloaders['train'] = DataLoader(trainset, batch_size = args.batch_size, shuffle = False)
        dataloaders['val'] = DataLoader(valset, batch_size = args.batch_size, shuffle = False)
        dataloaders['test'] = DataLoader(testset, batch_size = args.batch_size, shuffle = False)
        return dataloaders

    
    elif args.dataset == "breastmnist":
        info = INFO[args.dataset] 
        DataClass = getattr(medmnist, info['python_class']) # get relative dataset class
        data_path = os.path.join(args.dataset_root,'medmnist') # they are all under {dataset_root}/medmnist
        trainset = DataClass(split='train', transform=transforms, download=False, as_rgb=True, size=224, root=data_path, mmap_mode='r')
        valset = DataClass(split='val', transform=transforms, download=False, as_rgb=True, size=224, root=data_path, mmap_mode='r')
        testset = DataClass(split='test', transform=transforms, download=False, as_rgb=True, size=224, root=data_path, mmap_mode='r')
        dataloaders['train'] = DataLoader(trainset, batch_size = args.batch_size, shuffle = False)
        dataloaders['val'] = DataLoader(valset, batch_size = args.batch_size, shuffle = False)
        dataloaders['test'] = DataLoader(testset, batch_size = args.batch_size, shuffle = False)
        return dataloaders
    
    # STL and OXFORD PETS do not have validation set
    elif args.dataset == "stl10":
        trainvalset = torchvision.datasets.STL10(root=data_path, split='train', download=True, transform=transforms)
        testset = torchvision.datasets.STL10(root=data_path, split='test', download=True, transform=transforms)

    elif args.dataset == "oxford-pets":
        trainvalset = torchvision.datasets.OxfordIIITPet(root=data_path, split='trainval', download=True, transform=transforms)
        testset = torchvision.datasets.OxfordIIITPet(root=data_path, split='test', download=True, transform=transforms)

    else:
        raise ValueError(f"{args.dataset} not available")


    # Extract labels for stratification
    labels = [label for _, label in trainvalset]

    # Split the dataset into training and validation
    # >> use always the same split (fix seed independently)!
    train_idx, val_idx = train_test_split(range(len(trainvalset)), test_size=0.10, random_state=12345, stratify=labels)

    # Create subsets for training and validation
    trainset = Subset(trainvalset, train_idx)
    valset = Subset(trainvalset, val_idx)

    dataloaders['train'] = DataLoader(trainset, batch_size = args.batch_size, shuffle = False)
    dataloaders['val'] = DataLoader(valset, batch_size = args.batch_size, shuffle = False)
    dataloaders['test'] = DataLoader(testset, batch_size = args.batch_size, shuffle = False)
    return dataloaders


def get_databases(args, anonymized=False):
    """
    Lead pre-generated `npz` databases, using the relative keys `embeddings` and `labels`.
    This allows to significantly increase the runtime speed, with enough storage capacities.

    :param args: Namespace containing parameters and hyperparameters of the current run.
    :param anonymized: Boolean value used to retrieve the raw training set or the anonymized one.
    :return x_train, y_train, x_test, y_test: Numpy arrays.
    """

    db_path = os.path.join(args.database_root, 
                           args.dataset)    
    
    if anonymized:
        train_path = os.path.join(db_path,f'train_{get_anonymizer_ds_flag(args)}.npz')
    else:
        train_path = os.path.join(db_path,'train.npz')
    
    train_data = np.load(train_path)

    x_train = train_data['embeddings']
    y_train = train_data['labels'].reshape(-1,).astype(int)

    val_path = os.path.join(db_path,'val.npz')
    val_data = np.load(val_path)
    
    x_val = val_data['embeddings']
    y_val = val_data['labels'].reshape(-1,).astype(int)

    test_path = os.path.join(db_path,'test.npz')
    test_data = np.load(test_path)

    x_test = test_data['embeddings']
    y_test = test_data['labels'].reshape(-1,).astype(int)

    return x_train, y_train, x_val, y_val, x_test, y_test


def normalize_embeddings(x_train, x_val, x_test):
    """
    Normalization of feature-vectors.
    
    :param x_train: train features.
    :param x_val: val features.
    :param x_test: test features.
    :return: normalized datasets.
    """
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_norm = (x_train - mean) / std
    x_val_norm = (x_val - mean) / std
    x_test_norm = (x_test - mean) / std
    return x_train_norm, x_val_norm, x_test_norm
    

def get_dataloaders_from_embeddings(args, x_train, y_train, x_val, y_val, x_test, y_test, generator):
    """
    Returns dataloaders, given input data.
    
    :param args: argparse for the current run.
    :param x_train|x_val|x_test: feature vectors of the given split.
    :param y_train|y_val|y_test: labels of the current split.
    :param generator: torch's Generator for dataloaders.
    :return: dataloaders (train,val,test).
    """
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers=1, shuffle = True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers=1, shuffle = False, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, num_workers=1, shuffle = False, generator=generator)
    return train_loader, val_loader, test_loader


def label_distribution(y):
    """
    Calculate label class-categorical distribution.
    Note that we assume that the labels are between 0-(num_labels-1).
    
    :param y: list of labels.
    :return: list of class-probabilities.
    """
    num_labels = np.max(y) + 1
    
    # Initialize an array of zeros with a size equal to the number of labels
    label_counts = np.zeros(num_labels)
    
    # Count occurrences of each label
    for label in y:
        label_counts[label] += 1
    
    # Calculate probabilities by dividing each count by the total number of labels
    probabilities = label_counts / len(y)
    
    return probabilities