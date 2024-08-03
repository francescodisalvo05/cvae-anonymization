from src.utils.data import get_databases
from scipy.spatial import distance
import numpy as np


def average_mindistance(args, x_train_anonymized):
    """
    Calculate the average nearest neighbour distance.
    
    :param args: argparse with running configuration.
    :param x_train_anonymized: anonymized training data.
    :return the average nearest neighbour distance
    """
    
    # Load initial dataset
    x_train, _, _, _, _, _ = get_databases(args)

    # Normalize
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / std

    x_train = x_train[:len(x_train_anonymized)] # with k-Same we might have slightly less samples
                                                # due to the equal-size clustering
    
    # pairwise distances from each point in x_train to x_train_anonymized
    dist_matrix = distance.cdist(x_train_anonymized, x_train, 'euclidean')  
    # minimum distance to the nearest neighbor for each element in x_train
    min_distances = np.min(dist_matrix, axis=1)
    # Calculate the average of these minimum distances
    average_min_distance = np.mean(min_distances)
    return average_min_distance
