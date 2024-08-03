"""
Apply k-same on a latent vector.

Same-size clustering adapted from: https://github.com/hcholab/k-salsa/blob/main/k-SALSA_algorithm/scripts/k-SALSA.py#L236
I polished it and translated it into torch in order to run it on cuda.
"""

from copy import deepcopy

import torch 
import scipy


class kSAME:
    def __init__(self, 
                 k : int = 2, 
                 device : str = "cuda"):
        """
        :param k: bumber of samples to collapse into their centroid.
        :param device: where to run the experiment.
        """
        assert k, f"`k` must must be defined via args.k"
        self.k = k
        self.device = device


    def apply(self, x, y):
        """
        Apply kSame on a given dataset

        :param x: training embeddings.
        :param y: training labels.
        :return anonymized embeddings and corresponding labels.
        """

        # len(data) must be multiple by k
        #   if not, drop the last len(data)%k samples
        if len(x) % self.k != 0:
            x = x[:-(len(x)%self.k)]
            y = y[:-(len(y)%self.k)]
        
        x = torch.Tensor(x).to(self.device)

        # pairwise distance matrix
        pdist = torch.pdist(x, p=2)
        PDISTANCE = torch.Tensor(scipy.spatial.distance.squareform(pdist.cpu().numpy())).to(self.device)

        # get centroids according to nearestNeighborClust
        cluster_indices = self.nearest_neighbour_clustering(PDISTANCE)

        # replace each cluster with its centroid
        x_anonymized = deepcopy(x)
        for id_centroid in cluster_indices.unique():
            curr_cluster = torch.where(cluster_indices == id_centroid)[0]
            x_anonymized[curr_cluster] = torch.mean(x_anonymized[curr_cluster],0)

        return x_anonymized.cpu().numpy(), y
        
    
    def nearest_neighbour_clustering(self, PDISTANCE):
        """
        Compute the nearest neighbour clustering.

        :args PDISTANCE: pairwise distance of training data.
        :return cluster_indices
        """
        PDISTANCE = deepcopy(PDISTANCE)
        assert PDISTANCE.shape[0] == PDISTANCE.shape[1]

        size = PDISTANCE.shape[0]

        PDISTANCE += torch.diag(torch.Tensor([float('nan')] * size).to(self.device))

        cluster_indices = torch.zeros(size, dtype=torch.uint8) + float('nan')
        current_index = 1

        while torch.any(torch.isnan(cluster_indices)):
            
            next_point = self._maxD(PDISTANCE, torch.isnan(cluster_indices))
            nearest = torch.argsort(PDISTANCE[next_point,:])[:self.k-1]
            
            # Assign cluster
            cluster = torch.cat((nearest, torch.tensor([next_point]).to(self.device))).long()
            cluster_indices[cluster] = current_index
            
            # Update matrix
            PDISTANCE[cluster,:] = float('nan')
            PDISTANCE[:,cluster] = float('nan')
            
            current_index += 1

        return cluster_indices

    
    def _nanargmax(self, tensor, dim=None, keepdim=False):
        # https://github.com/pytorch/pytorch/issues/61474#issuecomment-1735537507
        min_value = torch.finfo(tensor.dtype).min
        output = tensor.nan_to_num(min_value).argmax(dim=dim, keepdim=keepdim)
        return output


    def _maxD(self, M, allowed):
        s = torch.nansum(M,axis=1)
        s[~allowed] = float('nan')
        return self._nanargmax(s)