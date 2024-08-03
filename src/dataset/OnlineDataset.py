from torch.utils.data import Dataset
import numpy as np


class OnlineDataset(Dataset):

    def __init__(self, 
                 cvae, 
                 label_distribution, 
                 data_size,
                 variance = 1.0,
                 transform=None):
        """
        Dataset class based on CVAE's decoder. 
        This allows to generate novel samples at each batch.

        :param cvae: CVAE Anonymizer class
        :param label_distribution: list containing the distribution probabilities for each class.
                                   Assume list[i] represents class i.
        :param data_size: length of the initial dataset. 
                          Just used as a reference number for the epochs.
        :param transform: transforms class.
        """
        
        self.cvae = cvae
        self.label_distribution = label_distribution
        self.data_size = data_size
        self.transform = transform
        self.unique_labels = np.arange(len(self.label_distribution))
        self.variance = variance

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):

        # sample y according to label_distribution
        label = np.random.choice(self.unique_labels, p=self.label_distribution)

        # generate sample of the given class
        # i.e. sample z \sim N(0,1)
        #      cvae.decode(z,label)
        sample = self.cvae.generate_one(label, self.variance)
        
        # transform, if required
        if self.transform:
            sample = self.transform(sample)

        return sample, label
