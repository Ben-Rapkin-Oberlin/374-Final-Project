
#partition {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
#labels {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}

import torch

class dataGen():

    def __init__(self, list_IDs, labels):
        #labels is a dict that given an id, will tell what set it is apart of
        #list_IDs is a list of ids that are in the set
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
    #  'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
    #'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y