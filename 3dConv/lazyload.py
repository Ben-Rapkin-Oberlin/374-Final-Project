import os

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import h5py
from plot3D import *

from PIL import Image
import numpy as np


class MyDataset(torch.utils.Dataset):
    def __init__(self,path):
        self.temp = os.listdir(path)
        self.data_files = []

        #make an instances as 2 frames
        #start from 1 because we need to compare the current frame with the previous frame
        #end at len(temp)-1 because we don't have label for the last frame
        for i in range(1,len(self.temp)-1):
            self.data_files.append([self.temp[i-1],self.temp[i]])
        
        #make labels as next frame
        self.labels = self.temp[2:]


    def __getindex__(self, idx):
        
        #paths are stored in data_files, so use idx to get the path
        path1=self.data_files[idx][0]
        path2=self.data_files[idx][1]

        a1=np.asarray(Image.open(path1))
        a2=np.asarray(Image.open(path2))

        #make a 4d tensor
        #([2,x,y,3]) 2 for 2 frames, x,y for the size of the image, 3 for RGB
        instance=torch.tensor(np.array([a1,a2]))
        

        return instance

    def __len__(self):
        return len(self.data_files)



def make_dataset(path, workers=4, batch_size=4):
    #not currently shuffling the data
    #should probably normalize the data at some stage
    dset=MyDataset(path)
    return torch.utils.DataLoader(dset, num_workers=workers, batch_size=batch_size)