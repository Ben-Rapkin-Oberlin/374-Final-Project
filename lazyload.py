import os

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *


from PIL import Image
import numpy as np

# TODO Add labels for discriminator
# Look at cugan2.py line 235

# TODO Add this dataload to cugan2.py
# Look at cugan2.py line 197 

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,path):
        self.temp = np.sort(os.listdir(path))
        print(self.temp)
        self.data_files = []
        self.path=path
        #make an instances as 2 frames
        #start from 1 because we need to compare the current frame with the previous frame
        #end at len(temp)-1 because we don't have label for the last frame
        for i in range(1,len(self.temp)-1):
            self.data_files.append([self.temp[i-1],self.temp[i],self.temp[i+1]])
        
        #make labels as next frame
        self.labels = self.temp[2:]
        print(self.data_files)

    def __getitem__(self, idx):
        
        #paths are stored in data_files, so use idx to get the path
        path1=self.data_files[idx][0]
        path2=self.data_files[idx][1]
        path3=self.data_files[idx][2]

        img1 = Image.open(self.path+'/'+path1)
        img1 = img1.convert('RGB')
        img1 = img1.resize((128,128))
        img1 = np.asarray(img1)/255
        img1 = np.transpose(np.float32(img1), (2,0,1))

        img2 = Image.open(self.path+'/'+path2)
        img2 = img2.convert('RGB')
        img2 = img2.resize((128,128))
        img2 = np.asarray(img2)/255
        img2 = np.transpose(np.float32(img2), (2,0,1))
        
        

        instance = torch.cat((torch.tensor(img1),torch.tensor(img2)),0)
        
        img3 = Image.open(self.path+'/'+path3)
        img3 = img3.convert('RGB')
        img3 = img3.resize((128,128))
        img3 = np.asarray(img3)/255
        img3 = np.transpose(np.float32(img3), (2,0,1))

        label = torch.tensor(img3)

        #make a 4d tensor
        #([2,x,y,3]) 2 for 2 frames, x,y for the size of the image, 3 for RGB
        #instance=torch.tensor(np.array([img1,img2]))
        

        return instance, label


    def __len__(self):
        return len(self.data_files)



def make_dataset(path, workers=4, batch_size=4):
    #not currently shuffling the data
    #should probably normalize the data at some stage
    dset=MyDataset(path)
    return torch.utils.data.DataLoader(dset, batch_size=batch_size)#num_workers=workers, 