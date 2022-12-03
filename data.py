import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


import os
os.getcwd()
# place the files in your IDE working dicrectory .
labels = pd.read_csv(r'/aerialcactus/train.csv')
submission = pd.read_csv(r'/aerialcactus/sample_submission.csv)

train_path = r'/aerialcactus/train/train/'
test_path = r'/aerialcactus/test/test/'


label = 'Has Cactus', 'Hasn\'t Cactus'

#not sure where idx is coming from
for i,idx in enumerate(labels[labels['has_cactus'] == 1]['id'][-5:]):
    path = os.path.join(train_path,idx)
    ax[i].imshow(img.imread(path))


class CactiDataset(Dataset):
    def __init__(self, data, path , transform = None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        img_name,label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


#no validation or test phase for gan


train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ToTensor()])

train, valid_data = train_test_split(labels, stratify=labels.has_cactus, test_size=0.2)


train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle=False, num_workers=0)