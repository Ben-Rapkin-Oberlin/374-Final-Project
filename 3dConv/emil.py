

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *




# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        
        #currently not sure is there is a relationship between 
        #number arguments in conv_layer and Linear layer
        # and the shape of the tensor, so for now I will just
        # leave them be


        self.conv_layer1 = self._conv_layer_set(3, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)


        self.fc1 = nn.Linear(2**3*64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        #kernal size is window, for now a 3x3x3 works 
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        
        return out

class Generator(Module):
    def __init__(self):
 
        # calling constructor of parent class
        super().__init__()
        nz = 100
        ngf = 64
        self.gen = Sequential(
            # ConvTranspose2d(in_channels = 100, out_channels =  ngf * 16 , kernel_size = 4, stride = 1, padding = 0, bias = False),
            # # the output from the above will be b_size ,512, 4,4
            # BatchNorm2d(num_features = ngf * 16), # From an input of size (b_size, C, H, W), pick num_features = C
            # LeakyReLU(inplace = True),


            ConvTranspose2d(in_channels = 100, out_channels =  ngf * 8 , kernel_size = 4, stride = 1, padding = 0, bias = False),
            # the output from the above will be b_size ,512, 4,4
            BatchNorm2d(num_features = ngf * 8), # From an input of size (b_size, C, H, W), pick num_features = C
            LeakyReLU(inplace = True),
            
            
            
            ConvTranspose2d(in_channels = ngf * 8, out_channels =  ngf * 4 , kernel_size = 4, stride = 2, padding = 1, bias = False),
            # the output from the above will be b_size ,256, 8,8
            BatchNorm2d(num_features =  ngf * 4),
            LeakyReLU(inplace = True),

            ConvTranspose2d(in_channels =  ngf * 4, out_channels =  ngf * 2 , kernel_size = 4, stride = 2, padding = 1, bias = False),
            # the output from the above will be b_size ,128, 16,16
            BatchNorm2d(num_features = ngf * 2),
            LeakyReLU(inplace = True),

            ConvTranspose2d(in_channels =  ngf * 2, out_channels =  ngf , kernel_size = 4, stride = 2, padding = 1, bias = False),
            # the output from the above will be b_size ,128, 16,16
            BatchNorm2d(num_features = ngf),
            LeakyReLU(inplace = True),

 
            ConvTranspose2d(in_channels = ngf, out_channels = 3 , kernel_size = 4, stride = 2, padding = 1, bias = False),
            # the output from the above will be b_size ,3, 32,32
            Tanh()
         
        )