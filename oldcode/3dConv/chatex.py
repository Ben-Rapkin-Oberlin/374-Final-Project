import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Define the network layers
        self.layer1 = nn.ConvTranspose2d(100, 512, 4, 1, 0)
        self.layer2 = nn.BatchNorm2d(512)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.layer5 = nn.BatchNorm2d(256)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.layer8 = nn.BatchNorm2d(128)
        self.layer9 = nn.ReLU()
        self.layer10 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.layer11 = nn.BatchNorm2d(64)
        self.layer12 = nn.ReLU()
        self.layer13 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
        self.layer14 = nn.Tanh()
        
    def forward(self, x):
        # Forward pass through the network
        x = x.view(-1, 100, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        
        # Reshape the output to be a 512x512 image with 3 channels
        return x.view(-1, 3, 512, 512)

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Define the network layers
        self.layer1 = nn.Conv2d(6, 64, 4, 2, 1)
        self.layer2 = nn.LeakyReLU(0.2)
        self.layer3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.layer4 = nn
