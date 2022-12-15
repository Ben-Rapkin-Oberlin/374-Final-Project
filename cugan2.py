import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
from numpy import asarray
from numpy import savez_compressed

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, LeakyReLU, BatchNorm2d, ReLU, Tanh, Sigmoid, BCELoss 

import lazyload

 # dir_data is the name of the folder where the images are stored. 
def process_data(dir_data):
    #
    img_shape = (256,256, 3)
    nm_imgs   = np.sort(os.listdir(dir_data))
    print(nm_imgs)
    X_train = []

    #read, resize and normalize
    for file in nm_imgs:
        try:
            img = Image.open(dir_data+'/'+file)
            img = img.convert('RGB')
            img = img.resize((256,256))
            img = np.asarray(img)/255
            
            X_train.append(img)
        except:
            print("something went wrong")
    
    X_train = np.array(X_train)
    print (X_train.shape)

    #save the processed data
    savez_compressed('data_256x256.npz', X_train)
    print("X_train")
    print(X_train[0])
    # load dict of arrays
    dict_data = np.load('data_256x256.npz')
    
    # extract the first array
    data = dict_data['arr_0']
    


#plot the generated images and save it. 
def plot_images(imgs, grid_size = 5, index=0):
     
    columns = rows = grid_size
    #this i cannot be greater than the number of images in a batch. 
    for i in range (1):
        plt.imshow(imgs[i])
        plt.axis("off")
        plt.savefig('generation/'+str(index)+'-'+str(i)+'generation.png',dpi='figure')



# def test_data(imgs):
#     plot_images(imgs['arr_0'], 3)

#class dataset
class data_set(Dataset):
 
    def __init__(self, npz_imgs):
        self.imgs = npz_imgs

 
    def __len__(self):
        return len(self.imgs)
 
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
 
        image = self.imgs[idx]
 
        return image


#generator class, now 256x256
class Generator(Module):
    def __init__(self):
 
        super().__init__()
        nz = 100
        ngf = 64
        self.gen = Sequential(
            #in_channels was 100
            ConvTranspose2d(in_channels = 100, out_channels =  ngf * 32 , kernel_size = 4, stride = 1, padding = 0, bias = False),
            BatchNorm2d(num_features = ngf * 32),
            LeakyReLU(inplace = True),

            ConvTranspose2d(in_channels = ngf * 32, out_channels =  ngf * 16 , kernel_size = 4, stride = 2, padding = 1, bias = False),    
            BatchNorm2d(num_features = ngf * 16), 
            LeakyReLU(inplace = True),

            ConvTranspose2d(in_channels = ngf * 16, out_channels =  ngf * 8 , kernel_size = 4, stride = 2, padding = 1, bias = False),
            BatchNorm2d(num_features = ngf * 8), 
            LeakyReLU(inplace = True),
            
            ConvTranspose2d(in_channels = ngf * 8, out_channels =  ngf * 4 , kernel_size = 4, stride = 2, padding = 1, bias = False),
            BatchNorm2d(num_features =  ngf * 4),
            LeakyReLU(inplace = True),

            ConvTranspose2d(in_channels =  ngf * 4, out_channels =  ngf * 2 , kernel_size = 4, stride = 2, padding = 1, bias = False),
            BatchNorm2d(num_features = ngf * 2),
            LeakyReLU(inplace = True),

            ConvTranspose2d(in_channels =  ngf * 2, out_channels =  ngf , kernel_size = 4, stride = 2, padding = 1, bias = False),
            BatchNorm2d(num_features = ngf),
            LeakyReLU(inplace = True),

            ConvTranspose2d(in_channels = ngf, out_channels = 3 , kernel_size = 4, stride = 2, padding = 1, bias = False),
            Tanh()
        )
 
    def forward(self, input):
        return self.gen(input)
 
class Discriminator(Module):
    def __init__(self):
 
        super().__init__()
        w = 64
        
        
        self.dis = Sequential(
 
            # input is (3, 32, 32)
            Conv2d(in_channels = 3, out_channels = w, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 32, 16, 16
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = w, out_channels = w*2, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 32*2, 8, 8
            BatchNorm2d(w * 2),
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = w*2, out_channels = w*4, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 32*4, 4, 4
            BatchNorm2d(w * 4),
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = w*4, out_channels = w*8, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 256, 2, 2
            # NOTE: spatial size of this layer is 2x2, hence in the final layer, the kernel size must be 2 instead (or smaller than) 4
            BatchNorm2d(w * 8),
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = w*8, out_channels = w * 16, kernel_size = 4, stride = 2, padding = 1, bias=False),
            BatchNorm2d(w * 16),
            LeakyReLU(0.2, inplace=True),

            Conv2d(in_channels = w*16, out_channels = w*32, kernel_size = 4, stride = 2, padding = 1, bias=False),
            BatchNorm2d(w * 32),
            LeakyReLU(0.2, inplace=True),

            Conv2d(in_channels = w*32, out_channels = 1, kernel_size = 4, stride = 1, padding = 0, bias=False),
            # ouput from above layer is b_size, 1, 1, 1
            Sigmoid()
        )
     
    def forward(self, input):
        return self.dis(input)


def init_weights(m):
    if type(m) == ConvTranspose2d:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif type(m) == BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)



def train(imgs,device,epochs,gen_epochs,path):
    
    
    #transpose_imgs = np.transpose(np.float32(imgs['arr_0']), (0, 3, 1, 2)) #re-structure the data
 
    #dset = lazyload.MyDataset(transpose_imgs) # passing the npz variable to the constructor class
    #batch_size = 3
    shuffle = True
    
    #dataloader = DataLoader(dataset = dset, batch_size = 1, shuffle = shuffle)
    dataloader = lazyload.make_dataset(path, workers=1, batch_size = 1)
    
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    # initializing the weights
    netD.apply(init_weights)
    netG.apply(init_weights)
    
    opt_D = optim.Adam(netD.parameters(), lr = 3e-4, betas= (0.5, 0.999))
    opt_G = optim.Adam(netG.parameters(), lr = 3e-4, betas= (0.5, 0.999))#lr=0.0002

    loss = BCELoss()


    for epoch in range(epochs):

        for i, b in enumerate(dataloader):
            # Loss on real images
            #print(b.shape)
            label=b[1]
            attribute = b[0]
            #b = torch.transpose(b,(2,0,1))
            # stack_img = torch.cat((b[0],b[2]),0)
            # ground_truth = b[1]
            # clear the gradient
            opt_D.zero_grad() # set the gradients to 0 at start of each loop because gradients are accumulated on subsequent backward passes
            # compute the D model output
            
            yhat = netD(label.to(device)).view(-1)
            #yhat = netD(b.to(device)).view(-1) # view(-1) reshapes a 4-d tensor of shape (2,1,1,1) to 1-d tensor with 2 values only
            # specify target labels or true labels
            target = torch.ones(len(b[0]), dtype=torch.float, device=device)
            # calculate loss
            
            loss_real = loss(yhat, target)
            # calculate gradients -  or rather accumulation of gradients on loss tensor
            loss_real.backward()
    
            # Loss on fake images
    
            # generate batch of fake images using G
            # Step1: creating noise to be fed as input to G
            #G_input = attribute
            noise = torch.randn(len(b[0]), 100, 1, 1, device = device)
            # Step 2: feed noise to G to create a fake img (this will be reused when updating G)
            fake_img = netG(noise) 
            #fake_img = netG(G_input.to(device))
    
            # compute D model output on fake images
            yhat = netD.cuda()(fake_img.detach()).view(-1) 
            
            # specify target labels
            target = torch.zeros(len(b[0]), dtype=torch.float, device=device)
            # calculate loss
            loss_fake = loss(yhat, target)
            # calculate gradients
            loss_fake.backward()
    
            # total error on D
            loss_disc = loss_real + loss_fake
    
            # Update weights of D
            opt_D.step()
    
            ##########################
            #### Update Generator ####
            ##########################
    
            # clear gradient
            opt_G.zero_grad()
            # pass fake image through D
            yhat = netD.cuda()(fake_img).view(-1)
            # specify target variables - remember G wants D *to think* these are real images so label is 1
            target = torch.ones(len(b[0]), dtype=torch.float, device=device)
            # calculate loss
            loss_gen = loss(yhat, target)
            # calculate gradients
            loss_gen.backward()
            # update weights on G
            opt_G.step()
    
    
            ####################################
            #### Plot some Generator images ####
            ####################################
    
            # during every epoch, print images at every 10th iteration.
            if i% 10 == 0:
                # convert the fake images from (b_size, 3, 32, 32) to (b_size, 32, 32, 3) for plotting 
                
                #print("********************")
                print(" Epoch %d and iteration %d dloss= %f gloss= %f " % (epoch, i,loss_disc, loss_gen))
        if epoch % 50 == 0:
            img_plot = np.transpose(fake_img.detach().cpu(), (0,2,3,1)) # .detach().cpu() is imp for copying fake_img tensor to host memory first
            plot_images(img_plot,index=epoch)


def main():
    #print('1')
    
    dev = 'cuda' if torch.cuda.is_available() == True else 'cpu'
    device = torch.device(dev)
    #process_data('images/low-res_mini')
    imgs = np.load('data_256x256.npz')
    
    #test_data(imgs)
    epochs = 1000
    path = 'images/low-res_mini'
    train(imgs, device, epochs, 1500, path)
    
main()