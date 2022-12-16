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
import generator
import discriminator


#plot the generated images and save it. 
def plot_images(imgs, grid_size = 5, index=0):
     
    columns = rows = grid_size
    #this i cannot be greater than the number of images in a batch. 
    for i in range(len(imgs)):
        plt.imshow(imgs[i])
        plt.axis("off")
        plt.savefig('generation/'+str(index)+'-'+str(i)+'generation.png',dpi='figure')



def init_weights(m):
    if type(m) == ConvTranspose2d:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif type(m) == BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)



def train(device,epochs,path):
    shuffle = True

    dataloader = lazyload.make_dataset(path, workers=1, batch_size = 1)
    
    netG = generator.Generator().to(device)
    netD = discriminator.Discriminator().to(device)
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
            
            #noise = torch.randn(len(b[0]), 100, 1, 1, device = device)
            # Step 2: feed noise to G to create a fake img (this will be reused when updating G)
            #fake_img = netG(noise) 
            fake_img = netG(attribute.to(device))
            print(fake_img.shape)
    
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
            if i% 20 == 0:
                # convert the fake images from (b_size, 3, 32, 32) to (b_size, 32, 32, 3) for plotting 
                
                #print("********************")
                print(" Epoch %d and iteration %d dloss= %f gloss= %f " % (epoch, i,loss_disc, loss_gen))
        if (epoch) % 50 == 0:
            
            img_plot = np.transpose(fake_img.detach().cpu(), (0,2,3,1)) # .detach().cpu() is imp for copying fake_img tensor to host memory first
            plot_images(img_plot,index=epoch)
            
            torch.save({
            'epoch': epoch,
            'generator_state_dict': netG.state_dict(),
            'discriminator_state_dict': netD.state_dict(),
            'G_optimizer_state_dict': opt_G.state_dict(),
            'D_optimizer_state_dict': opt_D.state_dict(),
            'G_loss': loss_gen,
            'D_loss': loss_disc
            }, 'checkpoints/'+str(epoch))
        #     img_plot = np.transpose(fake_img.detach().cpu(), (0,2,3,1)) # .detach().cpu() is imp for copying fake_img tensor to host memory first
        #     plot_images(img_plot,index=epoch)


def main():
    #print('1')
    
    dev = 'cuda' if torch.cuda.is_available() == True else 'cpu'
    device = torch.device(dev)
    #process_data('images/low-res_mini')
    #imgs = np.load('data_256x256.npz')
    
    #test_data(imgs)
    epochs = 1000
    path = 'images/low-res_mini'
    train(device, epochs, path)
    
main()