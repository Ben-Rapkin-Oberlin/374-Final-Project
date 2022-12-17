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

import cv2




model_path = 'goodmodels/30.pth'
data_path = 'images/low-res_mini'

dev = 'cuda' if torch.cuda.is_available() == True else 'cpu'
device = torch.device(dev)



G = generator.Generator().to(device)
#D = discriminator.Discriminator().to(device)
G_opt = BCELoss()
#D_opt = BCELoss()

checkpoint = torch.load(model_path)
G.load_state_dict(checkpoint['generator_state_dict'])
#D.load_state_dict(checkpoint['discriminator_state_dict'])
#G_opt.load_state_dict(checkpoint['G_optimizer_state_dict'])
#D_opt.load_state_dict(checkpoint['D_optimizer_state_dict'])

G.eval()
frames = np.sort(os.listdir(data_path))
for i in range(len(frames)-1):
    path1=frames[i]
    path2=frames[i+1]
    
    img1 = Image.open(data_path+'/'+path1)
    img1 = img1.convert('RGB')
    img1 = img1.resize((256,256))
    img1 = np.asarray(img1)/255
    img1 = np.transpose(np.float32(img1), (2,0,1))

    img2 = Image.open(data_path+'/'+path2)
    img2 = img2.convert('RGB')
    img2 = img2.resize((256,256))
    img2 = np.asarray(img2)/255
    img2 = np.transpose(np.float32(img2), (2,0,1))

    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    attribute = torch.cat((torch.tensor(img1),torch.tensor(img2)),1)
    
    print(attribute.shape)
    
    fake_img = G(attribute.to(device))
    
    img_plot = np.transpose(fake_img.detach().cpu(), (0,2,3,1))
    #print(img_plot[0].shape)
    #img = np.
    
    #img.save('run_results/'+str((i+1)*2)+'.jpg')
    img = np.asarray(img_plot[0])*255
    
    cv2.imwrite('run_results/'+str((i+1)*2)+'.jpg', img)
    # plt.imshow(img_plot[0])
    # plt.axis("off")
    # plt.savefig('run_results/'+str(i)+'.png',dpi='figure')



for i in range(len(frames)):
    path1 = frames[i]
    temp = Image.open(data_path+'/'+path1)
    temp = temp.convert('RGB')
    temp = temp.resize((256,256))
    temp.save('run_results/'+str((i+1)*2+1)+'.jpg')
    #cv2.imwrite('run_results/'+str((i+1)*2-1)+'.png', temp)
    
result_list = np.sort(os.listdir('run_results'))
video=cv2.VideoWriter('video.mp4',-1,60,(256,256))
for i in range(len(result_list)):
    img = cv2.imread('run_results/'+result_list[i])
    video.write(img)
video.release()
# torch.save({
#             'epoch': epoch,
#             'generator_state_dict': netG.state_dict(),
#             'discriminator_state_dict': netD.state_dict(),
#             'G_optimizer_state_dict': opt_G.state_dict(),
#             'D_optimizer_state_dict': opt_D.state_dict(),
#             'G_loss': loss_gen,
#             'D_loss': loss_disc
#             }, 'checkpoints/'+str(epoch))