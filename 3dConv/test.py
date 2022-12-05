import numpy as np
from PIL import Image
import torch

a1=np.asarray(Image.open('..\\images\\low-res\\00001.jpg'))
a2=np.asarray(Image.open('..\\images\\low-res\\00002.jpg'))
a=np.array([a1,a2])
#a=np.vstack((a1, a2))
a=torch.tensor(a)
print (a.shape)
