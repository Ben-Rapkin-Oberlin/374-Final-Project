import numpy as np
from PIL import Image
import time

#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
#import keras 
from tensorflow.keras.utils import load_img, img_to_array

from os import listdir
from os.path import isfile, join

dir='images\\low-res'

onlyfiles = [dir+'\\'+f for f in listdir(dir) if isfile(join(dir, f))]
#print(len(onlyfiles))
#print(onlyfiles[0])
#exit()

# 3840 x 2160 x 3 x 3 = 74,649,600
# 720 x 1280 x 3 x 3= 2,764,800
# 480 x 640 x 3 x 3= 691,200


all_images = []
count=0
for file in onlyfiles:
    all_images=[np.asarray(Image.open(onlyfiles[x])) for x in range(10)]
    all_images=np.array(all_images)
    all_images=np.array([x.flatten() for x in all_images])
    print(all_images.shape)
    np.savetxt('data' +str(count)+'.csv', all_images, delimiter=',')
    count+=1
    print(count)
    onlyfiles = onlyfiles[10:]





