import os
import numpy as np
import urllib.request
import cv2


def store_raw_images(images_link):
    images_urls = urllib.request.urlopen(images_link).read().decode()

    if not os.path.exists("Images"):
        os.makedirs("Images")

    for i in images_urls.split("\n"):
        try:
            print(i)
            urllib.request.urlretrieve(i, f"Images/{str(len(os.listdir(f'Images'))+1)}.jpg")


        except Exception as e:
            print(str(e))

#this link no longer works
store_raw_images(
    "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02958343")

def get_training_data(images_location):
  
  real_image_treat_as_y = []
  downsize_image_treat_as_x = []

  for img in os.listdir(images_location):

    try:
      image = cv2.imread(f"{images_location}/{img}", cv2.IMREAD_UNCHANGED)

      reshaped_image = cv2.resize(image, (256, 256))

      if reshaped_image.shape[-1] == 3:
        real_image_treat_as_y.append(reshaped_image)

      image = cv2.resize(image, (100, 100))

      reshaped_image = cv2.resize(image, (256, 256))

      if reshaped_image.shape[-1] == 3:
        downsize_image_treat_as_x.append(cv2.resize(image, (256, 256)))

    except Exception as e:
      # print(str(e))
      pass
      

  return (np.array(downsize_image_treat_as_x), np.array(real_image_treat_as_y))

downized_images, real_images = get_training_data('Images')
tf.config.experimental_run_functions_eagerly(True)
auto_encoder.fit(downized_images,
                 real_images,
                 epochs=4,
                 batch_size=10,
                 shuffle=True,
                 validation_split=0.15)



