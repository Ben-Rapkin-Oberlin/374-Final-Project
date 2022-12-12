
from model import make_model
import tensorflow as tf
import numpy as np

#make model structure
tam=make_model()
tam.compile(optimizer='adadelta', loss='mean_squared_error') 
#consider trying Perceptual Loss
#Adam optimizer instead of Adadelta.

#Conv2D Transpose layers rather than Upscaling layers.????



#get data
downized_images, real_images = get_training_data('Images')
tf.config.experimental_run_functions_eagerly(True)


#train model
tam.fit(downized_images,
                 real_images,
                 epochs=4,
                 batch_size=10,
                 shuffle=True,
                 validation_split=0.15)








x_train_n, x_train_down = train_batches(just_load_dataset=True)
tam.load_weights("/content/Completed_Notebook_Data_Autoencoders/data/rez/sr.img_net.mse.final_model5.patch.weights.best.hdf5")
sr1 = np.clip(tam.predict(x_train_down), 0.0, 1.0)
image_index = 251








