import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import numpy as np

x_train= tfds.load('div2k',split='train', as_supervised=True)

#print(x_train)
#print(x_train.take(1))

def preprocess(x,y):
  """print(type(x))
  print(x)
  print(type(y))
  print(y)"""
  
  #tensor = x['hr']
  tensor1 = tf.image.resize(x, (256, 256))
  tensor1 = tf.image.resize(tensor1,  (512, 512))

  tensor2 = tf.image.resize(y, (512, 512))


  return tensor1, tensor2

x_train = x_train.map(lambda x,y: preprocess(x,y))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=14000)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

device_name = tf.test.gpu_device_name()
with tf.device(device_name):
  input_img = tf.keras.layers.Input(shape=(512, 512, 3))

  # Increase the number of filters in the convolutional layers to
  # better handle the increased resolution of the input and output images
  l1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(input_img)
  l2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l1)
  l3 = tf.keras.layers.MaxPool2D(padding='same')(l2)

  l4 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l3)
  l5 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l4)
  l6 = tf.keras.layers.MaxPool2D(padding='same')(l5)

  l7 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l6)

  # Add additional layers to handle the increased resolution of the input and output images
  l8 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l7)
  l9 = tf.keras.layers.UpSampling2D()(l8)
  l10 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l9)
  l11 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l10)

  l12 = tf.keras.layers.add([l11, l5])

  # Add additional layers to handle the increased resolution of the input and output images
  l13 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l12)
  l14 = tf.keras.layers.UpSampling2D()(l13)
  l15 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l14)
  l16 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l15)

  l17 = tf.keras.layers.add([l16, l2])

  decoded_image = tf.keras.layers.Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l17)

  auto_encoder = tf.keras.models.Model(inputs=(input_img), outputs=decoded_image)
  auto_encoder.compile(optimizer='adadelta', loss='mean_squared_error',metrics=['accuracy'])

x_train=x_train.batch(10)
auto_encoder.fit(
    x=x_train,
    epochs=4
)
auto_encoder.save('upscaler.h5')