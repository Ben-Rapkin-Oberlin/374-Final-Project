import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
#t=tfds.load('div2k', split='train', shuffle_files=True)


import tensorflow as tf
import tensorflow_datasets as tfds

x_train= tfds.load('div2k',split='train', as_supervised=False)
y_train= tfds.load('div2k',split='train', as_supervised=False)
print(type(x_train))
print(type(y_train))


x_train=x_train.batch(10)
y_train=y_train.batch(10)





print(type(train_batches))
print(train_batches.take(1))

#train_batches=train_batches.map(lambda x: (tf.image.resize(x, (256,256)),3))
#small=data.map(lambda x,y: (tf.image.resize(x, (256,256)), y))
#big=data.map(lambda x,y: (tf.image.resize(x, (512,512)), y))


#train=tfds.as_numpy(small)
print("##############\n")
exit()
print(type(train))
#print(train.shape)

test=tfds.as_numpy(big)


#data= tfds.load('fashion_mnist', split='train', as_supervised=True)

"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Resizing(
    height=28,
    width=28),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', 
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='Adam', loss='binary_crossentropy',
metrics=['accuracy'])

history = model.fit(train_batches, epochs=10)
"""

#resize=tf.keras.layers.Resizing(height=256, width=256)
input_img = tf.keras.layers.Input(shape=(256, 256, 3))
#l0 = tf.keras.layers.add([resize, input_img])
#resize=tf.keras.layers.Resizing(height=256, width=256)(input_img)

l1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(input_img)
l2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l1)
l3 = tf.keras.layers.MaxPool2D(padding='same')(l2)

l4 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l3)
l5 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l4)
l6 = tf.keras.layers.MaxPool2D(padding='same')(l5)

l7 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l6)

l8 = tf.keras.layers.UpSampling2D()(l7)
l9 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l8)
l10 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l9)

l11 = tf.keras.layers.add([l10, l5])

l12 = tf.keras.layers.UpSampling2D()(l11)
l13 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l12)
l14 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l13)

l15 = tf.keras.layers.add([l14, l2])

decoded_image = tf.keras.layers.Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l15)

auto_encoder = tf.keras.models.Model(inputs=(input_img), outputs=decoded_image)


auto_encoder.compile(optimizer='adadelta', loss='mean_squared_error',metrics=['accuracy']) 
#consider trying Perceptual Loss
#Adam optimizer instead of Adadelta.

#Conv2D Transpose layers rather than Upscaling layers.????



#get data
#tf.config.experimental_run_functions_eagerly(True)
#auto_encoder.summary()

#train model



auto_encoder.fit(train,
        test,
        epochs=4,
        batch_size=1,
        shuffle=True,

        )
