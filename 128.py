#defining model

#read in matrix of data, x * y * 3 * 2

from fourd.py import Conv4D


a=


height=20
width=20

model=Sequential()
#adding convolution layer
#channels are last
# two images, each of size heightxwidth, 3 channels
input_shape =(2, height, width, 3)
x = tf.random.normal(input_shape)
# 2 is output space, 3 is kernal size? 
model.add.Conv3D(2, 3, activation='relu', input_shape=input_shape[1:])
print(y.shape)
(4, 26, 26, 26, 2)



model=Sequential()
#adding convolution laye


model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
#adding pooling layer
model.add(MaxPool2D(2,2))
#adding fully connected layer
model.add(Flatten())
model.add(Dense(100,activation='relu'))
#adding output layer
model.add(Dense(10,activation='softmax'))
#compiling the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#fitting the model
model.fit(X_train,y_train,epochs=10)