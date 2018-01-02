import tensorflow as tf 
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
#session = tf.Session(config=config)
set_session(tf.Session(config=config))


import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras import layers
from keras import models
from keras.models import Sequential, Model
from keras import Input
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
import imgdata

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# set_session(tf.Session(config=config))

(x_train, y_train) = imgdata.load_data()

x_train = x_train.astype('float32')
x_train /= 255

# x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
# y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

samples = 202533
text_rows = 224
text_cols = 224
dims = 16
split = 1000

x_test = x_train[:split]
x_val = x_train[split:(2 * split)]
x_train = x_train[(2 * split):]




y_test = y_train[:split]
y_val = y_train[split:(2 * split)]
y_train = y_train[(2 * split):]



model = models.Sequential()


input_tensor = Input(shape=(224,224,3))

#Encoder layer
x = VGG16(weights='imagenet', include_top=False, input_shape=(text_rows,text_cols,3))(input_tensor)

VGG16(weights='imagenet', include_top=False, input_shape=(text_rows,text_cols,3)).summary()

#Decoder layers
x = layers.UpSampling2D()(x)
x = layers.Conv2D(512,3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(512,3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(512,3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)

x = layers.UpSampling2D()(x)
x = layers.Conv2D(512,3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(512,3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(512,3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)

x = layers.UpSampling2D()(x)
x = layers.Conv2D(256,3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256,3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256,3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)

x = layers.UpSampling2D()(x)
x = layers.Conv2D(128,3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128,3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)

x = layers.UpSampling2D()(x)
x = layers.Conv2D(64,3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64,3, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
#Classification layer
x = layers.Conv2D(16,1, activation='softmax', padding='same')(x)


model = Model(input_tensor,x)

model.summary()

model.compile(
	loss='categorical_crossentropy',
	optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
	metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_val,y_val))



model.save('ascii_nn2.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo')
plt.plot(epochs, val_acc, 'b')
plt.title('Train/Val Accuracy')

plt.figure()

plt.plot(epochs,loss,'bo')
plt.plot(epochs,val_loss,'b')
plt.title('Train/Val Loss')

plt.show()
