import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras import layers
from keras import models
from keras.models import Sequential, Model
from keras import Input
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import imgdata



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

#Classification layer
x = layers.Conv2D(16,1, activation='softmax', padding='same')(x)


model = Model(input_tensor,x)

model.summary()

model.compile(
	loss='categorical_crossentropy',
	optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=5e-4),
	metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_val,y_val))





