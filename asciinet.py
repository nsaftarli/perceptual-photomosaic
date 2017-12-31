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


# x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
# y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

# print(x_train.get_shape)
# print(y_train.get_shape)

samples = 202533
text_rows = 224
text_cols = 224
dims = 16
split = 100

x_test = x_train[:split]
x_val = x_train[split:(2 * split)]
x_train = x_train[(2 * split):]




y_test = y_train[:split]
y_val = y_train[split:(2 * split)]
y_train = y_train[(2 * split):]



model = models.Sequential()


input_tensor = Input(shape=(224,224,3))
x = VGG16(weights='imagenet', include_top=False, input_shape=(text_rows,text_cols,3))(input_tensor)
# x = layers.Conv2DTranspose(512, 3, padding='same', activation='softmax')(x)
# x = layers.Conv2DTranspose(512, 3,  activation='softmax')(x)
# x = layers.Conv2DTranspose(512, 3,  activation='softmax')(x)
# x = layers.Conv2DTranspose(512, 3,  activation='softmax')(x)
# x = layers.Conv2DTranspose(512, 3,  activation='softmax')(x)
# x = layers.Conv2DTranspose(512, 3,  activation='softmax')(x)
# x = layers.Conv2DTranspose(256, 3,  activation='softmax')(x)



# x = layers.Conv2DTranspose(64, 3, padding='same', activation='softmax')(x)
# x = layers.Conv2DTranspose(32, 3, padding='same', activation='softmax')(x)
# output_tensor = layers.Conv2DTranspose(16, 3, padding='same', activation='softmax')(x)
l = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
l.summary()


x = layers.UpSampling2D()(x)
x = layers.Conv2D(256,2, activation='softmax',padding='same')(x)
x = layers.UpSampling2D()(x)
x = layers.Conv2D(128,2, activation='softmax',padding='same')(x)
x = layers.UpSampling2D()(x)
x = layers.Conv2D(64,2, activation='softmax',padding='same')(x)
x = layers.UpSampling2D()(x)
x = layers.Conv2D(32,2, activation='softmax',padding='same')(x)
x = layers.UpSampling2D()(x)
x = layers.Conv2D(16,2, activation='softmax',padding='same')(x)

# x = layers.Conv2D(4,2, activation='softmax',padding='same')(x)
# x = layers.UpSampling2D()(x)
# x = layers.Conv2D(5,2, activation='softmax',padding='same')(x)
# x = layers.UpSampling2D()(x)
# x = layers.Conv2D(6,2, activation='softmax',padding='same')(x)
# x = layers.UpSampling2D()(x)
# x = layers.Conv2D(7,2, activation='softmax',padding='same')(x)



# x = layers.UpSampling2D(size=(2,2))(x)
# x = layers.UpSampling2D(size=(2,2))(x)
# x = layers.UpSampling2D(size=(2,2))(x)
# x = layers.UpSampling2D(size=(2,2))(x)
# x = layers.UpSampling2D(size=(2,2))(x)

# x = layers.ZeroPadding2D(padding=(1,1))(x)
# x = layers.ZeroPadding2D(padding=(1,1))(x)
# x = layers.ZeroPadding2D(padding=(1,1))(x)
# x = layers.ZeroPadding2D(padding=(1,1))(x)
# x = layers.ZeroPadding2D(padding=(1,1))(x)
# x = layers.ZeroPadding2D(padding=(1,1))(x)
# x = layers.ZeroPadding2D(padding=(1,0))(x)
# x = layers.ZeroPadding2D(padding=(1,0))(x)
# x = layers.ZeroPadding2D(padding=(1,0))(x)

# x = layers.ZeroPadding2D(padding=(1,0))(x)

# x = layers.ZeroPadding2D(padding=(1,1))(x)

# x = layers.ZeroPadding2D(padding=(1,1))(x)

# x = layers.ZeroPadding2D(padding=(1,1))(x)

# x = layers.ZeroPadding2D(padding=(1,1))(x)
# x = layers.ZeroPadding2D(padding=(1,1))(x)
# x = layers.ZeroPadding2D(padding=(1,1))(x)
# x = layers.ZeroPadding2D(padding=(1,1))(x)

# x = layers.Conv2D(512, (3,3), padding='valid')(x)
# x = layers.BatchNormalization()(x)
# x = layers.UpSampling2D(size=(2,2))(x)

# x = layers.ZeroPadding2D(padding=(1,1))(x)
# x = layers.Conv2D(256, (3,3), padding='valid')(x)
# x = layers.BatchNormalization()(x)
# x = layers.UpSampling2D(size=(2,2))(x)


# x = layers.ZeroPadding2D(padding=(1,1))(x)
# x = layers.Conv2D(128, (3, 3), padding='valid')(x)
# x = layers.BatchNormalization()(x)
# x = layers.UpSampling2D(size=(2,2))(x)


# x = layers.ZeroPadding2D(padding=(1,1))(x)
# x = layers.Conv2D(64, 3, 3, border_mode='valid')(x)
# x = layers.BatchNormalization()(x)
# x - layers.UpSampling2D(size=(2,2))(x)


# x = layers.ZeroPadding2D(padding=(1,1))(x)
# x = layers.Conv2D(32, 3, 3, border_mode='valid')(x)
# x = layers.BatchNormalization()(x)
# x - layers.UpSampling2D(size=(2,2))(x)



# x = layers.ZeroPadding2D(padding=(1,1))(x)
# x = layers.Conv2D(16, 3, 3, border_mode='valid')(x)
# x = layers.BatchNormalization()(x)
# x - layers.UpSampling2D(size=(2,2))(x)
# x = layers.UpSampling2D(2)(x)
# x = layers.BatchNormalization()(x)

# output_tensor = layers.Conv2D(16, 3, activation='softmax')(x)

model = Model(input_tensor,x)

model.summary()

model.compile(
	loss='categorical_crossentropy',
	optimizer=optimizers.SGD(lr=1e-4),
	metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_val,y_val))


# model.add(layers.Conv2D(512,3, padding='same',data_format='channels_last',activation='softmax',input_shape=(218,178,3)))
# model.add(layers.Conv2D(256,3, padding='same',data_format='channels_last',activation='softmax'))
# model.add(layers.Conv2D(128,3, padding='same',data_format='channels_last',activation='softmax'))
# model.add(layers.Conv2D(64, 3, padding='same', data_format='channels_last',activation='softmax'))
# model.add(layers.Conv2D(32,3, padding='same',data_format='channels_last',activation='softmax'))

# model.add(conv_base)
# model.add(conv_head)

# model.add(layers.MaxPooling2D((2,2), strides=None,padding='same'))


# model.add(layers.Conv2D(16,3,padding='same', activation='softmax'))




# model.summary()

# model.compile(
# 	loss='categorical_crossentropy',
# 	optimizer=optimizers.SGD(lr=1e-4),
# 	metrics=['accuracy'])

# history = model.fit(x_train,y_train, epochs=30, batch_size=32, validation_data=(x_val,y_val))
# test_loss, test_acc = model.evaluate(x_test,y_test)

# print(test_loss)
# print(test_acc)



