import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras import layers
from keras import models
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import imgdata



(x_train, y_train) = imgdata.load_data()
print(x_train.shape)
print(y_train.shape)



x_test = x_train[:100]
x_val = x_train[100:200]
x_train = x_train[200:]




y_test = y_train[:100]
y_val = y_train[100:200]
y_train = y_train[200:]



conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(218,178,3))

conv_head = layers.Conv2D(16,(3,3),activation='softmax')

model = models.Sequential()


# model.add(layers.Conv2D(512,3, padding='same',data_format='channels_last',activation='softmax',input_shape=(218,178,3)))
# model.add(layers.Conv2D(256,3, padding='same',data_format='channels_last',activation='softmax'))
# model.add(layers.Conv2D(128,3, padding='same',data_format='channels_last',activation='softmax'))
# model.add(layers.Conv2D(64, 3, padding='same', data_format='channels_last',activation='softmax'))
# model.add(layers.Conv2D(32,3, padding='same',data_format='channels_last',activation='softmax'))

model.add(conv_base)
model.add(conv_head)

# model.add(layers.MaxPooling2D((2,2), strides=None,padding='same'))


# model.add(layers.Conv2D(16,3,padding='same', activation='softmax'))




model.summary()

model.compile(
	loss='categorical_crossentropy',
	optimizer=optimizers.SGD(lr=1e-4),
	metrics=['accuracy'])

history = model.fit(x_train,y_train, epochs=30, batch_size=32, validation_data=(x_val,y_val))
test_loss, test_acc = model.evaluate(x_test,y_test)

print(test_loss)
print(test_acc)



