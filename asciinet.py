import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras import layers
from keras import models
from keras.applications import VGG16
import imgdata



char_array = ['M','N','H','Q','0','C','7','>','!',':','-',';','.',' ']

(x_train, y_train) = imgdata.load_data()
print(x_train.shape)
print(y_train.shape)

x_test = x_train[:100]
x_train = x_train[100:]

y_test = y_train[:100]
y_train = y_train[100:]



conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(218,178,3))
# conv_base.summary()

conv_head = layers.Conv2D(4,(3,3),activation='relu')

model = models.Sequential()
model.add(conv_base)
model.add(conv_head)

model.summary()

model.compile(
	loss='categorical_crossentropy',
	optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, decay=5e-4),
	metrics=['accuracy'])

history = model.fit(x_train,y_train, epochs=30)
test_loss, test_acc = model.evaluate(x_test,y_test)

print(test_loss)
print(test_acc)



