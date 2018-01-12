import tensorflow as tf 
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers import *
from keras.models import Sequential, Model
from keras import Input
from keras.applications import VGG16
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import imgdata
from sklearn.utils import class_weight 

model = models.load_model('ascii_nn4.h5')
model.summary()


(x_train, y_train) = imgdata.load_data(batch_size=10000)

x_train = x_train.astype('float32')



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



model.compile(
	loss='categorical_crossentropy',
	optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, decay=5e-3),
	metrics=['accuracy']
	
)

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val,y_val))

model.save('ascii_nn5.h5')

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
