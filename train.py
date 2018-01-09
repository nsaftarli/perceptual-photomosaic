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

model = models.load_model('ascii_nn4.h5')
model.summary()


(x_train, y_train) = imgdata.load_data(batch_size=10000)

x_train = x_train.astype('float32')
x_train /= 255
# x_train -= np.mean(x_train,axis=3, keepdims=True)




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

#Median Frequency Balancing:
class_freq = imgdata.get_class_weights(size=8000, targets=y_train)
class_freq = np.sort(class_freq)
median_freq = np.median(class_freq)
balanced_freq = median_freq/class_freq



# input_tensor = Input(shape=(224,224,3))

model.compile(
	loss='categorical_crossentropy',
	optimizer=optimizers.SGD(lr=1e-3, momentum=0.9, decay=5e-4),
	metrics=['accuracy']
	
)

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val,y_val), class_weight=balanced_freq)


# history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_val,y_val), class_weight=balanced_freq)

	# class_weight=[ 
	# 0.0931816,   0.07889178,  0.0683163,  0.0599477,  0.0576133,  
	# 0.0596811, 0.0649174,  0.0713340,  0.0777491,  0.0805642,  
	# 0.0774501,  0.0663205, 0.0514884,  0.0368726,  0.0278029,  
	# 0.0278682
	# ]

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