from keras import optimizers
from keras.layers import *
from keras.models import Sequential, Model
from keras import Input
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
from keras.backend.tensorflow_backend import set_session


import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import imgdata
import predict
from constants import Constants
import weighting


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

'''Data constants'''
const = Constants()
img_data_dir = const.img_data_dir
ascii_data_dir = const.ascii_data_dir
val_data_dir = const.val_data_dir
char_array = const.char_array
char_dict = const.char_dict
img_rows = const.img_rows
img_cols = const.img_cols
text_rows = const.text_rows
text_cols = const.text_cols
dims = const.char_count


epoch_accuracies = []



def main(size=22000, split=1000, train_type='g'):

	#Get per character weights, use them for loss
	weights = weighting.median_freq_balancing()
	# weights = [1] * 16
	wcc = weighted_categorical_crossentropy(weights)
	print('WEIGHTS: ' + str(weights))

	model = build_model()
	model.compile(
		loss=wcc,
		optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
		metrics=['accuracy']
		)

	#Use generators 
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5)
	history = model.fit_generator(
		imgdata.load_data(num_batches=5937,batch_size=32), 
		steps_per_epoch=5937,
		epochs=5,
		validation_data=imgdata.load_val_data(num_batches=393,batch_size=32),
		validation_steps=393,
		callbacks=[reduce_lr]
		)
	model.save('ascii_nn_gen.h5')
	get_results(history)

	# history = model.fit_generator(
	# 	imgdata.load_data(num_batches=5,batch_size=32), 
	# 	steps_per_epoch=5,
	# 	epochs=2,
	# 	validation_data=imgdata.load_val_data(batch_size=32),
	# 	validation_steps=31
	# 	)
	# # model.summary()
	# model.save('ascii_nn_gen.h5')
	# get_results(history)


		


def build_model(vgg_train=False):
	input_tensor = Input(shape=(None,None,3))

	vgg = VGG16(weights='imagenet', include_top=False, input_shape=(None, None,3))

	if vgg_train is False:
		for layer in vgg.layers:
			layer.trainable = False

	l1_1 = Model.get_layer(vgg, 'block1_conv1')
	l1_2 = Model.get_layer(vgg, 'block1_conv2')
	l1_p = Model.get_layer(vgg, 'block1_pool')

	l2_1 = Model.get_layer(vgg, 'block2_conv1')
	l2_2 = Model.get_layer(vgg, 'block2_conv2')
	l2_p = Model.get_layer(vgg, 'block2_pool')

	l3_1 = Model.get_layer(vgg, 'block3_conv1')
	l3_2 = Model.get_layer(vgg, 'block3_conv2')
	l3_3 = Model.get_layer(vgg, 'block3_conv3')
	l3_p = Model.get_layer(vgg, 'block3_pool')

	l4_1 = Model.get_layer(vgg, 'block4_conv1')
	l4_2 = Model.get_layer(vgg, 'block4_conv2')
	l4_3 = Model.get_layer(vgg, 'block4_conv3')
	l4_p = Model.get_layer(vgg, 'block4_pool')

	l5_1 = Model.get_layer(vgg, 'block5_conv1')
	l5_2 = Model.get_layer(vgg, 'block5_conv2')
	l5_3 = Model.get_layer(vgg, 'block5_conv3')
	l5_p = Model.get_layer(vgg, 'block5_pool')


	#Encoder: Basically re-building VGG layer by layer, because Keras's concat only takes tensors, not layers
	x = l1_1(input_tensor)
	o1 = l1_2(x)
	x = l1_p(o1)
	x = l2_1(x)
	o2 = l2_2(x)
	x = l2_p(o2)
	x = l3_1(x)
	x = l3_2(x)
	o3 = l3_3(x)
	x = l3_p(o3)
	x = l4_1(x)
	x = l4_2(x)
	o4 = l4_3(x)
	x = l4_p(o4)
	x = l5_1(x)
	x = l5_2(x)
	o5 = l5_3(x)
	x = l5_p(o5)

	#Decoder layers: VGG architecture in reverse with skip connections and dropout layers
	#Block 1
	up1 = UpSampling2D()(x)
	conv1 = Conv2D(512, 3, activation='relu', padding='same')(up1)
	conv1 = Conv2D(512, 3, activation='relu', padding='same')(conv1)
	conv1 = Conv2D(512, 3, activation='relu', padding='same')(conv1)
	# conv1 = add([conv1,o5])
	batch1 = BatchNormalization()(conv1)


	#Block 2
	up2 = UpSampling2D()(batch1)
	conv2 = Conv2D(512, 3, activation='relu', padding='same')(up2)
	conv2 = Conv2D(512, 3, activation='relu', padding='same')(conv2)
	conv2 = Conv2D(512, 3, activation='relu', padding='same')(conv2)
	conv2 = add([conv2,o4])
	batch2 = BatchNormalization()(conv2)


	#Block 3
	up3 = UpSampling2D()(batch2)
	conv3 = Conv2D(256, 3, activation='relu', padding='same')(up3)
	conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
	conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
	conv3 = add([conv3,o3])
	batch3 = BatchNormalization()(conv3)

	#Block 4
	up4 = UpSampling2D()(batch3)
	conv4 = Conv2D(128, 3, activation='relu', padding='same')(up4)
	conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
	conv4 = add([conv4,o2])
	batch4 = BatchNormalization()(conv4)

	#Block 5
	up5 = UpSampling2D()(batch4)
	conv5 = Conv2D(64, 3, activation='relu', padding='same')(up5)
	conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
	conv5 = add([conv5,o1])
	batch5 = BatchNormalization()(conv5)

	#Final prediction layer
	soft5 = Conv2D(dims, kernel_size=8, strides=8, activation='softmax', padding='same')(batch5)

	model = Model(input_tensor,soft5)
	model.summary()

	return model


def get_results(history):
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(len(acc))

	plt.figure(figsize=(10,10))

	plt.subplot(211)
	plt.plot(epochs, acc, 'bo', label='Training Accuracy')
	plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
	plt.title('Train/Val Accuracy')
	plt.legend()

	plt.subplot(212)
	plt.plot(epochs,loss,'bo', label='Training Loss')
	plt.plot(epochs,val_loss,'b', label='Validation Loss')
	plt.title('Train/Val Loss')
	plt.legend()

	plt.savefig('latest_fig_gen.jpg')
	plt.show()



def weighted_categorical_crossentropy(w):
    def loss(y_true, y_pred):
    	y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    	y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    	loss = y_true * K.log(y_pred) #* w
    	loss = -K.sum(loss, -1)
    	return loss
    return loss 


def on_batch_end():
	epoch_accuracies.append(predict.per_char_accs(size=20000,textrows=28,textcols=28))

def on_epoch_end(self, epoch, logs=None):
	print(K.eval(self.model.optimizer.lr))


main()