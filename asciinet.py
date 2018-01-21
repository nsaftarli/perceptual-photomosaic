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
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
import imgdata
# from sklearn.utils import class_weight
# import loss
import predict
from itertools import product
import something

'''
Dataset settings
'''
text_rows = 224
text_cols = 224
dims = 16

img_data_dir = '/home/nsaftarl/Documents/ascii-art/ASCIIArtNN/assets/rgb_in/img_celeba/'
# ascii_data_dir = '/home/nsaftarl/Documents/ascii-art/ASCIIArtNN/assets/ascii_out/'
ascii_data_dir = '/home/nsaftarl/Documents/ascii-art/ASCIIArtNN/assets/ssim_imgs/'

char_array = np.asarray(['M','N','H','Q', '$', 'O','C', '?','7','>','!',':','-',';','.',' '])
char_dict = {'M':0,'N':1,'H':2,'Q':3,'$':4,'O':5,'C':6,'?':7,'7':8,'>':9,'!':10,':':11,'-':12,';':13,'.':14,' ':15}


def main(size=22000, split=1000, train_type='m'):

	#Get per character weights, use them for loss
	weights = mfb()
	wcc = weighted_categorical_crossentropy(weights)
	print('WEIGHTS: ' + str(weights))

	#Create the model
	model = build_model()

	#Compile the model with our loss function
	model.compile(
	loss=wcc,
	optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
	metrics=['accuracy']
	)


	if train_type is 'm':

		(x_train, y_train) = imgdata.load_data(batch_size=size)
		x_train = x_train.astype('float32')

		

		#Shuffle arrays.
		rng_state = np.random.get_state()
		np.random.shuffle(x_train)
		np.random.set_state(rng_state)
		np.random.shuffle(y_train)

		#Split both into train/val/test sets
		x_test = x_train[:split]
		x_val = x_train[split:(2 * split)]
		x_train = x_train[(2 * split):]

		y_test = y_train[:split]
		y_val = y_train[split:(2 * split)]
		y_train = y_train[(2 * split):]

		reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, min_lr=1e-5)

		history = model.fit(
			x_train, 
			y_train, 
			epochs=60, 
			batch_size=32, 
			validation_data=(x_val,y_val),
			callbacks=[reduce_lr]
			)
		model.save('ascii_nn8.h5')
		get_results(history)
		
	elif train_type is 'g':
		reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, min_lr=1e-4)
		history = model.fit_generator(use_generator(), steps_per_epoch=250, epochs=10)
		model.save('ascii_nn7.h5')
		


def build_model(vgg_train=False):
	input_tensor = Input(shape=(224,224,3))

	vgg = VGG16(weights='imagenet', include_top=False, input_shape=(text_rows,text_cols,3))
	# vgg.summary()

	if vgg_train is False:
		# Freeze VGG layers
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
	conv1 = add([conv1,o5])
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
	soft5 = Conv2D(dims, 1, strides=8, activation='softmax', padding='same')(batch5)


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
	# plt.subplots_adjust(top=2)

	plt.plot(epochs,loss,'bo', label='Training Loss')
	plt.plot(epochs,val_loss,'b', label='Validation Loss')
	plt.title('Train/Val Loss')
	plt.legend()

	plt.savefig('latest_fig.jpg')

	plt.show()

def use_generator():
	while True:
		(x_train, y_train) = imgdata.load_data(batch_size=32)
		yield (x_train, y_train)




def mfb():
	total_counts, appearances = predict.char_counts(textrows=28,textcols=28)
	total_counts = np.asarray(total_counts,dtype='float32')
	appearances = np.asarray(appearances,dtype='float32')
	print(total_counts.shape)
	print(appearances.shape)


	freq_counts = total_counts / appearances
	print("FREQUENCY COUNTS: ")
	print(freq_counts)



	median_freqs = np.median(total_counts) / freq_counts

	
	return median_freqs



def weighted_categorical_crossentropy(w):
    def loss(y_true, y_pred):
    	y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    	y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    	loss = y_true * K.log(y_pred) * w
    	loss = -K.sum(loss, -1)
    	return loss
    return loss 

main(train_type='m')