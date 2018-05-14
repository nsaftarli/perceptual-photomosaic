import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras import optimizers
from keras.layers import *
from keras.models import Sequential, Model, load_model
from keras import Input
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, TensorBoard, CSVLogger, ModelCheckpoint
import keras.backend as K
from keras.backend.tensorflow_backend import set_session

import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imdata
import callbacks
# from ASCIIModel import ASCIIModel
# import predict
from constants import Constants
import weighting
import model
# import multipLayer


config = tf.ConfigProto(allow_soft_placement=True)
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
experiments_dir = const.experiments_dir

def build_model(vgg_train=False):
	# input_tensor = Input(shape=(None,None,3))
	input_tensor = Input(shape=(224,224,3))

	vgg = VGG16(weights='imagenet', include_top=False, input_shape=(None, None,3))



	# if vgg_train is False:
	# 	for layer in vgg.layers:
	# 		layer.trainable = False

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
	up1 = UpSampling2D(name='de_block1_up')(x)
	conv1 = Conv2D(512, 3, activation='relu', padding='same', name='de_block1_conv1')(up1)
	conv1 = Conv2D(512, 3, activation='relu', padding='same', name='de_block1_conv2')(conv1)
	conv1 = Conv2D(512, 3, activation='relu', padding='same', name='de_block1_conv3')(conv1)
	conv1 = add([conv1,o5])
	batch1 = BatchNormalization(name='de_block1_batch')(conv1)


	#Block 2
	up2 = UpSampling2D(name='de_block2_up')(batch1)
	conv2 = Conv2D(512, 3, activation='relu', padding='same', name='de_block2_conv1')(up2)
	conv2 = Conv2D(512, 3, activation='relu', padding='same', name='de_block2_conv2')(conv2)
	conv2 = Conv2D(512, 3, activation='relu', padding='same', name='de_block2_conv3')(conv2)
	conv2 = add([conv2,o4], name='de_block2_add')
	batch2 = BatchNormalization(name='de_block2_batch')(conv2)


	#Block 3
	up3 = UpSampling2D(name='de_block3_up')(batch2)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', name='de_block3_conv1')(up3)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', name='de_block3_conv2')(conv3)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', name='de_block3_conv3')(conv3)
	conv3 = add([conv3,o3], name='de_block3_add')
	batch3 = BatchNormalization(name='de_block3_batch')(conv3)

	#Block 4
	up4 = UpSampling2D(name='de_block4_up')(batch3)
	conv4 = Conv2D(128, 3, activation='relu', padding='same', name='de_block4_conv1')(up4)
	conv4 = Conv2D(128, 3, activation='relu', padding='same', name='de_block4_conv2')(conv4)
	conv4 = add([conv4,o2], name='de_block4_add')
	batch4 = BatchNormalization(name='de_block4_batch')(conv4)

	#Block 5
	up5 = UpSampling2D(name='de_block5_up')(batch4)
	conv5 = Conv2D(64, 3, activation='relu', padding='same', name='de_block5_conv1')(up5)
	conv5 = Conv2D(64, 3, activation='relu', padding='same', name='de_block5_conv2')(conv5)
	conv5 = add([conv5,o1], name='de_block5_add')
	batch5 = BatchNormalization(name='de_block5_batch')(conv5)

	#Final prediction layer
	soft5 = Conv2D(dims, kernel_size=8, strides=8, activation='softmax', padding='same', name='de_block6_softmax')(batch5)

	input_tensor2 = Input(shape=(28,28,16))
	print(soft5)
	print(input_tensor2)
	# x = Lambda(lambda x: K.tf.multiply(soft5,input_tensor2))
	x = multiply([soft5,input_tensor2])
	x = Lambda(lambda x: K.sum(x, axis=-1))(x)
	print(x)



	model = Model([input_tensor,input_tensor2],x)


	#Add linear combination
	# model = get_lin_comb(model)
	# model.summary()

	return model

def get_lin_comb(model):
	input_tensor2 = Input(shape=(28,28,16))
	print(model.get_shape())
	x = K.tf.multiply(model,input_tensor2)
	# x = K.sum(x, axis=3)
	return x

build_model()