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


'''Settings'''
epochs=20
lrate = 0.0000001
batchsize = 8
batches = 23750
valbatches = 393
flip=True




def lr_sched(index):
	if index < 10:
		return 0.0001
	elif index >= 10 and index <  20:
		return 0.00025
	elif index > 20 and index < 30:
		return 0.00001
	else:
		return  float(0.00000125)


def main():
	dt = datetime.datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
	path = experiments_dir + dt + '/'

	#Get per character weights, use them for loss
	weights = weighting.median_freq_balancing()
	# weights = weighting.plain_weighting()
	# weights = np.asarray([1] * 16,dtype='float32')
	wcc = weighted_categorical_crossentropy(weights)
	print('WEIGHTS: ' + str(weights))

	# model = ASCIIModel(soft_depth=dims)
	model = build_model()
	model.compile(
		loss=wcc,
		optimizer=optimizers.RMSprop(lr=lrate),
		metrics=['accuracy']
		)

	#Use generators
	checkpoint = ModelCheckpoint(filepath=path + 'weights.{epoch:1d}.hdf5')
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3)
	lr_schedule = LearningRateScheduler(lr_sched)
	tf_board = TensorBoard(log_dir='./logs/Graph')
	csv_log = CSVLogger('training.log')

	logs_callback = callbacks.SettingLogs(logs={
		'exp_file':path,
		'epochs':epochs,
		'lr':lrate,
		'batch_size':batchsize,
		'flip':flip, 
		'weights':weights})

	checkpath = path + 'ascii_nn_checkpt.h5'
	char_acc_callback = callbacks.ClassAccs(logs={'model_dir':checkpath})
	history = model.fit_generator(
		imdata.load_data(num_batches=batches,batch_size=batchsize,flipped=flip,validation=False),
		steps_per_epoch=batches,
		epochs=30,
		validation_data=imdata.load_data(num_batches=valbatches,batch_size=batchsize,flipped=False,validation=True),
		validation_steps=valbatches,
		callbacks=[checkpoint,logs_callback]


		)

	model.save(path + 'ascii_nn.h5')
	get_results(history,path)





def build_model(vgg_train=False):
	input_tensor = Input(shape=(None,None,3))

	vgg = VGG16(weights='imagenet', include_top=False, input_shape=(None, None,3))


	# model = Conv2D(64,3,activation='relu', padding='same')(input_tensor)
	# model = Conv2D(64,3,activation='relu', padding='same')(model)
	# model = MaxPooling2D((2,2), strides=(2,2))(model)

	# model = Conv2D(128,3,activation='relu', padding='same')(model)
	# model = Conv2D(128,3,activation='relu', padding='same')(model)
	# model = MaxPooling2D((2,2), strides=(2,2))(model)

	# model = Conv2D(256,3,activation='relu',padding='same')(model)
	# model = Conv2D(256,3,activation='relu',padding='same')(model)
	# model = Conv2D(256,3,activation='relu',padding='same')(model)
	# model = MaxPooling2D((2,2), strides=(2,2))(model)

	# model = Conv2D(256,3,activation='relu',padding='same')(model)
	# model = Conv2D(256,3,activation='relu',padding='same')(model)
	# model = Conv2D(256,3,activation='relu',padding='same')(model)
	# model = MaxPooling2D((2,2), strides=(2,2))(model)

	# model = Conv2D(256,3,activation='relu',padding='same')(model)
	# model = Conv2D(256,3,activation='relu',padding='same')(model)
	# model = Conv2D(256,3,activation='relu',padding='same')(model)
	# model = MaxPooling2D((2,2), strides=(2,2))(model)





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
	# drop2 =  Dropout(0.5, name='de_block2_drop')(conv2)
	batch2 = BatchNormalization(name='de_block2_batch')(conv2)


	#Block 3
	up3 = UpSampling2D(name='de_block3_up')(batch2)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', name='de_block3_conv1')(up3)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', name='de_block3_conv2')(conv3)
	conv3 = Conv2D(256, 3, activation='relu', padding='same', name='de_block3_conv3')(conv3)
	conv3 = add([conv3,o3], name='de_block3_add')
	# drop3 = Dropout(0.5, name='de_block3_drop')(conv3)
	batch3 = BatchNormalization(name='de_block3_batch')(conv3)

	#Block 4
	up4 = UpSampling2D(name='de_block4_up')(batch3)
	conv4 = Conv2D(128, 3, activation='relu', padding='same', name='de_block4_conv1')(up4)
	conv4 = Conv2D(128, 3, activation='relu', padding='same', name='de_block4_conv2')(conv4)
	conv4 = add([conv4,o2], name='de_block4_add')
	# drop4 = Dropout(0.5, name='de_block4_drop')(conv4)
	batch4 = BatchNormalization(name='de_block4_batch')(conv4)

	#Block 5
	up5 = UpSampling2D(name='de_block5_up')(batch4)
	conv5 = Conv2D(64, 3, activation='relu', padding='same', name='de_block5_conv1')(up5)
	conv5 = Conv2D(64, 3, activation='relu', padding='same', name='de_block5_conv2')(conv5)
	conv5 = add([conv5,o1], name='de_block5_add')
	# drop5 = Dropout(0.5, name='de_block_drop')(conv5)
	batch5 = BatchNormalization(name='de_block5_batch')(conv5)

	#Final prediction layer
	soft5 = Conv2D(dims, kernel_size=8, strides=8, activation='softmax', padding='same', name='de_block6_softmax')(batch5)

	model = Model(input_tensor,soft5)
	model.summary()

	return model


def get_results(history,filepath):
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

	plt.savefig(filepath + 'train_val_curve.jpg')
	plt.show()



def weighted_categorical_crossentropy(w):
    def loss(y_true, y_pred):
    	y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    	y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    	loss = y_true * K.log(y_pred) * w.T
    	loss = -K.sum(loss, -1)
    	return loss
    return loss



main()
