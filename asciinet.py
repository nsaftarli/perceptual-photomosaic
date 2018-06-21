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
lrate = 1e-4
batchsize = 8
batches = 23750
valbatches = 393
flip=False




def lr_sched(index):
	if index < 10:
		return 1e-4
	elif index >= 10 and index <  20:
		return 1e-5
	elif index > 20 and index < 30:
		return 1e-6
	else:
		return  float(0.00000125)


def main():
	dt = datetime.datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
	path = experiments_dir + dt + '/'

	m = model.build_model()
	m.compile(
		loss='categorical_crossentropy',
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
		'flip':flip})

	checkpath = path + 'ascii_nn_checkpt.h5'
	char_acc_callback = callbacks.ClassAccs(logs={'model_dir':checkpath})
	history = m.fit_generator(
		imdata.load_data(num_batches=batches,batch_size=batchsize,flipped=flip,validation=False),
		steps_per_epoch=batches,
		epochs=30
		# validation_data=imdata.load_data(num_batches=valbatches,batch_size=batchsize,flipped=False,validation=True),
		# validation_steps=valbatches
		# callbacks=[checkpoint,logs_callback]


		)

	m.save(path + 'ascii_nn.h5')
	get_results(history,path)





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



main()
