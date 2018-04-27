import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np 
import os
from keras import models
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
from constants import Constants
from PIL import Image

from matplotlib import pyplot as plt

import tensorflow as tf 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

const = Constants()

img_data_dir = const.img_data_dir
ascii_data_dir = const.ascii_data_dir
val_data_dir = const.val_data_dir
extra_data_dir = '/home/nsaftarl/Documents/ascii-art/non-resized/img_celeba/'
char_array = const.char_array
char_dict = const.char_dict
def loss(y_true, y_pred):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * K.log(y_pred) 
    loss = -K.sum(loss, -1)
    return loss


def per_class_acc(model_dir,data='validation', imgrows=224, imgcols=224, textrows=28, textcols=28, chars=16):
	base_model=models.load_model(model_dir, custom_objects={'loss':loss})
	if data is 'training':
		directory = ascii_data_dir
		size = const.train_set_size
	elif data is 'validation':
		directory = val_data_dir
		size = const.val_set_size
	elif data is 'overfitting':
		directory = ascii_data_dir
		size = 1

	x_eval = np.zeros((size,imgrows,imgcols,3))
	y_eval = np.zeros((size,textrows,textcols))
	total_characters = np.zeros((chars,), dtype='float32')
	correct_characters = np.zeros((chars,), dtype='float32')
	accs = np.zeros((chars,), dtype='float32')


	for n, el in enumerate(y_eval):
		if directory is ascii_data_dir and data is not 'overfitting':
			img_name = 'in_' + str(n) + '.jpg'
		elif directory is val_data_dir:
			img_name = 'in_' + str(n+190000) + '.jpg'
		elif directory is ascii_data_dir and data is 'overfitting':
			img_name = 'in_4.jpg'

		img_path = img_data_dir + img_name
		label_path = directory + img_name + '.txt'

		img = np.asarray(Image.open(img_path), dtype='float32')
		x_eval[n] = img 

		img_label = get_label(label_path,textrows,textcols,chars)
		y_eval[n] = img_label
		
	y_pred = base_model.predict(x_eval)
	y_pred = np.argmax(y_pred,axis=3)
	flattened_labels = np.asarray(y_eval.flatten(), dtype='uint8')


	for n,el in enumerate(flattened_labels):
		total_characters[el] += 1


	for m,element in enumerate(char_array):
		#Create a mask the same shape as the predicted labels, fill it with one element
		mask = np.full((size,textrows,textcols), fill_value=m)
		#Get array where elements of mask are the same as elements of labels and predictions
		z = np.logical_and(np.equal(mask,y_eval), np.equal(mask,y_pred))
		#Count the number of accurate predictions
		a = np.sum(z == True)
		accs[m] = a 

	accs = (accs / total_characters) * 100 
	print(accs)
	return accs.T

def acc_per_epoch(exp_dir,num_epochs=10):
	accs = np.zeros((len(char_array),num_epochs))

	model_prefix = './experiments/' + exp_dir + '/weights.'
	model_suffix = '.hdf5'

	styles = ['b','r','g','c','m','y','k']



	for i in range(num_epochs):
		m = model_prefix + str(i+1) + model_suffix
		print(m)
		accs[:,i] = per_class_acc(m)

	# plt.plot(1,accs[0,:], 'r', 2, accs[1,:], 'g', 3, accs[2,:], 'b')

	x = np.arange(num_epochs)
	print(x.shape)
	print(accs[0,:].shape)

	fig = plt.figure()
	fig.show()
	ax = fig.add_subplot(111)

	ax.plot(x,accs[0,:], c='r', ls='-', label=char_array[0])
	ax.plot(x,accs[1,:], c='b', ls='-', label=char_array[1])
	ax.plot(x,accs[2,:], c='g', ls='-', label=char_array[2])
	ax.plot(x,accs[3,:], c='c', ls='-', label=char_array[3])
	ax.plot(x,accs[4,:], c='m', ls='-', label=char_array[4])
	ax.plot(x,accs[5,:], c='y', ls='-', label=char_array[5])
	ax.plot(x,accs[6,:], c='k', ls='-', label=char_array[6])
	ax.plot(x,accs[7,:], c='b', ls='-', marker='o', label=char_array[7])
	ax.plot(x,accs[8,:], c='r', ls='-', label=char_array[8])
	ax.plot(x,accs[9,:], c='b', ls='--', label=char_array[9])
	ax.plot(x,accs[10,:], c='g', ls='--', label=char_array[10])
	ax.plot(x,accs[11,:], c='c', ls='--', label=char_array[11])
	ax.plot(x,accs[12,:], c='m', ls='--', label=char_array[12])
	ax.plot(x,accs[13,:], c='y', ls='--', label=char_array[13])
	ax.plot(x,accs[14,:], c='k', ls='--', label=char_array[14])
	ax.plot(x,accs[15,:], c='r', ls='--', marker='o', label=char_array[15])

	plt.legend(loc=2)
	plt.yscale('log')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')


	plt.savefig('./experiments/' + exp_dir + '/val_curve.jpg')





def get_label(label_path,textrows,textcols,dims):
	'''
	Gets the ASCII text file, which corresponds to the ground truth label 

	Inputs:

		label_path: file directory
		textrows: height of file
		textcols: width of file
		dims: number of unique characters 

	Output: 2D array of indices, where each index is a number that represents a character specified by char_dict 

	'''
	# print(label_path)
	f = open(label_path,'r')
	arr = np.zeros((textrows,textcols), dtype='uint8')
	n = 0
	m = 0 
	acc = 0
	for y,row in enumerate(f):
		for x,col in enumerate(row):
			acc+=1
			if x % 28 == 0 and x is not 0:
				n += 1
				m = 0
			arr[n][m] = char_dict[col]
			m += 1
	return arr