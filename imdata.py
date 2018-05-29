import os 
import numpy as np 
from PIL import Image
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt 
# from constants import Constants
from utils.constants import Constants
import tensorflow as tf 

const = Constants()

img_data_dir = const.img_data_dir
ascii_data_dir = const.ascii_data_dir
flipped_data_dir = const.ascii_data_dir_flip
val_data_dir = const.val_data_dir
char_array = const.char_array
char_dict = const.char_dict
train_set_size = const.train_set_size


def load_data(
	num_batches=32, 
	batch_size=1, 
	img_rows=224, 
	img_cols=224, 
	txt_rows=28, 
	txt_cols=28, 
	flipped=False,
	validation=False,
	test=False):

	ind = 0
	x = np.zeros((batch_size,img_rows,img_cols,3), dtype='uint8')

	while True:
		if ind == num_batches:
			ind = 0

		if flipped:
			count = batch_size / 2
		else:
			count = batch_size

		for i in range(count):
			if validation:
				imgpath = img_data_dir + 'in_' + str(train_set_size + (ind * count) + i) + '.jpg'
			else:
				imgpath = img_data_dir + 'in_' + str(ind * count + i) + '.jpg'

			img = Image.open(imgpath)
			x[i] = np.asarray(img,dtype='uint8')

			if flipped:
				img = img.transpose(Image.FLIP_LEFT_RIGHT)
				x[i + count] = np.asarray(img,dtype='uint8')

		ind += 1

		if test:
			break

		yield (x)

def get_templates(path='./assets/char_set/', num_chars=16):
	images = np.zeros((1,8,8,num_chars))

	for j in range(num_chars):
		# im = Image.open(path + str(j) + '.png').convert('L')
		# images[0,:,:,j] = np.asarray(im,dtype='uint8')
		im = Image.open(path + str(j) + '.png')
		im = np.asarray(im,dtype='uint8')
		images[0,:,:,j] = np.mean(im,axis=-1)
	# return tf.convert_to_tensor(images,tf.float32)
	return images