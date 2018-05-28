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
	y = np.zeros((batch_size,txt_rows,txt_cols,len(char_array)), dtype='uint8')

	while True:
		if ind == num_batches:
			ind = 0

		if flipped:
			count = batch_size / 2
		else:
			count = batch_size

			# print(batch_size)


		for i in range(count):
			if validation:
				imgpath = img_data_dir + 'in_' + str(train_set_size + (ind * count) + i) + '.jpg'
				labelpath = val_data_dir + 'in_' + str(train_set_size + (ind * count) + i) + '.jpg.txt'
			else:
				imgpath = img_data_dir + 'in_' + str(ind * count + i) + '.jpg'
				labelpath = ascii_data_dir + 'in_' + str(ind * count + i) + '.jpg.txt'

			img = Image.open(imgpath)
			x[i] = np.asarray(img,dtype='uint8')


			# txtfile = open(labelpath,'r')
			# labelarr = encode_label(txtfile,txt_rows,txt_cols,test)
			# y[i] = labelarr

			if flipped:
				img = img.transpose(Image.FLIP_LEFT_RIGHT)
				x[i + count] = np.asarray(img,dtype='uint8')

				# labelpath_f = flipped_data_dir + 'in_' + str(ind * count + i) + '.jpg.txt'
				# txtfile = open(labelpath_f,'r')
				# labelarr = encode_label(txtfile,txt_rows,txt_cols,test)
				# y[i + count] = labelarr
		ind += 1

		if test:
			break
		# y = load_templates(batch_size)
		# print(y.shape)
		# print(x.shape)
		# print(y.shape)

		yield (x)
		# yield [[x,y],x]


def encode_label(text, rows, cols,test=False):
	'''
	Given some text, encode it as integers and then one-hot encode it
	'''


	result = np.zeros((rows,cols),dtype='uint8')

	row_index = 0
	col_index = 0
	for i,row in enumerate(text):
		for j,col in enumerate(row):
			result[row_index,col_index] = char_dict[col]
			col_index += 1

			if col_index == rows:
				col_index = 0
				row_index += 1
	text.close()
	result = np.eye(len(char_array))[result]

	if test:
		ints_to_text(result)

	return result



def ints_to_text(arr):
	txtarr = ''
	cols = arr.shape[1]
	for row,n in enumerate(arr):
		for col,m in enumerate(n):
			x = 0
			for p in m:
				if p == 1:
					txtarr += char_array[x]
				x += 1
			if col == cols -1:
				txtarr += '\n'
	print(txtarr)





def load_templates(batch_size):
	path = './assets/char_set/'
	images = np.zeros((batch_size,8,8,16))

	for i in range(batch_size):
		for j in range(16):
			im = Image.open(path + str(j) + '.jpg').convert('L')
			# print(im.size)
			# print(np.asarray(im,dtype='uint8').shape)
			images[i,:,:,j] = np.asarray(im,dtype='uint8')
		# print(np.asarray(im,dtype='uint8').shape)


	return images


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