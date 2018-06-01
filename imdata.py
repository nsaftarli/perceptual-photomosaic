import os 
import numpy as np 
from PIL import Image
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt 
# from constants import Constants
from utils.constants import Constants
import tensorflow as tf 


###############################################
from skimage import feature
from scipy import ndimage as ndi 
##############################################

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

def get_pebbles(path='./pebbles.jpg'):
	edges = np.zeros((224,224,3), dtype='uint8')
	img = Image.open(path).resize((224,224))
	img_arr = np.asarray(img)
	# r = img_arr[:,:,0]
	# g = img_arr[:,:,1]
	# b = img_arr[:,:,2]

	# r_edges = feature.canny(r,sigma=3)
	# g_edges = feature.canny(g,sigma=3)
	# b_edges = feature.canny(b,sigma=3)

	# edges[...,0] = r_edges * 255
	# edges[...,1] = g_edges * 255
	# edges[...,2] = b_edges * 255

	# # edges = np.concatenate([r_edges,g_edges,b_edges],axis=-1)
	# print(edges.dtype)
	# print(np.asarray(img).reshape((-1,224,224,3)).dtype)
	# print('AAAAAAAAAAAAAAa')
	# rescaled = np.asarray(Image.fromarray(edges))
	return img_arr.reshape((-1,224,224,3))
	# return np.asarray(Image.fromarray(edges)).reshape((-1,224,224,3))
	# return np.asarray(img).reshape((-1,224,224,3))