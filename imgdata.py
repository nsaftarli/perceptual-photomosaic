import os 
import numpy as np 
from PIL import Image
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 
from constants import Constants

const = Constants()

img_data_dir = const.img_data_dir
# ascii_data_dir = '/home/nsaftarl/Documents/ascii-art/ASCIIArtNN/assets/ascii_out/'
ascii_data_dir = const.ascii_data_dir
val_data_dir = const.val_data_dir

char_array = const.char_array
char_dict = const.char_dict


samples = 202533
text_rows = 28
text_cols = 28



def load_data(num_batches=32, batch_size=32, img_rows=224, img_cols=224, txt_rows=28, txt_cols=28):
	ind = 0
	while True:
		imgs = load_images(batch_size, img_rows, img_cols, start_index=ind)
		labels = load_labels(batch_size, txt_rows, txt_cols, start_index=ind)
		ind += 1
		yield (imgs,labels)
		if ind == num_batches:
			ind = 0


def load_val_data(num_batches=32, batch_size=32, img_rows=224, img_cols= 224, txt_rows=28, txt_cols=28):
	ind = 0
	while True:
		imgs = load_images(batch_size, img_rows, img_cols, start_index=ind, val=True)
		labels = load_labels(batch_size, txt_rows, txt_cols, start_index=ind, val=True)
		ind += 1
		yield (imgs,labels)
		if ind == num_batches:
			ind = 0





def load_images(size, rows, cols, start_index, val=False):
	x = np.zeros((size,rows,cols,3), dtype='uint8')
	for i in range(size):
		if val:
			imgpath = img_data_dir + 'in_' + str((start_index * size + i)+190000) + '.jpg'
		else:
			imgpath = img_data_dir + 'in_' + str(start_index * size + i) + '.jpg'
		img = Image.open(imgpath)
		x[i] = np.asarray(img,dtype='uint8')
		i += 1
	return x


def load_labels(size, rows, cols, start_index, val=False):
	y = np.zeros((size, rows, cols, len(char_array)), dtype='uint8')
	for i in range(size):
		if val:
			labelpath = 'in_' + str((start_index * size + i)+190000) + '.jpg.txt'
		else:
			labelpath = 'in_' + str(start_index * size + i) + '.jpg.txt'
		indices = text_to_ints(labelpath, rows, cols, val)
		y[i] = np.eye(len(char_array))[indices]
	return y


def text_to_ints(text, rows, cols, val):
	if val:
		textfile = open(val_data_dir + text)
	else:
		textfile = open(ascii_data_dir + text)
	result = np.zeros((text_rows,text_cols), dtype='uint8')

	row_index = 0
	col_index = 0
	for i,row in enumerate(textfile):
		for j,col in enumerate(row):
			result[row_index][col_index] = char_dict[col]
			col_index += 1
			if col_index == text_rows:
				col_index = 0 
				row_index += 1
	textfile.close()
	return result

def ints_to_text(arr):
	txtarr = ''
	for row,n in enumerate(arr):
		for col,m in enumerate(n):
			x = 0
			for p in m:
				if p == 1:
					txtarr += char_array[x]
				x += 1
			if col == text_cols -1:
				txtarr += '\n'
	return txtarr


	
def make_histogram():
	num_bins = 255
	grouping = 3

	x = load_images(size=10000, rows=224, cols=224)
	x = np.sum(x, axis=3)
	x = x.flatten()
	
	plt.hist(x, bins=255, range=(0,765))
	plt.yscale('log', nonposy='clip')
	plt.title('Number of pixels with given RGB values')
	plt.xlabel('Sum of RGB values')
	plt.ylabel('Number of pixels')
	plt.figure()

	y = load_labels(size=10000, rows=224, cols=224)
	y_indices = np.argmax(y, axis=3)
	y_indices = y_indices.flatten()

	fig, ax = plt.subplots()


	n, bins, patches = ax.hist(y_indices, bins=16, range=(0,15), align='mid', rwidth=4, ec='black')

	bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
	locs, labels = plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w),('M','N','H','Q', '$', 'O','C', '?','7','>','!',':','-',';','.',' '))

	plt.yscale('log')
	plt.title('Generator Character Choice in Output')
	plt.ylabel('Number of Appearances in Output')
	plt.xlabel('Character')
	ax.set_xticklabels(labels)
	plt.show()

def make_histogram_by_slice(images, labels):
	x = np.sum(images, axis=3)
	y = np.argmax(labes, axis=3)

	x = x.flatten()
	y = y.flatten()

	plt.hist(x, bins=255, range=(0,765))
	plt.yscale('log', nonposy='clip')
	plt.title('Number of pixels with given RGB values')
	plt.xlabel('Sum of RGB values')
	plt.ylabel('Number of pixels')
	plt.figure()


	fig, ax = plt.subplots()


	n, bins, patches = ax.hist(y, bins=16, range=(0,15), align='mid', rwidth=4, ec='black')

	bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
	locs, labels = plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w),('M','N','H','Q', '$', 'O','C', '?','7','>','!',':','-',';','.',' '))

	plt.yscale('log')
	plt.title('Generator Character Choice in Output')
	plt.ylabel('Number of Appearances in Output')
	plt.xlabel('Character')
	ax.set_xticklabels(labels)
	plt.show()


def get_class_weights(size=10000, rows=224, cols=224, targets=None):
	if targets is not None:
		chars = char_counts(labels=targets)
	else:
		chars = char_counts(size)

	char_sum = np.sum(chars)
	char_weight = chars / char_sum

	print(char_weight)

	return char_weight


	
def get_label(label_path,textrows,textcols,dims):
	f = open(label_path,'r')
	buff = ''
	arr = np.zeros((textrows,textcols), dtype='uint8')
	n = 0
	m = 0
	for y,row in enumerate(f):
		for x,col in enumerate(row):
			if x % 224 == 0 and x is not 0:
				n += 1
				m = 0
			arr[n][m] = char_dict[col]
			m += 1
	return arr

def load_chars(path='./assets/char_set/',img_size=28):
	length = len(os.listdir(path))
	char_imgs = np.zeros((length,img_size,img_size,3))
	char_labels = []
	for i,filename in enumerate(os.listdir(path)):
		# print(filename)
		img = Image.open(path + filename)
		char_imgs[i] = np.asarray(img,dtype='uint8')
		char_labels.append(filename[:-4])
	return char_imgs, char_labels

load_data()