import os 
import numpy as np 
from PIL import Image
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 

img_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/rgb_in/img_celeba/'
ascii_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/ascii_out/'
# ascii_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/testfile/'


char_array = np.asarray(['M','N','H','Q', '$', 'O','C', '?','7','>','!',':','-',';','.',' '])
char_dict = {'M':0,'N':1,'H':2,'Q':3,'$':4,'O':5,'C':6,'?':7,'7':8,'>':9,'!':10,':':11,'-':12,';':13,'.':14,' ':15}
# char_array = np.asarray(['#', ' '])
# char_dict = {'#':0, ' ':1}


samples = 202533
text_rows = 224
text_cols = 224



def load_data(batch_size=10000, rows=224, cols=224):
	imgs = load_images(batch_size, rows, cols)
	labels = load_labels(batch_size, rows, cols)
	return (imgs,labels)



def load_images(size, rows, cols):
	x = np.zeros((size,rows,cols,3), dtype='uint8')
	for i in range(size):
		imgpath = img_data_dir + 'in_' + str(i) + '.jpg'
		img = Image.open(imgpath)
		x[i] = np.asarray(img,dtype='uint8')
		i += 1
	return x


def load_labels(size, rows, cols):
	y = np.zeros((size, rows, cols, len(char_array)), dtype='uint8')
	for i in range(size):
		labelpath = 'in_' + str(i) + '.jpg'
		indices = text_to_ints(labelpath, rows, cols)
		y[i] = np.eye(len(char_array))[indices]
	return y


def text_to_ints(text, rows, cols):
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


	# count_dict = {'M':0,'N':0,'H':0,'Q':0,'$':0,'O':0,'C':0,'?':0,'7':0,'>':0,'!':0,':':0,'-':0,';':0,'.':0,' ':0}


def char_counts(size=10000, textrows=224, textcols=224, dims=16, labels=None):
	# import imgdata as im 
	if labels is None:
		y_labels = load_labels(size,textrows,textcols)
	else:
		y_labels = labels 

	y_labels = np.argmax(y_labels,axis=3)
	total_characters = np.zeros((dims,))

	for n, el in enumerate(y_labels):
		label_name = 'in_' + str(2000 + n) + '.jpg'
		label_path = ascii_data_dir + label_name

		y_labels[n] = get_label(label_path, textrows, textcols, dims)

	flattened_labels = np.asarray(y_labels.flatten(), dtype='uint8')

	for n, el in enumerate(flattened_labels):
		total_characters[el] += 1

	print("#######################################################################")
	print("TOTAL NUMBER OF CHARACTERS: " + str(np.sum(total_characters)))
	print("#######################################################################")
	print("NUMBER OF TIMES EACH CHARACTER HAS APPEARED")
	print("M: " + str(total_characters[0]))
	print("N: " + str(total_characters[1]))
	print("H: " + str(total_characters[2]))
	print("Q: " + str(total_characters[3]))
	print("$: " + str(total_characters[4]))
	print("O: " + str(total_characters[5]))
	print("C: " + str(total_characters[6]))
	print("?: " + str(total_characters[7]))
	print("7: " + str(total_characters[8]))
	print(">: " + str(total_characters[9]))
	print("!: " + str(total_characters[10]))
	print(":: " + str(total_characters[11]))
	print("-: " + str(total_characters[12]))
	print(";: " + str(total_characters[13]))
	print(".: " + str(total_characters[14]))
	print("SPACE: " + str(total_characters[15]))	

	return total_characters

	
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