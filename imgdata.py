import os 
import numpy as np 
from PIL import Image
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 

#On Ubuntu
img_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/rgb_in/img_celeba/'
# ascii_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/ascii_out/'
ascii_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/testfile/'

#On OSX
# img_data_dir = '/Users/Nariman/Documents/GitHub/ASCIIArtNN/assets/rgb_in/'
# ascii_data_dir = '/Users/Nariman/Documents/GitHub/ASCIIArtNN/assets/ascii_out/'


# char_array = np.asarray(['M','N','H','Q', '$', 'O','C', '?','7','>','!',':','-',';','.',' '])
# char_dict = {'M':0,'N':1,'H':2,'Q':3,'$':4,'O':5,'C':6,'?':7,'7':8,'>':9,'!':10,':':11,'-':12,';':13,'.':14,' ':15}
char_array = np.asarray(['#', ' '])
char_dict = {'#':0, ' ':1}
'''
Change this on beta. Data of different dimensions
'''
samples = 202533
text_rows = 224
text_cols = 224
dims = 2


test_batch_num = 10000

x_train = np.zeros((test_batch_num,text_rows,text_cols,3), dtype='uint8')
y_train = np.zeros((test_batch_num,text_rows, text_cols, dims), dtype='uint8')




def load_images():
	for i in range(test_batch_num):
		imgpath = img_data_dir + 'in_' + str(i) + '.jpg'
		img = Image.open(imgpath)
		nparray = np.asarray(img,dtype='uint8')
		x_train[i] = nparray
		i += 1
	return x_train


def load_labels():
	for i in range(test_batch_num):
		labelpath = 'in_' + str(i) + '.txt'
		indices = text_to_ints(labelpath)
		# print(indices)
		o_h_indices = np.eye(2)[indices]
		y_train[i] = o_h_indices
	return y_train


def text_to_ints(text):
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

def load_data():
	imgs = load_images()
	labels = load_labels()
	return (imgs,labels)

	
def make_histogram():
	num_bins = 255
	grouping = 3

	x = load_images()
	x = np.sum(x, axis=3)
	x = x.flatten()
	
	plt.hist(x, bins=255, range=(0,765))
	plt.yscale('log', nonposy='clip')
	plt.title('Number of pixels with given RGB values')
	plt.xlabel('Sum of RGB values')
	plt.ylabel('Number of pixels')
	plt.figure()

	y = load_labels()
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

	
