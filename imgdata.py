import os 
import numpy as np 
from PIL import Image
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 

#On Ubuntu
img_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/rgb_in/img_celeba/'
ascii_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/ascii_out/'

#On OSX
# img_data_dir = '/Users/Nariman/Documents/GitHub/ASCIIArtNN/assets/rgb_in/'
# ascii_data_dir = '/Users/Nariman/Documents/GitHub/ASCIIArtNN/assets/ascii_out/'


# images = np.zeros((202599,218,178,3), dtype='uint8')
# texts = np.zeros((202599,218,178,15), dtype='uint8')

char_array = np.asarray(['M','N','H','Q', '$', 'O','C', '?','7','>','!',':','-',';','.',' '])
char_dict = {'M':0,'N':1,'H':2,'Q':3,'$':4,'O':5,'C':6,'?':7,'7':8,'>':9,'!':10,':':11,'-':12,';':13,'.':14,' ':15}
'''
Change this on beta. Data of different dimensions
'''
samples = 202533
text_rows = 224
text_cols = 224
dims = 16

# x_train = np.zeros((samples,text_rows,text_cols,3), dtype='uint8')
# y_train = np.zeros((samples,text_rows*text_cols,dims), dtype='uint8')

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
		labelpath = 'in_' + str(i) + '.jpg'
		indices = text_to_ints(labelpath)
		o_h_indices = np.eye(dims)[indices]
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
	print(arr.shape)
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
	x = x.flatten()

	# i = 0
	# j = 0
	# k = 0
	# for pixel in x:
	# 	if pixel == 0:
	# 		i += 1
	# 	if pixel == 255:
	# 		j += 1
	# 	if pixel == 128:
	# 		k += 1

	# print("Values: " + str(i) + ", " + str(j) + ", " + str(k))
	
	plt.hist(x, bins=256, range=(0,255))
	plt.yscale('log', nonposy='clip')
	# plot.show()
	# plt.show()
	plt.figure()

	y = load_labels()
	print(y.shape)
	y_indices = np.argmax(y, axis=3)
	y_indices = y_indices.flatten()



	plt.hist(y_indices, bins=16, range=(0,15))
	plt.show()

	# plt.hist2d(x,y_indices)
	# plt.show()


	# hist, bins = np.histogram(x, bins=255, range=(0,255))
	# width = 0.7 * (bins[1]-bins[0])
	# center = (bins[:-1] + bins[1:]) / 2

	# y = mlab.normpdf(bins)
	# l = plt.plot(bins, y, 'r--', linewidth=1)

	# plt.bar(center, hist, align='center', width=width)
	# plt.show()




# index_arr = text_to_ints('in_0.jpg')
# o_h_index_arr = np.eye(dims)[index_arr]
# full_arr = ints_to_text(o_h_index_arr)
# print(full_arr)
# print(y_train.shape)

make_histogram()

