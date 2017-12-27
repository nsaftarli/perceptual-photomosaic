import os 
import numpy as np 
from PIL import Image

#On Ubuntu
img_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/rgb_input/img_align_celeba_png/'
ascii_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/ascii_output/'

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
text_rows = 218
text_cols = 178
dims = 16

# x_train = np.zeros((samples,text_rows,text_cols,3), dtype='uint8')
# y_train = np.zeros((samples,text_rows*text_cols,dims), dtype='uint8')

test_batch_num = 1000

x_train = np.zeros((test_batch_num,text_rows,text_cols,3), dtype='uint8')
y_train = np.zeros((test_batch_num,text_rows * text_cols, dims), dtype='uint8')




def load_images():
	# tstimg = img_data_dir + image
	# img = Image.open(tstimg)
	# print("Image size is: " + str(img.size))
	# nparray = np.asarray(img,dtype='uint8')
	# print("Numpy array of image dims are: " + str(nparray.shape))

	for i in range(test_batch_num):
		imgpath = img_data_dir + 'in_' + str(i) + '.png'
		img = Image.open(imgpath)
		nparray = np.asarray(img,dtype='uint8')
		x_train[i] = nparray

	return x_train
	# print(x_train.shape)
	# print(x_train[80].shape)

def load_labels():
	for i in range(test_batch_num):
		labelpath = 'in_' + str(i) + '.png'
		indices = text_to_ints(labelpath).flatten()
		o_h_indices = np.eye(dims)[indices]
		y_train[i] = o_h_indices
	# print(y_train[0].shape)
	return y_train


def text_to_ints(text):
	textfile = open(ascii_data_dir + text)
	result = np.zeros((text_rows,text_cols), dtype='uint8')
	for row,line in enumerate(textfile):
		if row != text_rows:
			for col,char  in enumerate(line):
				if char != '\n':
					result[row,col] = char_dict[char]
	textfile.close()
	# print(result)
	return result

	

# index_arr = text_to_ints('in_0.png')
# index_arr = index_arr.flatten()
# o_h_index_arr = np.eye(dims)[index_arr]
# print("One hot array of one sample is: " + str(o_h_index_arr.shape))
# print(o_h_index_arr)
# load_data('in_0.png')

def load_data():
	imgs = load_images()
	# print(imgs.shape)
	labels = load_labels()
	# print(labels.shape)
	return (imgs,labels)




