import os 
import numpy as np 
from scipy import misc
from PIL import Image
# import keras
# import keras.preprocessing.text
# import tensorflow as tf 

#On Ubuntu
#img_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/rgb_input/img_align_celeba_png/'
#ascii_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/ascii_output/'

#On OSX
img_data_dir = '/Users/Nariman/Documents/GitHub/ASCIIArtNN/assets/rgb_in/'
ascii_data_dir = '/Users/Nariman/Documents/GitHub/ASCIIArtNN/assets/ascii_out/'


# images = np.zeros((202599,218,178,3), dtype='uint8')
# texts = np.zeros((202599,218,178,15), dtype='uint8')

char_array = np.asarray(['M','N','H','Q', '$', 'O','C', '?','7','>','!',':','-',';','.',' '])
char_dict = {'M':0,'N':1,'H':2,'Q':3,'$':4,'O':5,'C':6,'?':7,'7':8,'>':9,'!':10,':':11,'-':12,';':13,'.':14,' ':15}
'''
Change this on beta. Data of different dimensions
'''
samples = 202533
text_rows = 219
text_cols = 350
dims = 16

x_train = np.zeros((samples,text_rows,text_cols,3), dtype='uint8')
y_train = np.zeros((samples,text_rows*text_cols,dims), dtype='uint8')



def load_data(image):
	tstimg = img_data_dir + image
	img = Image.open(tstimg)
	print(img.size)
	nparray = np.asarray(img,dtype='uint8')
	print(nparray)
	# tstimg = img_data_dir + 'in_0'
	# face = misc.face()
	# misc.imsave(tstimg,face)
	# face = misc.imread('in_0')
	# type(face)
	# face.shape, face.dtype

def text_to_ints(text):
	textfile = open(ascii_data_dir + text)
	result = np.zeros((text_rows,text_cols), dtype='uint8')
	for row,line in enumerate(textfile):
		if row != text_rows:
			for col,char  in enumerate(line):
				if char != '\n':
					result[row,col] = char_dict[char]
	textfile.close()
	print(result)
	return result

	

index_arr = text_to_ints('in_0')
index_arr = index_arr.flatten()
o_h_index_arr = np.eye(dims)[index_arr]
print(o_h_index_arr.shape)
print(o_h_index_arr)
load_data('in_0.png')





