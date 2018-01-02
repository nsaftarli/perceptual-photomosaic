import tensorflow as tf 
from keras import models
from keras.preprocessing import image 
from keras.models import Model 
import numpy as np 
# import imgdata as im

img_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/rgb_in/img_celeba/'
ascii_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/ascii_out/'

char_array = np.asarray(['M','N','H','Q', '$', 'O','C', '?','7','>','!',':','-',';','.',' '])
char_dict = {'M':0,'N':1,'H':2,'Q':3,'$':4,'O':5,'C':6,'?':7,'7':8,'>':9,'!':10,':':11,'-':12,';':13,'.':14,' ':15}

# index_arr = text_to_ints('in_0.jpg')
# o_h_index_arr = np.eye(dims)[index_arr]
# full_arr = ints_to_text(o_h_index_arr)
# print(full_arr)


def main(network='ascii_nn.h5', img_name='in_0.jpg'):
	base_model = models.load_model(network)
	base_model.summary()


	img_path = img_data_dir + img_name
	img = image.load_img(img_path, target_size=(224,224))
	img.show()
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	n = base_model.predict(x)

	print(type(n))

	maxes = np.argmax(n,axis=3)
	print(maxes)

	# preds = im.ints_to_text(maxes)

	preds = one_hot(maxes)

	print(preds)

def one_hot(arr):
	x = 0
	y = 0
	out = ''
	for i,row in enumerate(arr):
		for j,col in enumerate(row):
			for k,el in enumerate(col):
				out += char_array[el]
				if k == 223:
					out += '\n'
	return out
			# out += char_array[col]



			
