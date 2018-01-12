from sklearn import metrics
import predict as p 
import imgdata as im 
import numpy as np 
import tensorflow as tf 
from keras import models
from PIL import Image

img_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/rgb_in/img_celeba/'
ascii_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/ascii_out/'

label_array = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]


def main(model='ascii_nn4.h5', samples=8000, text_rows=224, text_cols=224, chars=16):
	'''
	Gets confusion matrix for a model, given predictions and ground truth
	'''

	base_model = models.load_model(model)

	x = np.zeros((samples, text_rows, text_cols, 3))
	y_pred = np.zeros((samples,text_rows,text_cols,chars))
	y_eval = np.zeros((samples,text_rows,text_cols))

	for i,el in enumerate(y_pred):
		img_name = 'in_' + str(2000 + i) + '.jpg'
		img_dir = img_data_dir + img_name
		ascii_dir = ascii_data_dir + img_name

		img = np.asarray(Image.open(img_dir))
		x[i] = img 

		y_eval[i] = p.get_label(ascii_dir,text_rows,text_cols,chars)

	y_pred = base_model.predict(x)
	y_pred = np.argmax(y_pred, axis=3)

	print(y_eval.shape)
	print(y_pred.shape)

	c_m = metrics.confusion_matrix(y_eval.flatten(),y_pred.flatten())
	print('\n\n\n')
	buff = ''
	for n,row in enumerate(c_m):
		print(str(row))


main(samples=8000)


