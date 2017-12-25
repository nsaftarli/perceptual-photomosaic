import os 
import numpy as np 
from scipy import misc
from PIL import Image

img_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/rgb_input/img_align_celeba_png/'
ascii_data_dir = '/home/nsaftarli/Documents/ascii-art/ASCIIArtNN/assets/ascii_output/'


# images = np.zeros((202599,218,178,3), dtype='uint8')
# texts = np.zeros((202599,218,178,15), dtype='uint8')

def load_data(image):
	tstimg = img_data_dir + image
	img = Image.open(tstimg)
	print(img.size)
	nparray = np.asarray(img,dtype='uint8')
	print(nparray.shape)
	# tstimg = img_data_dir + 'in_0'
	# face = misc.face()
	# misc.imsave(tstimg,face)
	# face = misc.imread('in_0')
	# type(face)
	# face.shape, face.dtype
	

