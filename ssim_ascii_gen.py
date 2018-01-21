import numpy as np 
import matplotlib.pyplot as plt 
from scipy import ndimage as ndi 
from skimage import feature 
from skimage.util import view_as_blocks
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.measure import compare_ssim as ssim
from PIL import Image, ImageFilter, ImageEnhance
import math
# import keras.backend as K

char_set = ['M','N','H','Q', '$', 'O','C', '?','7','>','!',':','-',';','.',' ']
# char_set = ['#','#','#','#','#','#','#','#',' ',' ',' ',' ',' ',' ',' ',' ']
char_set_dir = './assets/char_set/'
out_dir = './assets/ssim_imgs/'
img_dir = './assets/rgb_in/img_celeba/'
c1 = 0.01 ** 2
c2 = 0.03 ** 2

patch_size = 8
img_size = 224
num_of_patches = img_size / patch_size

def main(img_name='in_4.jpg'):
	img = Image.open(img_dir + img_name)
	img = img.resize((img_size,img_size))
	img = img.convert('L')
	imgarray = np.asarray(img, dtype='uint8')
	imgarray = feature.canny(imgarray)
	img = Image.fromarray(imgarray.astype('uint8'))

	char_arr = load_chars()
	char_arr = np.asarray(char_arr, dtype='float32')
	char_arr /= 255
	# print(char_arr[15])
	# print(imgarray.shape)
	# print(char_arr.shape)
	ascii_img = get_ssim(imgarray, char_arr)

	f = open(out_dir + img_name + '.txt', 'w')
	f.write(ascii_img)
	f.close()

def load_chars():
	result = []
	for i in range(len(char_set)):
		char = Image.open(char_set_dir + str(i) + '.jpg')
		char = char.convert('L')
		char = np.asarray(char, dtype='uint8')

		result.append(np.asarray(char, dtype='float32'))
	return result

def get_ssim(imgarray, char_arr):
	# u2 = get_char_mus(char_arr)
	# var2 = get_char_vars(char_arr)
	patches = get_patches(imgarray)	
	ssims = np.zeros((patches.shape[0], char_arr.shape[0]))
	patches = patches.astype('float32')
	for i,patch in enumerate(patches):
		for j,char in enumerate(char_arr):
			ssims[i,j] = ssim(patch,char)
	# print(ssims[0].shape)

	true_ssim = np.argmax(ssims, axis=1)
	# print(true_ssim.shape)

	x = 0
	e = 0
	a = 0
	result = np.zeros((num_of_patches,num_of_patches))
	for i, patch in enumerate(true_ssim):
		row = i % num_of_patches
		col = i // num_of_patches
		result[row,col] = patch

	# print(result)
	result = result.T
	result = result.astype('uint8')
	buff = ''
	for i,row in enumerate(result):
		for j,col in enumerate(row):
			buff += char_set[col]
		# buff += '\n'

	# print(buff)
	return buff




	# for patch in patches:
	# 	lum = get_luminance(patch,mus)
	# 	cont = get_contrast(patch,var)
	# 	struct = get_structure(patch,mus,var,char_arr)
	# 	ssims.append(np.multiply(lum,cont))
	# print(len(ssims))




	
	#Reconstruction to make sure I extracted patches properly

	# x = 0
	# e = 0
	# a = 0

	# for i,patch in enumerate(patches):
	# 	if x == 0 and e == 0:
	# 		result_partial = patch
	# 	elif x < 14:
	# 		result_partial = np.concatenate((result_partial,patch), axis=1)
	# 	else:
	# 		if e == 0:
	# 			result = result_partial
	# 			result_partial = patch 
	# 			e = 1
	# 			x = 0
	# 			a = 1
	# 		else:
	# 			a += 1
	# 			print('APPENDED: ' + str(a))
	# 			result = np.concatenate((result,result_partial), axis=0)
	# 			result_partial = patch
	# 			x = 0

	# 	x += 1

	# result = np.concatenate((result,result_partial), axis=0)

	# buff = ''

	# for i,row in enumerate(result):
	# 	for j,col in enumerate(row):
	# 		buff += str(col)
	# 	buff += '\n'
	# print(buff)




def get_patches(imgarray):
	result = []
	result = np.zeros((num_of_patches * num_of_patches,patch_size,patch_size))
	x = 0
	y = 0
	# a = [[1,2,3],[4,5,6],[7,8,9]]
	# a = np.asarray(a, dtype='uint8')
	# print(a[0])
	# print(imgarray.shape)
	imgarray = np.asarray(imgarray, dtype='uint8')
	for i in range(num_of_patches):
		for j in range(num_of_patches):
			patch = imgarray[patch_size * i : patch_size * (i+1), patch_size * j : patch_size * (j+1)]
			# print(result[x,:,:].shape)
			result[x,:,:] = patch
			x += 1
	# print(x)
	result = np.asarray(result,dtype='uint8')
	# print(result)
	return result
def get_char_mus(arr):
	result = []
	for char in arr:
		result.append(np.mean(char))
	return result

def get_char_vars(arr):
	result = []
	for char in arr:
		result.append(np.var(char))
	return result

def get_luminance(patch,char):
	u1 = np.mean(patch)
	u2 = np.mean(char)
	
	return ((2*u1*u2+c1)/(u1**2 + u2**2 + c1))

def get_contrast(patch,char):
	std1 = np.std(patch)
	std2 = np.std(char)

	return ((2*std1*std2+c2)/(std1**2 + std2**2 + c2))

def get_structure(patch,char):
	covar = np.cov(patch,char)
	std1 = np.std(patch)
	std2 = np.std(char)

	return ((covar + (c2/2))/(std1 * std2 + (c2/2)))

	# for char in char_mus:
	# 	lum = 2 * 

for i in range(30000):
	print(str(i))
	x = 'in_' + str(i) + '.jpg'
	main(img_name=x)