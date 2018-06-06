import numpy as np 
from PIL import Image

def create_char_img(path='../assets/char_set/'):
	tiles = np.zeros((224,224,3)).astype('uint8')
	# im = np.asarray(Image.open(path),dtype='uint8')
	im2 = np.asarray(Image.open('../assets/char_set/14.png'),dtype='uint8')
	im3 = np.asarray(Image.open('../assets/char_set/15.png'),dtype='uint8')
	# print(im.shape)
	# print(type(im))
	for i in range(0,28):
		for j in range(0,28):
			im1_ind = np.random.randint(0,15)
			im2_ind = np.random.randint(0,15)
			im1 = np.asarray(Image.open(path + str(im1_ind) + '.png'))
			im2 = np.asarray(Image.open(path + str(im2_ind) + '.png'))


			y = np.random.randint(1,6)
			if j % y == 0:
				tiles[i*8:i*8+8, j*8:j*8+8,0:3] = im1
			elif j % 5 == 0:
				tiles[i*8:i*8+8, j*8:j*8+8,0:3] = im2
			else:
				tiles[i*8:i*8+8, j*8:j*8+8, 0:3] = im3

	new_img = Image.fromarray(tiles)
	new_img.show()
	print(new_img.size)
	new_img.save('test.jpg')
if __name__ == '__main__':
	create_char_img()
