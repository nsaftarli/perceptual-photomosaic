from PIL import Image
import numpy as np 


path = './assets/stars.jpg'
path_out = './assets/star_temps/'
im = np.asarray(Image.open(path), dtype='uint8')

x = 0
for i in range(40):
	for j in range(64):
		patch = Image.fromarray(im[i*32:(i+1)*32, j*32:(j+1)*32, :])
		patch.save(path_out + str(x) + '.png')
		x += 1

