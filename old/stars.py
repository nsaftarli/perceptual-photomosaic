from PIL import Image
import numpy as np 


path = './assets/stars_cropped.jpg'
path_out = './assets/star_temps_2/'
im = np.asarray(Image.open(path), dtype='uint8')
# im = Image.open(path).crop()

x = 0
for i in range(21):
	for j in range(44):
		patch = Image.fromarray(im[i*32:(i+1)*32, j*32:(j+1)*32, :])
		patch.save(path_out + str(x) + '.png')
		x += 1

