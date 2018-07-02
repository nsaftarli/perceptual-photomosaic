in_path = './assets/mario_temps/'
out_path = './assets/mario_temps_8/'
import os
from PIL import Image
directory = os.listdir(in_path)

for i in range(len(directory)):
	print(directory[i])
	im = Image.open(in_path + directory[i]).resize((8,8), resample=Image.BICUBIC)
	im.save(out_path + str(i) + '.png')
