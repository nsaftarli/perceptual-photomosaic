in_path = './assets/mario_temps/'
out_path = './assets/mario_temps_8/'
import os
from PIL import Image


def resize(in_path, out_path, new_size, method):
	in_path = './assets/' + in_path + '/'
	out_path = './assets/' + out_path + '/'
	directory = os.listdir(in_path)
	for i in range(len(directory)):
		print(directory[i])
		if method == 'bicubic':
			im = Image.open(in_path + directory[i]).resize((new_size, new_size), resample=Image.BICUBIC)
		elif method == 'nn':
			im = Image.open(in_path + directory[i]).resize((new_size, new_size), resample=Image.NEAREST)
		im.save(out_path + directory[i])
