from PIL import Image
import numpy as np

path = './assets/mario_sprites.png'
path_out = './assets/mario_templates/'

sprite_map = np.asarray(Image.open(path), dtype='uint8')
# sprite_map.show()
x = 0
for i in range(8):
	for j in range(50):
		im = Image.fromarray(sprite_map[i*16:(i+1)*16, j*16:(j+1)*16, :])
		im.save(path_out + str(x) + '.png')
		x += 1
# im = Image.fromarray(sprite_map[16:32, 0:16, :])
# im.show()
