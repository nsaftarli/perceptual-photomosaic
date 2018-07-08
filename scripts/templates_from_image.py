from PIL import Image
from scipy.misc import imread

# Slices image into non-overlapping templates and writes them out
def templates_from_image(path_in, path_out, patch_size=8):
    im = imread(path_in)
    x = 0
    for i in range(im.shape[0] // patch_size):
        for j in range(im.shape[1] // patch_size):
            patch = im[i*patch_size:i*(patch_size+1), j*patch_size:j*(patch_size+1), :]
            out = Image.fromarray(patch.astype('uint8'))
            out.save(path_out + str(x) + '.png', 'PNG')
            x += 1
