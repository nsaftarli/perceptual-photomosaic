import os
import numpy as np
from PIL import Image
import itertools
from scipy.misc import imread


class Dataset:

    def __init__(self, path):
        self.path = path

    def data_generator(self):
        files = sorted(os.listdir(self.path))
        num_files = len(files)
        for i in itertools.count(1):
            img = imread(self.path + files[i % num_files], mode='RGB').astype('float32')
            yield img, i


def get_templates(path):
    files = sorted(os.listdir(path))
    num_templates = len(files)
    images = []
    for i in range(num_templates):
        im = imread(path + files[i], mode='RGB').astype('float32')
        images.append(im)
    images = np.expand_dims(np.stack(images, axis=-1), axis=0)
    return images


# Move to utils
def turn_im_into_templates(path, patch_size=8):
    im = imread(path)
    x = 0
    for i in range(im.shape[0] // patch_size):
        for j in range(im.shape[1] // patch_size):
            patch = im[i*patch_size:i*(patch_size+1), j*patch_size:j*(patch_size+1), :]
            out = Image.fromarray(patch.astype('uint8'))
            out.save('./assets/cam_templates_2/' + str(x) + '.png', 'PNG')
            x += 1
