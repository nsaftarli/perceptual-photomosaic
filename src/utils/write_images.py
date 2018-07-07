import os
import scipy.misc as misc


def write_images(images, path, inds):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(images.shape[0]):
        misc.imsave(path + '/' + str(inds[i]).zfill(8) + '.png', images[i])
