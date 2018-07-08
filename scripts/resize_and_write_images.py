import os
from scipy.misc import *


def resize_and_write_images(in_path, out_path, size=(512, 512)):
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  for im in os.listdir(in_path):
    img = imresize(imread(in_path + '/' + im, mode='RGB'), size=size, interp='bicubic')
    imsave(out_path + '/' + im, img)
