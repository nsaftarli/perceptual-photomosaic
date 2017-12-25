import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras import layers
from keras import models
from keras.applications import VGG16



char_array = ['M','N','H','Q','0','C','7','>','!',':','-',';','.',' ']


conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(300,300,3))
conv_base.summary()

conv_head = layers.Conv2D(1,(3,3),activation='relu')

model = models.Sequential()
model.add(conv_base)
model.add()