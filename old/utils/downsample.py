import tensorflow as tf
from gauss2d_kernel import *

def downsample(input):
    w = tf.reshape(tf.constant(gauss2d_kernel(), dtype=tf.float32),
                   [3, 3, 1, 1]) 
    return tf.nn.conv2d(input, w, strides=[1, 2, 2, 1], padding='SAME')