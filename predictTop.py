import sys
sys.path.append('layers/')
sys.path.append('utils/')
import os
import time
import argparse
import tensorflow as tf
import numpy as np
import imdata
import math
from tfmodel import *
from tfoptimizer import *
from layers import LossLayer
from utils import *
import scipy.misc as misc


def predictTop(argmax, templates, patch_size=8, batch_size=1, rgb=False, num_temps=62, img_size=224):
    reshaped_argmax = tf.reshape(argmax,[-1, (img_size//8) ** 2, num_temps])
    ##########################################################################################################

    if not rgb:
        with tf.name_scope('output_and_tile'):
            output = tf.matmul(reshaped_argmax, templates)
            output = tf.reshape(tf.transpose(tf.reshape(
                output, [batch_size, (img_size//patch_size), (img_size//patch_size), patch_size, patch_size]),
                perm=[0, 1, 3, 2, 4]), [batch_size, img_size, img_size, 1])
            view_output = tf.tile(output, [1, 1, 1, 3])
            return view_output
    else:
        with tf.name_scope('output_and_tile'):
            print(templates.get_shape())
            print(reshaped_argmax.get_shape())
            print('/////////////////////////////////////////////////////////////////////////////')
            r, g, b = tf.split(templates, 3, axis=3)
            r = tf.transpose(tf.reshape(tf.squeeze(r, axis=3), [-1, patch_size ** 2, num_temps]), [0, 2, 1])
            r = tf.tile(r, [batch_size, 1, 1])
            g = tf.transpose(tf.reshape(tf.squeeze(g, axis=3), [-1, patch_size ** 2, num_temps]), [0, 2, 1])
            g = tf.tile(g, [batch_size, 1, 1])
            b = tf.transpose(tf.reshape(tf.squeeze(b, axis=3), [-1, patch_size ** 2, num_temps]), [0, 2, 1])
            b = tf.tile(b, [batch_size, 1, 1])

            o_r = tf.matmul(reshaped_argmax, r)
            o_r = tf.reshape(tf.transpose(tf.reshape(
                o_r, [batch_size, (img_size//patch_size), (img_size//patch_size), patch_size, patch_size]),
                perm=[0, 1, 3, 2, 4]), [batch_size, img_size, img_size, 1])

            o_g = tf.matmul(reshaped_argmax, g)
            o_g = tf.reshape(tf.transpose(tf.reshape(
                o_g, [batch_size, (img_size//patch_size), (img_size//patch_size), patch_size, patch_size]),
                perm=[0, 1, 3, 2, 4]), [batch_size, img_size, img_size, 1])

            o_b = tf.matmul(reshaped_argmax, b)
            o_b = tf.reshape(tf.transpose(tf.reshape(
                o_b, [batch_size, (img_size//patch_size), (img_size//patch_size), patch_size, patch_size]),
                perm=[0, 1, 3, 2, 4]), [batch_size, img_size, img_size, 1])

            return tf.concat([o_r, o_g, o_b], axis=3)
