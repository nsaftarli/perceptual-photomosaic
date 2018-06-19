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


def predictTop(argmax, templates):
    reshaped_argmax = tf.reshape(argmax,[-1, (224//8) ** 2, 62])
    ##########################################################################################################
    with tf.name_scope('output_and_tile'):
        output = tf.matmul(reshaped_argmax, templates)
        output = tf.reshape(tf.transpose(tf.reshape(
            output, [1, 28, 28, 8, 8]),
            perm=[0, 1, 3, 2, 4]), [1, 224, 224, 1])
        view_output = tf.tile(output, [1, 1, 1, 3])
        return view_output
