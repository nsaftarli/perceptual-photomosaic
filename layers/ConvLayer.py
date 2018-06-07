import tensorflow as tf 
import numpy as np
from Normalization import *
from Regularizers import *


def ConvLayer(x,name,ksize=3, stride=1,layer_type='Decoder',out_channels=None, trainable=True, patch_size=None, norm_type=None):

	in_channels = x.get_shape()[3].value
	shape_in = [ksize, ksize, in_channels, out_channels]

	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		w = tf.get_variable('weight', initializer=tf.contrib.layers.xavier_initializer(), shape=shape_in, trainable=trainable)
		b = tf.get_variable('bias', initializer=tf.constant(0.0, shape=[out_channels], dtype=tf.float32), trainable=trainable)
		z = tf.nn.conv2d(x, w, strides=[1,stride, stride,1], padding='SAME') + b

		######Normalization########
		if norm_type=='batch':
			z = batch_norm_layer(z)
		elif norm_type=='instance':
			z = InstanceNorm(z)
		elif norm_type=='group':
			z = GroupNorm(z,G=2)
		elif norm_type=='layer':
			z = LayerNorm(z)
		############################


		if layer_type == 'Softmax':
			activation = z

		else:
			activation = tf.nn.leaky_relu(z)
			tf.add_to_collection('activations',activation)

		tf.add_to_collection('conv_weights',w)
		tf.add_to_collection('conv_biases',b)
		tf.add_to_collection('pre-act', z)

		return activation,w

