import tensorflow as tf 
import numpy as np
from batch_norm_layer import *

vgg_weights = np.load('./weights/vgg16.npy',encoding='latin1').item()


def ConvLayer(x,name,ksize=3,layer_type='VGG16',out_channels=None, trainable=True, patch_size=None, batch_norm=False):

	in_channels = x.get_shape()[3].value

	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		if layer_type == 'VGG16':
			w = get_vgg_weights(name, trainable=trainable)
			b = get_vgg_biases(name, trainable=trainable)
			z = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
			activation = tf.nn.relu(z)

			return activation,w

		elif layer_type == 'Softmax':
			shape_in = [patch_size, patch_size, in_channels, out_channels]
			w = tf.get_variable('weight', initializer=tf.contrib.layers.xavier_initializer(), shape=shape_in, trainable=trainable)
			b = tf.get_variable('bias', initializer=tf.constant(0.0, shape=[out_channels], dtype=tf.float32), trainable=trainable)
			z = tf.nn.conv2d(x, w, strides=[1,patch_size,patch_size,1], padding='SAME') + b

			if batch_norm:
				z = batch_norm_layer(z)

			# zmax = tf.reduce_max(z, axis=-1, keep_dims=True)
			# zmin = tf.reduce_min(z, axis=-1, keep_dims=True)
			# z = ((z-zmin)/(zmax-zmin))+1

			return z,w

			# activation = tf.exp(z) / tf.reduce_sum(tf.exp(z), axis=-1, keep_dims=True)


		else:
			shape_in = [ksize,ksize,in_channels,out_channels]
			w = tf.get_variable('weight', initializer=tf.contrib.layers.xavier_initializer(), shape=shape_in, trainable=trainable)
			b = tf.get_variable('bias',  initializer=tf.constant(0.0,shape=[out_channels],dtype=tf.float32), trainable=trainable)
			z = tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME') + b 

			if batch_norm:
				z = batch_norm_layer(z)

			# zmax = tf.reduce_max(z, axis=-1, keep_dims=True)
			# zmin = tf.reduce_min(z, axis=-1, keep_dims=True)
			# z = ((z-zmin)/(zmax-zmin))+1
			z = tf.tanh(z)

			activation = tf.nn.relu(z)
			return activation,w

		tf.add_to_collection('conv_weights',w)
		tf.add_to_collection('conv_biases',b)
		tf.add_to_collection('pre-act', z)
		tf.add_to_collection('activations',activation)

		return activation

def get_vgg_weights(name, trainable):
	return tf.get_variable(initializer=vgg_weights[name][0], name='filter', trainable=trainable)


def get_vgg_biases(name, trainable):
	return tf.get_variable(initializer=vgg_weights[name][1], name='biases', trainable=trainable)

