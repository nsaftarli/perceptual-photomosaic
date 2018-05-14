import os
import tensorflow as tf
import numpy as np 



def init_weights(shape):
	init_random_dist = tf.random_normal(shape=shape)
	return tf.Variable(init_random_dist)

def init_bias(shape,init_val=0.0):
	b = tf.constant(init_val,shape=shape)

# def Conv2D(x,W,stride=1,pad='SAME'):
# 	return tf.nn.conv2d(x,W,strides=[1,stride,stride,1] ,padding=pad)

# def MaxPool(x,k_shape=[1,2,2,1], strides=1, pad='SAME', name='MaxPool'):
# 	return tf.nn.max_pool(x, ksize=kern_size, strides=[1,stride,stride,1], padding=pad)

def Conv2D(x,k_h,k_w,k_out,stride=1,padding='SAME',name='Conv2D'):
	#x is [samples,h,w,c]
	# print(name)
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		# print(str(k_h) + "," + str(k_w))
		# print(k_out)
		dim_in = x.get_shape()[3].value
		# print(dim_in)

		kshape = [k_h, k_w, dim_in, k_out]
		# print(kshape)
		W = tf.get_variable('weights', shape=kshape, initializer=tf.uniform_unit_scaling_initializer())
		b = tf.get_variable('bias', shape=dim_in, initializer=tf.uniform_unit_scaling_initializer(0))
		z = tf.nn.conv2d(x,W,strides=[1,stride,stride,1] ,padding=padding)

		tf.add_to_collection('WEIGHTS',W)

		return tf.nn.relu(z,name=name)

def MaxPool(x,k_h=2,k_w=2,stride=1,padding='SAME',name='MaxPool'):
	with tf.variable_scope('pool', reuse=tf.AUTO_REUSE):
		dim_in = x.get_shape()[3]
		k_shape = [1, k_h, k_w, dim_in]

		return tf.nn.max_pool(x, ksize=k_shape, strides=[1,stride,stride,1], padding=padding, name=name)



