import tensorflow as tf
import numpy as np 
import sklearn
import imdata
import argparse
from constants import Constants 

# from tfops import Model
from tfmodel import *
from tfoptimizer import * 
from Loss import *

'''Data constants'''
const = Constants()
img_data_dir = const.img_data_dir
ascii_data_dir = const.ascii_data_dir
val_data_dir = const.val_data_dir
char_array = const.char_array
char_dict = const.char_dict
img_rows = const.img_rows
img_cols = const.img_cols
text_rows = const.text_rows
text_cols = const.text_cols
dims = const.char_count
experiments_dir = const.experiments_dir

dataset = tf.data.Dataset.from_generator(imdata.load_data, (tf.float32))
it = dataset.make_one_shot_iterator()
next_batch = it.get_next()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

sess = tf.Session(config=config)
writer = tf.summary.FileWriter('./logs/')


x = sess.run(next_batch)
x = tf.convert_to_tensor(x,tf.float32)
y = imdata.get_templates()


with tf.device('/gpu:0'):
	m = ASCIINet(images=x,templates=y)
	print(m)
	l = Loss(m.gray_im, m.output)
	opt,lr = optimize(l.loss)


with sess:
	init = sess.run(tf.global_variables_initializer())
	for i in range(1000):
		summary = sess.run([opt,l.loss],feed_dict={lr: 0.001})

		if i % 10 == 0:
			print(i)
			print(summary[1])
			summary = sess.run(m.summaries)
			writer.add_summary(summary,global_step=i+1)
			with tf.variable_scope('conv1_1', reuse=True):
				conv1_1_weights = tf.get_variable('filter')
			with tf.variable_scope('conv6_1', reuse=True):
				conv6_1_weights = tf.get_variable('weight')
			print(sess.run([tf.reduce_mean(conv1_1_weights), tf.reduce_mean(conv6_1_weights)]))
			print(sess.run([tf.reduce_mean(tf.gradients(l.loss, [conv1_1_weights])),
				            tf.reduce_mean(tf.gradients(l.loss, [conv6_1_weights]))]))
			print("grad: ",sess.run(tf.reduce_mean(tf.gradients(m.softmax, [m.add10]))))
			print(sess.run(tf.reduce_mean(tf.gradients(m.conv11, [m.flat_softmax]))))




		


