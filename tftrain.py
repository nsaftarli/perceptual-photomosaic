import sys
sys.path.append('layers/')
sys.path.append('utils/')
import os
import tensorflow as tf
import numpy as np 
import sklearn
import imdata
import argparse
# from constants import Constants 

from tfmodel import *
from tfoptimizer import * 
from Loss import *

from utils import *

'''Data constants'''
const = constants.Constants()
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



####Command line arguments#######################
argParser = argparse.ArgumentParser(description='training')
argParser.add_argument('-g','--gpu',dest="gpu",action="store",default=1,type=int)
argParser.add_argument('-i','--iterations',dest='iterations',action='store',default=0,type=int)
cmdArgs = argParser.parse_args()
##################################################

########GPU Settings###########################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cmdArgs.gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
################################################





def print_network():
	return
	print('Loss:',str(sess.run(l.loss)))
	print("Conv6_1",sess.run(m.conv6_1[0,0,0,:]))
	print("Conv6_2",sess.run(m.conv6_2[0,0,0,:]))
	print("Conv6_3",sess.run(m.conv6_3[0,0,0,:]))
	print("Add6",sess.run(m.add6[0,0,0,:]))

	print("Conv7_1",sess.run(m.conv7_1[0,0,0,:]))
	print("Conv7_2",sess.run(m.conv7_2[0,0,0,:]))
	print("Conv7_3",sess.run(m.conv7_3[0,0,0,:]))
	print("Add7",sess.run(m.add7[0,0,0,:]))


	print("Conv8_1",sess.run(m.conv8_1[0,0,0,:]))
	print("Conv8_2",sess.run(m.conv8_2[0,0,0,:]))
	print("Conv8_3",sess.run(m.conv8_3[0,0,0,:]))
	print("Add8",sess.run(m.add8[0,0,0,:]))



	print("Conv9_1",sess.run(m.conv9_1[0,0,0,:]))
	print("Conv9_2",sess.run(m.conv9_2[0,0,0,:]))
	print("Add9",sess.run(m.add9[0,0,0,:]))


	print("Conv10_1",sess.run(m.conv10_1[0,0,0,:]))

	print("Conv10_2",sess.run(m.conv10_2[0,0,0,:]))
	print("Add10:", sess.run(m.add10[0,0,0,:]))
	print("Conv11:", sess.run(m.conv11[0,0,0,:]))

	# print("Batch11:", sess.run(m.batch11[0,0,0,:]))


	print("Softmax:", sess.run(m.softmax[0,0,0,:]))
	print("w:", sess.run(m.w[:,:,:,0]))
	print("Wshape:",sess.run(m.w).shape)
	print("w2:", sess.run(m.w2))




############Data Input######################
dataset = tf.data.Dataset.from_generator(imdata.load_data, (tf.float32))
it = dataset.make_one_shot_iterator()
next_batch = it.get_next()
#########################################



##########Logger########################
writer = tf.summary.FileWriter('./logs/')
########################################




x = sess.run(next_batch)
x = tf.convert_to_tensor(x,tf.float32)
y = imdata.get_templates()



############Hyper-Parameneters###############
t = 1.0
base_lr = 1e-6
#############################################



##############Build Graph###################
with tf.device('/gpu:'+str(cmdArgs.gpu)):
	m = ASCIINet(images=x,templates=y)
	print(m)
	l = Loss(m.gray_im, m.reshaped_output)
	# opt,lr = optimize(l.loss + (5e2 * m.entropy) + (3e1 * m.variance))
	opt, lr = optimize(l.loss)
############################################
tf.summary.scalar('loss',l.loss)
merged = tf.summary.merge_all()

############Training################################

with sess:
	sess.run(tf.global_variables_initializer())
	# print([sess.run(v) for v in tf.global_variables()])
	for i in range(cmdArgs.iterations):
		if i == 0:
			lrate = base_lr
			print_network()

		summary = sess.run([opt,l.loss],feed_dict={lr: lrate, m.temp:t})
		if i % 100 == 0:
			lrate = lr_schedule(base_lr,i)
			print('Iteration #:',str(i))
			print('Learning Rate:',str(lrate))
			print('Loss:',str(summary[1]))
			print('')
			summary = sess.run(m.summaries, feed_dict={lr: lrate, m.temp:t})
			writer.add_summary(summary,global_step=i+1)
			# tf.summary.scalar('loss',summary[1])



			values = sess.run(m.softmax[0, 16, :, :], feed_dict={m.temp:t})
			tf.summary.scalar('loss',l.loss)
			for j in range(28):
				# print(j)
				log_histogram(writer, 'coeff' + str(j), values[j,:],i)
		# print(summary[1])

		if (i+1) % 50 == 0 and t<=15:
			t += 0.05

			# with tf.variable_scope('conv1_1', reuse=True):
			# 	conv1_1_weights = tf.get_variable('filter')
			# with tf.variable_scope('conv6_1', reuse=True):
			# 	conv6_1_weights = tf.get_variable('weight')
			# print(sess.run([tf.reduce_mean(conv1_1_weights), tf.reduce_mean(conv6_1_weights)]))
			# print(sess.run([tf.reduce_mean(tf.gradients(l.loss, [conv1_1_weights])),
			# 	            tf.reduce_mean(tf.gradients(l.loss, [conv6_1_weights]))]))



			# print("decoder mean", sess.run(tf.reduce_mean(m.conv5_3)))
			# print('softmax',sess.run(m.lp[0,0,0,:]))
			# print('rescaled softmax', sess.run(m.x_re[0,0,0,:]))

			# print("grad: ",sess.run(tf.gradients(m.softmax, [m.x_re]))[0][0,0,0,:])
			# print(sess.run(tf.reduce_mean(tf.gradients(m.conv11, [m.flat_softmax]))))

#######################################################

