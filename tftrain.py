import sys
sys.path.append('layers/')
sys.path.append('utils/')
import os
import time
import argparse
import tensorflow as tf
import numpy as np 
import imdata


from tfmodel import *
from tfoptimizer import * 
from layers import LossLayer

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
argParser.add_argument('-u','--update',dest='update',action='store',default=100,type=int)
argParser.add_argument('-lr','--learning-rate', dest='lr',action='store',default=1e-6,type=float)
argParser.add_argument('-db','--debug',dest='debug',action='store',default=False,type=bool)
cmdArgs = argParser.parse_args()
##################################################


####Settings######################################
gpu = cmdArgs.gpu
iterations = cmdArgs.iterations
update = cmdArgs.update
base_lr = cmdArgs.lr
debug = cmdArgs.debug

########GPU Settings###########################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
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
	print("w2shape",sess.run(m.w2).shape)

	# print("Input Range:",sess.run(m.gray_im[0,1:6,1:6,:]))
	# print("Output Range:", sess.run(m.reshaped_output[0,1:6,1:6,:]))




############Data Input######################
dataset = tf.data.Dataset.from_generator(imdata.load_data, (tf.float32))
it = dataset.make_one_shot_iterator()
next_batch = it.get_next()
#########################################



##########Logger########################
# writer = tf.summary.FileWriter('./logs/',sess.graph)
########################################




x = sess.run(next_batch)
print(x.shape)
x = tf.convert_to_tensor(x,tf.float32)

y = imdata.get_templates()



############Pebbles Test#####################
x = imdata.get_pebbles(path='./pebbles.jpg')
x = tf.convert_to_tensor(x,tf.float32)
#############################################


############Hyper-Parameneters###############
t = 1.0
#############################################



##############Build Graph###################
with tf.device('/gpu:'+str(0)):
	m = ASCIINet(images=x,templates=y)
	# l = LossLayer(m.gray_im, m.reshaped_output).mse 
	l = m.loss

	e = m.entropy
	v = m.variance
	tLoss = l+e+v
	# l1 = LossLayer(m.feature_dict['conv1_2_1'], m.feature_dict['conv1_2_2'])
	# l2 = LossLayer(m.feature_dict['conv2_2_1'], m.feature_dict['conv2_2_2'])
	# l3 = LossLayer(m.feature_dict['conv3_3_1'], m.feature_dict['conv3_3_2'])
	# l4 = LossLayer(m.feature_dict['conv4_3_1'], m.feature_dict['conv4_3_2'])
	# l5 = LossLayer(m.feature_dict['conv5_3_1'], m.feature_dict['conv5_3_2'])
	# opt, lr = optimize(1 * l.loss + (1e2*l1.loss) + (1e0*l2.loss) + (1e0*l3.loss) + (1e3*l4.loss) + (1e3*l5.loss))
	opt, lr = optimize(tLoss)
merged = tf.summary.merge_all()
############################################

############Training################################

with sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./logs/',sess.graph)

	for i in range(iterations):
		startTime = time.time()

		if i == 0:
			lrate = base_lr
			print_network()

		lrate = lr_schedule(base_lr,i)
		feed_dict = {lr: lrate, m.temp:t}
		summary, result, totalLoss = sess.run([merged, opt, tLoss], feed_dict=feed_dict)

		if i % update == 0:
			now = time.time() - startTime
			itPerSec = update / now

			# lrate = lr_schedule(base_lr,i)
			print('Iteration #:',str(i))
			print('Iterations per second:',str(itPerSec))
			print('Learning Rate:',str(lrate))
			print('Loss:',str(totalLoss))
			print('Temperature:',str(t))
			print('##################################')

			if debug:
				print("Input Range:",sess.run(m.gray_im[0,3:7,3:7,:]))
				print("Output Range:", sess.run(m.reshaped_output[0,3:7,3:7,:], feed_dict={m.temp:t}))
			writer.add_summary(summary,i+1)


			values = sess.run(m.softmax[0, 16, :, :], feed_dict={m.temp:t})
			for j in range(28):
				log_histogram(writer, 'coeff' + str(j), values[j,:],i)

		# if (i+1) % 100 == 0 and t<=5:
			# t += 0.05
		# t += 0.01
#######################################################

