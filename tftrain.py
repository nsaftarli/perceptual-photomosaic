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
argParser.add_argument('-g', '--gpu', dest="gpu", action="store", default=1, type=int)
argParser.add_argument('-i', '--iterations', dest='iterations', action='store', default=0, type=int)
argParser.add_argument('-u', '--update', dest='update', action='store', default=100, type=int)
argParser.add_argument('-lr', '--learning-rate',  dest='lr', action='store', default=1e-6, type=float)
argParser.add_argument('-db', '--debug', dest='debug', action='store', default=False, type=bool)
argParser.add_argument('-t', '--temp', dest='temp',  action='store',  default=1.0,  type=float)
cmdArgs = argParser.parse_args()
##################################################


####Settings######################################
gpu = cmdArgs.gpu
iterations = cmdArgs.iterations
update = cmdArgs.update
base_lr = cmdArgs.lr
debug = cmdArgs.debug
t = cmdArgs.temp

########GPU Settings###########################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
################################################


############Data Input######################
dataset = tf.data.Dataset.from_generator(imdata.load_data_gen,  (tf.float32,tf.int32))
next_batch = dataset.make_one_shot_iterator().get_next()
# next_batch = it.get_next()
#########################################






# x,ind = sess.run(next_batch)

# print('BBBBBBBBBBBBb')
# print(x.shape)
# x = tf.convert_to_tensor(x,tf.float32)
# print(x)
# print(ind)
# x,ind = sess.run(next_batch)
# print(ind)
# x,ind = sess.run(next_batch)
# print(ind)
# # x = tf.convert_to_tensor(x, tf.float32)
# sess.run(next_batch)
# print(m)
# y = imdata.get_templates(path='./assets/char_set_alt/', num_chars=62)
y = imdata.get_templates(path='./assets/cam_templates/', num_chars=62)

# x = imdata.load_data_static()

# dataset = tf.data.Dataset.from_tensor_slices(x).batch(6)
# print(dataset)
# iter = dataset.make_one_shot_iterator()
# el = tf.convert_to_tensor(iter.get_next(), tf.float32)
# print(sess.run(el))



############Pebbles Test#####################
# x = imdata.get_pebbles(path='./pebbles.jpg')
# x = tf.convert_to_tensor(x, tf.float32)
#############################################

# input = tf.placeholder(dtype=tf.float32, shape=[])
# with tf.variable_scope('queuer'):
#     q = tf.FIFOQueue(capacity=6, dtypes=tf.float32)
#     enqueue_op = q.enqueue(input)
#     numberOfThreads = 1
#     qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
#     tf.train.add_queue_runner(qr)
#     x = tf.convert_to_tensor(q.dequeue(),tf.float32)





##############Build Graph###################
with tf.device('/gpu:'+str(0)):
    input = tf.placeholder(tf.float32, shape=(6,224,224,3))
    m = ASCIINet(images=input, templates=y)
    tLoss = m.tLoss
    opt,  lr = optimize(tLoss)
merged = tf.summary.merge_all()
############################################

chkpt = tf.train.Saver()
lrate = base_lr

print('AAAAAAAAAAAAAAAAAAAAAA')
print(next_batch)
############Training################################
with sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./logs/', sess.graph)

    
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)


    for i in range(iterations):
        x,ind = sess.run(next_batch)
        # print("Ind: " + str(ind))
        # x = tf.convert_to_tensor(x,tf.float32)

        feed_dict = {input: x,lr: lrate,  m.temp: t}
        summary,  result,  totalLoss = sess.run([merged,  opt,  tLoss],
                                                feed_dict=feed_dict)

        # writer.add_summary(summary,i+1)

        if i % update == 0:
            print('Iteration #:', str(i))
            print('temperature: ' + str(t))
            print('Learning Rate:', str(lrate))
            print('Loss:', str(totalLoss))
            print('Index:', ind)

            if debug:
                print("Input Range:", sess.run(m.gray_im[0, 3:7, 3:7, :]))
                print("Output Range:", sess.run(m.reshaped_output[0, 3:7, 3:7, :],  feed_dict={m.temp: t}))
            writer.add_summary(summary, i+1)

            values = sess.run(m.softmax[0,  17, :, :],  feed_dict={input: x,m.temp: t})
            for j in range(28):
                log_histogram(writer,  'coeff' + str(j),  values[j, :], i)

            # print(sess.run(m.conv12[0, 0, 0, :]))
            # print(sess.run(m.conv12[0, 0, 0, :] * t))
            print('here::::::::::::::::::')
            print(sess.run(m.softmax[0, 0, 0, :],  feed_dict={input: x, m.temp: t}))

            chkpt.save(sess,"snapshots/a/checkpoint2.ckpt")
            misc.imsave("snapshots/a/img.jpg", sess.run(m.view_output[0], feed_dict={input: x, m.temp: t}))

        # print("It: " + str(i))
        # if i % 1000 == 0 and i > 0:
            # t += 14

        if i > 1 and i % 1000 == 0:
            if i < 10000:
                t *= 2
        # if (i+1) % 50 == 0 and t<=1000:
        #   print(t)
        #   if t < 100:
        #       if i < 2000:
        #           t += 0.1
        #       else:
        #           t += 0.3

        # t += 0.1
#######################################################
