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
from predictTop import *

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
dataset = tf.data.Dataset.from_generator(imdata.load_data,  (tf.float32))
next_batch = dataset.make_one_shot_iterator().get_next()
# next_batch = it.get_next()
#########################################

# x = sess.run(next_batch)
# x = sess.run(next_batch)
# x = sess.run(next_batch)
# x = sess.run(next_batch)
# x = sess.run(next_batch)
# x = tf.convert_to_tensor(x, tf.float32)
y = imdata.get_templates(path='./assets/char_set_alt/',  num_chars=62)
# print(x)


x = imdata.get_pebbles(path='./kosta.jpg')
x = tf.convert_to_tensor(x, tf.float32)


with tf.device('/gpu:'+str(0)):
    m = ASCIINet(images=x, templates=y, batch_size=1)
    tLoss = m.tLoss
    opt,  lr = optimize(tLoss)
    argmax = tf.one_hot(tf.argmax(m.softmax, axis=-1),depth=62)
    o = predictTop(argmax, m.template_tensor)
    print(m.softmax)
    print(argmax)
merged = tf.summary.merge_all()

saver = tf.train.Saver()


lrate = base_lr

with sess:
    saver.restore(sess, "snapshots/a/checkpoint2.ckpt")

    for i in range(1):
        feed_dict = {lr: lrate,  m.temp: t}

        summary,  result,  totalLoss = sess.run([merged,  opt,  tLoss],
                                        feed_dict=feed_dict)
        print(totalLoss)
        misc.imsave("snapshots/a/img3.jpg",sess.run(o[0], feed_dict={m.temp: t}))
    print("Model restored")
