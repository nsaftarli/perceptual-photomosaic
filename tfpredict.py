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
argParser.add_argument('-v', '--val', dest='val',  action='store',  default=True,  type=bool)
argParser.add_argument('-c', '--ckpt', dest='ckpt', action='store', default=None)
cmdArgs = argParser.parse_args()
##################################################

####Settings######################################
gpu = cmdArgs.gpu
iterations = cmdArgs.iterations
update = cmdArgs.update
base_lr = cmdArgs.lr
debug = cmdArgs.debug
t = cmdArgs.temp
val = cmdArgs.val

########GPU Settings###########################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
################################################

############Data Input######################

if val:
    dataset = tf.data.Dataset.from_generator(imdata.load_val_data_gen,  (tf.float32))
    next_batch = dataset.make_one_shot_iterator().get_next()

y = imdata.get_templates(path='./assets/cam_templates/',  num_chars=62)

x = sess.run(next_batch)
with tf.device('/gpu:'+str(0)):
    input = tf.placeholder(tf.float32, shape=(6, 224, 224, 3))
    # m = ASCIINet(images=input, templates=y, batch_size=1)
    m = ASCIINet(images=input, templates=y, batch_size=6, trainable=False)
    tLoss = m.tLoss
    # opt,  lr = optimize(tLoss)
    argmax = tf.one_hot(tf.argmax(m.softmax, axis=-1),depth=62)
    o = predictTop(argmax, m.template_tensor, batch_size=6)
    print(m.softmax)
    print(argmax)
merged = tf.summary.merge_all()

saver = tf.train.Saver()

lrate = base_lr

with sess:
    saver.restore(sess, "snapshots/a/checkpoint2.ckpt")
    writer = tf.summary.FileWriter('./logs/validation', sess.graph)


    for i in range(iterations):
        x = sess.run(next_batch)
        # print(x[1])
        feed_dict = {input: x,  m.temp: t}

        summary,  totalLoss = sess.run([merged, tLoss],
                                       feed_dict=feed_dict)

        if i % update == 0: 
            print(totalLoss)
            misc.imsave("snapshots/a/img3.jpg", sess.run(o[0], feed_dict={input: x, m.temp: t}))
            writer.add_summary(summary, i+1)
    print("Model restored")
