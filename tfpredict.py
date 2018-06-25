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
coco_dir = const.coco_dir
img_size = const.img_size
patch_size = const.patch_size
num_patches = const.num_patches



####Command line arguments#######################
argParser = argparse.ArgumentParser(description='training')
argParser.add_argument('-g', '--gpu', dest="gpu", action="store", default=1, type=int)
argParser.add_argument('-i', '--iterations', dest='iterations', action='store', default=10, type=int)
argParser.add_argument('-u', '--update', dest='update', action='store', default=100, type=int)
argParser.add_argument('-lr', '--learning-rate',  dest='lr', action='store', default=1e-6, type=float)
argParser.add_argument('-db', '--debug', dest='debug', action='store', default=False, type=bool)
argParser.add_argument('-t', '--temp', dest='temp',  action='store',  default=1.0,  type=float)
argParser.add_argument('-val', '--val', dest='val',  action='store',  default=True,  type=bool)
argParser.add_argument('-c', '--ckpt', dest='ckpt', action='store', default=None)
argParser.add_argument('-d', '--exp', dest='exp', action='store', default=None, type=str)
argParser.add_argument('-rgb', '--colour', dest='rgb', action='store', default=False, type=str)
argParser.add_argument('-tmp', '--templates', dest='tmp', action='store', default='other', type=str)
argParser.add_argument('-v', '--video', dest='video', action='store', default=False, type=str)
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
exp = cmdArgs.exp
rgb = cmdArgs.rgb
video = cmdArgs.video
tmp = cmdArgs.tmp


#####File Handling###############################
folder = experiments_dir + exp
log_dir = folder + '/log/validation/'
snapshot_dir = folder + '/snapshots/'
im_dir = folder + '/images/validation/'

if not os.path.exists(folder):
    raise ValueError("Experiment doesn't exist")
    # os.makedirs(folder)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(im_dir):
    os.makedirs(im_dir)
# os.makedirs(snapshot_dir)
# os.makedirs(im_dir)





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
elif video:
    dataset = tf.data.Dataset.from_generator(imdata.load_vid_data_gen, (tf.float32, tf.int32))
dataset = tf.data.Dataset.from_generator(imdata.load_vid_data_gen, (tf.float32, tf.int32))

next_batch = dataset.make_one_shot_iterator().get_next()

if tmp == 'ascii':
    y = imdata.get_templates(path='./assets/char_set_alt/', num_temps=NUM_TEMPLATES)
else:
    y = imdata.get_templates(path='./assets/cam_templates/', num_temps=NUM_TEMPLATES)
#############################################

with tf.device('/gpu:'+str(0)):
    input = tf.placeholder(tf.float32, shape=(6, img_size, img_size, 3))
    # m = ASCIINet(images=input, templates=y, batch_size=1)
    m = ASCIINet(images=input, templates=y, batch_size=6, trainable=False, rgb=rgb)
    tLoss = m.tLoss
    # opt,  lr = optimize(tLoss)
    argmax = tf.one_hot(tf.argmax(m.softmax, axis=-1), depth=62)
    o = predictTop(argmax, m.temps, batch_size=6, rgb=True, num_temps=62, img_size=img_size)
    print(m.softmax)
    print(argmax)
    side_by_side = tf.concat([m.input, o], axis=2)
merged = tf.summary.merge_all()

saver = tf.train.Saver()

lrate = base_lr

with sess:
    saver.restore(sess, snapshot_dir + 'checkpoint.chkpt')
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    n = 0
    for i in range(iterations):
        x,ind = sess.run(next_batch)
        # print(x[1])
        feed_dict = {input: x,  m.temp: t}

        summary,  totalLoss = sess.run([merged, tLoss],
                                       feed_dict=feed_dict)

        if i % update == 0:
            print(totalLoss)
            if not video:
                misc.imsave(im_dir + str(n) + '.jpg', sess.run(side_by_side[0], feed_dict={input: x, m.temp: t}))
                n += 1
                misc.imsave(im_dir + str(n) + '.jpg', sess.run(side_by_side[1], feed_dict={input: x, m.temp: t}))
                n += 1
                misc.imsave(im_dir + str(n) + '.jpg', sess.run(side_by_side[2], feed_dict={input: x, m.temp: t}))
                n += 1
                misc.imsave(im_dir + str(n) + '.jpg', sess.run(side_by_side[3], feed_dict={input: x, m.temp: t}))
                n += 1
                misc.imsave(im_dir + str(n) + '.jpg', sess.run(side_by_side[4], feed_dict={input: x, m.temp: t}))
                n += 1
                misc.imsave(im_dir + str(n) + '.jpg', sess.run(side_by_side[5], feed_dict={input: x, m.temp: t}))
                n += 1
            else:
                misc.imsave(im_dir + str(n) + '.jpg', sess.run(o[1], feed_dict={input: x, m.temp: t}))
                n += 1
                misc.imsave(im_dir + str(n) + '.jpg', sess.run(o[2], feed_dict={input: x, m.temp: t}))
                n += 1
                misc.imsave(im_dir + str(n) + '.jpg', sess.run(o[3], feed_dict={input: x, m.temp: t}))
                n += 1
                misc.imsave(im_dir + str(n) + '.jpg', sess.run(o[4], feed_dict={input: x, m.temp: t}))
                n += 1
                misc.imsave(im_dir + str(n) + '.jpg', sess.run(o[5], feed_dict={input: x, m.temp: t}))
                n += 1
                misc.imsave(im_dir + str(n) + '.jpg', sess.run(o[0], feed_dict={input: x, m.temp: t}))
                n += 1
            writer.add_summary(summary, i+1)
    print("Model restored")
