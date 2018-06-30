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
img_orig_size = const.img_orig_size
patch_size = const.patch_size
num_patches = const.num_patches
img_new_size = const.img_new_size






####Command line arguments#######################
argParser = argparse.ArgumentParser(description='training')
argParser.add_argument('-g', '--gpu', dest="gpu", action="store", default=1, type=int)
argParser.add_argument('-i', '--iterations', dest='iterations', action='store', default=0, type=int)
argParser.add_argument('-u', '--update', dest='update', action='store', default=100, type=int)
argParser.add_argument('-lr', '--learning-rate',  dest='lr', action='store', default=1e-6, type=float)
argParser.add_argument('-t', '--temperature', dest='temp',  action='store',  default=1.0,  type=float)
argParser.add_argument('-n', '--notes', dest='notes', action='store', default=None, type=str)
argParser.add_argument('-d', '--dataset', dest='dset', action='store', default='Faces', type=str)
argParser.add_argument('-rgb', '--colour', dest='rgb', action='store', default=False, type=bool)
argParser.add_argument('-tmp', '--templates', dest='tmp', action='store', default='other', type=str)
argParser.add_argument('-v', '--video', dest='vid', action='store', default=False, type=bool)
cmdArgs = argParser.parse_args()
##################################################

####Settings######################################
gpu = cmdArgs.gpu
iterations = cmdArgs.iterations
update = cmdArgs.update
base_lr = cmdArgs.lr
t = cmdArgs.temp
notes = cmdArgs.notes
dset = cmdArgs.dset
rgb = cmdArgs.rgb
tmp = cmdArgs.tmp
vid = cmdArgs.vid
NUM_TEMPLATES = 62


#####File Handling###############################
now = time.strftime('%d%b-%X')
folder = experiments_dir + now
log_dir = folder + '/log/training/'
snapshot_dir = folder + '/snapshots/'
im_dir = folder + '/images/'

if not os.path.exists(folder):
    os.makedirs(folder)
    os.makedirs(log_dir)
    os.makedirs(snapshot_dir)
    os.makedirs(im_dir)

with open(folder + '/info.txt', 'w') as fp:
    fp.write('Hyperparameters:' + '\n')
    fp.write('# Iterations: ' + str(iterations) + '\n')
    fp.write('Snapshot Freq: ' + str(update) + '\n')
    fp.write('Learning Rate: ' + str(base_lr) + '\n')
    fp.write('Initial Temperature:' + str(t) + '\n')
    fp.write('Notes: ' + '\n')
    if notes is not None:
        fp.write(notes + '\n')


########GPU Settings###########################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
################################################


############Data Input######################
if not vid:
    dataset = tf.data.Dataset.from_generator(imdata.load_data_gen,  (tf.float32, tf.int32)).prefetch(12)
else:
    dataset = tf.data.Dataset.from_generator(imdata.load_vid_data_gen, (tf.float32, tf.int32))
next_batch = dataset.make_one_shot_iterator().get_next()

if tmp == 'ascii':
    y = imdata.get_templates(path='./assets/char_set_alt/', num_temps=NUM_TEMPLATES)
elif tmp == 'faces':
    y = imdata.get_templates(path='./assets/face_templates/', num_temps=NUM_TEMPLATES)
elif tmp == 'emoji':
    y = imdata.get_emoji_templates(path='./assets/emoji_temps/', num_temps=NUM_TEMPLATES)
else:
    y = imdata.get_templates(path='./assets/cam_templates/', num_temps=NUM_TEMPLATES)
#########################################


############Pebbles Test#####################
# x = imdata.get_pebbles(path='./pebbles.jpg')
# x = tf.convert_to_tensor(x, tf.float32)
#############################################


##############Build Graph###################
with tf.device('/gpu:'+str(0)):
    input = tf.placeholder(tf.float32, shape=(6, img_orig_size, img_orig_size, 3))
    m = ASCIINet(images=input, templates=y, rgb=rgb)
    tLoss = m.tLoss
    opt, lr = optimize(tLoss)
    argmax = tf.one_hot(tf.argmax(m.softmax, axis=-1), depth=NUM_TEMPLATES)
    o = predictTop(argmax, m.temps, batch_size=6, rgb=rgb, num_temps=NUM_TEMPLATES, img_size=img_new_size, patch_size=patch_size, softmax_size=m.softmax_size)
merged = tf.summary.merge_all()
############################################



############Training################################
chkpt = tf.train.Saver()
lrate = base_lr
print('Num Variables: ', np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.all_variables()]))

with sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    n = 0
    for i in range(iterations):
        #Temperature Schedule
        if i > 1 and i % 1000 == 0:
            if i < 8000:
                t *= 2

        #Forward Pass
        x, ind = sess.run(next_batch)


        feed_dict = {
                        input: x,
                        lr: lrate,
                        m.temp: t
                     }

        summary, result, totalLoss = sess.run([merged,  opt,  tLoss],
                                              feed_dict=feed_dict)

        ################Saving/Logging############################
        if i % update == 0:
            print('Template Set: ', tmp)
            print('Colour: ', rgb)
            print('Iteration #:', str(i))
            print('temperature: ' + str(t))
            print('Learning Rate:', str(lrate))
            print('Loss:', str(totalLoss))
            print('Index:', ind)
            print(':::::::::Softmax Max Values::::::::::::')
            print(sess.run(tf.reduce_max(m.softmax[0, 0, 0, :]),  feed_dict={input: x, m.temp: t}))
            print('Experiment directory: ', experiments_dir)
            print('Save directory: ', snapshot_dir)
            print('Log directory: ', log_dir)


            values = sess.run(m.softmax[0,  17, :, :],  feed_dict={input: x, m.temp: t})
            for j in range(64):
                log_histogram(writer,  'coeff' + str(j),  values[j, :], i, bins=NUM_TEMPLATES)
            writer.add_summary(summary, i+1)
            chkpt.save(sess, snapshot_dir + 'checkpoint.chkpt')
            misc.imsave(im_dir + str(i) + 'i.jpg', sess.run(m.input[1], feed_dict={input: x, m.temp: t}))
            misc.imsave(im_dir + str(i) + 's.jpg', sess.run(m.view_output[1], feed_dict={input: x, m.temp: t}))
            misc.imsave(im_dir + str(i) + 'o.jpg', sess.run(o[1], feed_dict={input: x, m.temp: t}))
            # misc.imsave(im_dir + str(i) + 'o.jpg', sess.run(o[1], feed_dict={input: x, m.temp: t}))
            # misc.imsave(im_dir + str(i) + 'o.jpg', sess.run(o[2], feed_dict={input: x, m.temp: t}))
            # misc.imsave(im_dir + str(i) + 'o.jpg', sess.run(o[3], feed_dict={input: x, m.temp: t}))
            # misc.imsave(im_dir + str(i) + 'o.jpg', sess.run(o[4], feed_dict={input: x, m.temp: t}))
            # misc.imsave(im_dir + str(i) + 'o.jpg', sess.run(o[5], feed_dict={input: x, m.temp: t}))
        ##############################################################

slack_msg = 'Experiment done on gpu #' + str(gpu) + " on Delta"
slack_notify('nariman_saftarli', slack_msg)