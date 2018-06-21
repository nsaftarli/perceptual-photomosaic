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
coco_dir = const.coco_dir






####Command line arguments#######################
argParser = argparse.ArgumentParser(description='training')
argParser.add_argument('-g', '--gpu', dest="gpu", action="store", default=1, type=int)
argParser.add_argument('-i', '--iterations', dest='iterations', action='store', default=0, type=int)
argParser.add_argument('-u', '--update', dest='update', action='store', default=100, type=int)
argParser.add_argument('-lr', '--learning-rate',  dest='lr', action='store', default=1e-6, type=float)
argParser.add_argument('-db', '--debug', dest='debug', action='store', default=False, type=bool)
argParser.add_argument('-t', '--temp', dest='temp',  action='store',  default=1.0,  type=float)
argParser.add_argument('-n', '--notes', dest='notes', action='store', default=None, type=str)
argParser.add_argument('-d', '--dataset', dest='dset', action='store', default='Faces', type=str)
cmdArgs = argParser.parse_args()
##################################################





####Settings######################################
gpu = cmdArgs.gpu
iterations = cmdArgs.iterations
update = cmdArgs.update
base_lr = cmdArgs.lr
debug = cmdArgs.debug
t = cmdArgs.temp
notes = cmdArgs.notes
dset = cmdArgs.dset


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
dataset = tf.data.Dataset.from_generator(imdata.load_data_gen,  (tf.float32,tf.int32))
next_batch = dataset.make_one_shot_iterator().get_next()
#########################################


y = imdata.get_templates(path='./assets/char_set_alt/', num_chars=62)
# y = imdata.get_templates(path='./assets/cam_templates/', num_chars=62)




############Pebbles Test#####################
# x = imdata.get_pebbles(path='./pebbles.jpg')
# x = tf.convert_to_tensor(x, tf.float32)
#############################################




##############Build Graph###################
with tf.device('/gpu:'+str(0)):
    input = tf.placeholder(tf.float32, shape=(6, 224, 224, 3))
    # input = tf.image.resize_images(input,size=[224,224])
    m = ASCIINet(images=input, templates=y)
    tLoss = m.tLoss
    opt,  lr = optimize(tLoss)
merged = tf.summary.merge_all()
############################################



############Training################################
chkpt = tf.train.Saver()
lrate = base_lr

with sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    for i in range(iterations):
        x, ind = sess.run(next_batch)
        # print(x.shape)
        feed_dict = {input: x,
                     lr: lrate,
                     m.temp: t}

        summary, result, totalLoss = sess.run([merged,  opt,  tLoss],
                                              feed_dict=feed_dict)


        if i % update == 0:
            print('Iteration #:', str(i))
            print('temperature: ' + str(t))
            print('Learning Rate:', str(lrate))
            print('Loss:', str(totalLoss))
            print('Index:', ind)



            values = sess.run(m.softmax[0,  17, :, :],  feed_dict={input: x,m.temp: t})


            print(':::::::::Softmax Values::::::::::::')
            print(sess.run(m.softmax[0, 0, 0, :],  feed_dict={input: x, m.temp: t}))

            #################Saving/Logging###################
            for j in range(28):
                log_histogram(writer,  'coeff' + str(j),  values[j, :], i)
            writer.add_summary(summary, i+1)
            chkpt.save(sess, snapshot_dir + 'checkpoint.chkpt')
            misc.imsave(im_dir + str(i) + '.jpg', sess.run(m.view_output[0], feed_dict={input: x, m.temp: t}))
            ##################################################

            if debug:
                print("Input Range:", sess.run(m.gray_im[0, 3:7, 3:7, :]))
                print("Output Range:", sess.run(m.reshaped_output[0, 3:7, 3:7, :],  feed_dict={m.temp: t}))


        ###############Temperature Schedule####################
        if i > 1 and i % 1000 == 0:
            if i < 10000:
                t *= 2
        #######################################################
#######################################################
