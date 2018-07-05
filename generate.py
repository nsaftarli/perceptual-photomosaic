import os
import time
import argparse
import tensorflow as tf
import numpy as np
from src.dataset import Dataset, get_templates
from src.model import *
from src.layers import LossLayer
from src.utils import *
from scipy.misc import imsave


# COMMAND LINE ARGS
parser = argparse.ArgumentParser(description='training')
parser.add_argument('-g', '--gpu', default=1, type=int)
parser.add_argument('-b', '--batch_size', default=1, type=int)
parser.add_argument('-i', '--iterations', default=1, type=int)
parser.add_argument('-tr', '--train', default=True, type=bool)
parser.add_argument('-lr', '--learning_rate', default=1e-6, type=float)
parser.add_argument('-t', '--init_temperature', default=1.0, type=float)
parser.add_argument('-n', '--notes', default=None, type=str)
parser.add_argument('-logf', '--log_freq', default=10, type=int)
parser.add_argument('-savef', '--save_freq', default=500, type=int)
parser.add_argument('-chkpt', '--chkpt_freq', default=500, type=int)
parser.add_argument('-dpath', '--data_folder', default='./data/coco_resized_512/', type=str)
parser.add_argument('-folder', '--template_folder', default='black_ascii_8', type=str)
parser.add_argument('-id', '--run_id', default=time.strftime('%d%b-%X'), type=str)
args = parser.parse_args()


# SETTINGS
my_config = {}
my_config['batch_size'] = args.batch_size
my_config['train'] = args.train
my_config['learning_rate'] = args.learning_rate
my_config['init_temperature'] = args.temperature
my_config['chkpt_freq'] = args.chkpt_freq
my_config['save_freq'] = args.save_freq
my_config['log_freq'] = args.log_freq
my_config['run_id'] = args.runid


# For Simplicity
gpu = args.gpu
batch_size = args.batch_size
iterations = args.iterations
train = args.train
base_lr = args.learning_rate
t = args.temperature
notes = args.notes
log_freq = args.log_freq
save_freq = args.save_freq
chkpt_freq = args.chkpt_freq
template_folder = args.template_folder
data_folder = args.data_folder
run_id = args.run_id


# GPU SETTINGS
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True


# FILE HANDLING
log_dir = '../logs/' + run_id + '/'
snapshot_dir = '../snapshots/' + run_id + '/'
im_dir = '../data/out/' + run_id + '/'
notes_dir = '../notes/' + run_id + '/'
os.makedirs(log_dir)
os.makedirs(snapshot_dir)
os.makedirs(im_dir)
os.makedirs(notes_dir)


# Write to notes file
with open(notes_dir + 'info.txt', 'w') as fp:
    fp.write(run_id + '\n')
    fp.write('Hyperparameters' + '\n')
    fp.write('# Iterations: ' + str(iterations) + '\n')
    fp.write('Checkpoint Freq: ' + str(chkpt_freq) + '\n')
    fp.write('Learning Rate: ' + str(base_lr) + '\n')
    fp.write('Initial Temperature: ' + str(t) + '\n')
    fp.write('Template Folder: ' + template_folder + '\n')
    fp.write('Notes: ' + '\n')
    if notes is not None:
        fp.write(notes + '\n')


# DATA INPUT
d = Dataset(path=data_folder)
dataset = tf.data.Dataset.from_generator(
    d.data_generator,
    (tf.float32, tf.int32)).prefetch(12).batch(batch_size)
# next_batch = dataset.make_one_shot_iterator().get_next()
templates = get_templates(path='./data/' + template_folder + '/')


# BUILD GRAPH
with tf.device('/gpu:' + str(gpu)):
    m = MosaicNet(dataset, templates, tf_config=config, my_config=my_config)


# TRAINING
m.train(iterations)
# chkpt = tf.train.Saver()
# lrate = base_lr
#
# with sess:
#     sess.run(tf.global_variables_initializer())
#     writer = tf.summary.FileWriter(log_dir, sess.graph)
#     n = 0
#     for i in range(iterations):
#         #Temperature Schedule
#         if i > 1 and i % 1000 == 0:
#             if i < 8000:
#                 t *= 2
#
#         #Forward Pass
#         x, ind = sess.run(next_batch)
#
#
#         feed_dict = {
#                         m.input: x,
#                         lr: lrate,
#                         m.temp: t
#                      }
#
#         summary, result, totalLoss = sess.run([merged,  opt,  tLoss],
#                                               feed_dict=feed_dict)
#
#         ################Saving/Logging############################
#         if i % savefreq == 0:
#             print('Template Set: ', tmp)
#             print('Colour: ', rgb)
#             print('Iteration #:', str(i))
#             print('temperature: ' + str(t))
#             print('Learning Rate:', str(lrate))
#             print('Loss:', str(totalLoss))
#             print('Index:', ind)
#             print(':::::::::Softmax Max Values::::::::::::')
#             print(sess.run(tf.reduce_max(m.softmax[0, 0, 0, :]),  feed_dict={input: x, m.temp: t}))
#             print('Experiment directory: ', experiments_dir)
#             print('Save directory: ', snapshot_dir)
#             print('Log directory: ', log_dir)
#             misc.imsave(im_dir + str(i) + 'i.png', sess.run(m.input[1], feed_dict={input: x, m.temp: t}))
#             misc.imsave(im_dir + str(i) + 's.png', sess.run(m.view_output[1], feed_dict={input: x, m.temp: t}))
#             misc.imsave(im_dir + str(i) + 'o.png', sess.run(o[1], feed_dict={input: x, m.temp: t}))
#
#
#         if i % logfreq == 0:
#             writer.add_summary(summary, i+1)
#         if i % chkptfreq == 0:
#             chkpt.save(sess, snapshot_dir + 'checkpoint_' + str(i) + '.chkpt')


            # values = sess.run(m.softmax[0,  17, :, :],  feed_dict={input: x, m.temp: t})
            # for j in range(64):
                # log_histogram(writer,  'coeff' + str(j),  values[j, :], i, bins=NUM_TEMPLATES)
        ##############################################################

slack_msg = 'Experiment done on gpu #' + str(gpu) + ' on Delta'
slack_notify('nariman_saftarli', slack_msg)
