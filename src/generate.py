import sys
# sys.path.append('layers/')
# sys.path.append('utils/')
import os
import time
import argparse
import tensorflow as tf
import numpy as np
import imdata
from model import *
# from tfoptimizer import *
from layers import LossLayer
from utils import *
import scipy.misc as misc
# from predictTop import *


######################## DATA ####################
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

####################################################


# COMMAND LINE ARGS
parser = argparse.ArgumentParser(description='training')
parser.add_argument('-g', '--gpu', default=1, type=int)
parser.add_argument('-b', '--batch_size', default=1, type=int)
parser.add_argument('-i', '--iterations', default=1, type=int)
parser.add_argument('-tr', '--train', default=True, type=bool)
parser.add_argument('-u', '--update', default=100, type=int)
parser.add_argument('-lr', '--learning_rate', default=1e-6, type=float)
parser.add_argument('-t', '--temperature', default=1.0, type=float)
parser.add_argument('-n', '--notes', default=None, type=str)
parser.add_argument('-d', '--dataset', default='Faces', type=str)
parser.add_argument('-rgb', '--colour', default=False, type=bool)
parser.add_argument('-tmp', '--templates', default='other', type=str)
parser.add_argument('-v', '--video', default=False, type=bool)
parser.add_argument('-logf', '--log_freq', default=10, type=int)
parser.add_argument('-savef', '--save_freq', default=500, type=int)
parser.add_argument('-chkpt', '--chkpt_freq', default=500, type=int)
parser.add_argument('-folder', '--template_folder', default='char_set_alt', type=str)
parser.add_argument('-id', '--runid', default='run', type=str)
args = parser.parse_args()


# SETTINGS
my_config = {}
my_config['gpu'] = args.gpu
my_config['batch_size'] = args.batch_size
my_config['iterations'] = args.iterations
my_config['train'] = args.train
my_config['learning_rate'] = args.learning_rate
my_config['temperature'] = args.temperature
my_config['notes'] = args.notes
my_config['template_folder'] = args.template_folder
my_config['chkpt_freq'] = args.chkpt_freq
my_config['save_freq'] = args.save_freq
my_config['log_freq'] = args.log_freq
my_config['runid'] = args.runid



gpu = args.gpu
iterations = args.iterations
train = args.train
update = args.update
base_lr = args.learning_rate
t = args.temperature
notes = args.notes
# dset = args.dset
# rgb = args.rgb
# tmp = args.tmp
# vid = args.vid
logfreq = args.log_freq
savefreq = args.save_freq
chkptfreq = args.chkpt_freq
template_folder = args.template_folder
runid = args.runid
NUM_TEMPLATES = const.num_templates

# GPU Settings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
# sess = tf.Session(config=config)


# FILE HANDLING
if runid == 'run':
    now = time.strftime('%d%b-%X')
    filename = runid + now + '/'
else:
    filename = runid + '/'
# folder = experiments_dir + now
log_dir = '../logs/' + filename
snapshot_dir = '../snapshots/' + filename
im_dir = '../images/' + filename
notes_dir = '../notes/' + filename
print(notes_dir)

# if not os.path.exists(folder):
    # os.makedirs(folder)
os.makedirs(log_dir)
os.makedirs(snapshot_dir)
os.makedirs(im_dir)
os.makedirs(notes_dir)

with open(notes_dir + 'info.txt', 'w') as fp:
    fp.write(runid + '\n')
    fp.write('Hyperparameters' + '\n')
    fp.write('# Iterations: ' + str(iterations) + '\n')
    fp.write('Snapshot Freq: ' + str(update) + '\n')
    fp.write('Learning Rate: ' + str(base_lr) + '\n')
    fp.write('Initial Temperature: ' + str(t) + '\n')
    fp.write('Template Folder: ' + template_folder + '\n')
    fp.write('Notes: ' + '\n')
    if notes is not None:
        fp.write(notes + '\n')


# DATA INPUT
if train:
    dataset = tf.data.Dataset.from_generator(
        imdata.load_data_gen,
        (tf.float32, tf.int32)).prefetch(12)
else:
    dataset = tf.data.Dataset.from_generator(
        imdata.load_vid_data_gen,
        (tf.float32, tf.int32)).prefetch(12)
next_batch = dataset.make_one_shot_iterator().get_next()

y = imdata.get_templates(path='../data/' + template_folder + '/')

############Pebbles Test#####################
# x = imdata.get_pebbles(path='./assets/pebbles.jpg')
# x = tf.convert_to_tensor(x, tf.float32)
#############################################


##############Build Graph###################
# with tf.device('/gpu:'+str(0)):
#     input = tf.placeholder(tf.float32, shape=(6, 376, img_orig_size, 3))
#     m = MosaicNet(images=input, templates=y, rgb=rgb, )
#     tLoss = m.tLoss
#     opt, lr = optimize(tLoss)
#     argmax = tf.one_hot(tf.argmax(m.softmax, axis=-1), depth=NUM_TEMPLATES)
#     o = predictTop(argmax, m.temps, batch_size=6, rgb=rgb, num_temps=NUM_TEMPLATES, img_size=img_new_size, patch_size=patch_size, softmax_h=m.softmax_h, softmax_w=m.softmax_w)
# merged = tf.summary.merge_all()
with tf.device('/gpu:' + str(0)):
    m = MosaicNet(y, config=config, my_config=my_config)
    tLoss = m.tLoss
    opt, lr = optimize(tLoss)
print('Num Variables: ', np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.all_variables()]))
############################################



############Training################################
chkpt = tf.train.Saver()
lrate = base_lr

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
        if i % savefreq == 0:
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
            misc.imsave(im_dir + str(i) + 'i.png', sess.run(m.input[1], feed_dict={input: x, m.temp: t}))
            misc.imsave(im_dir + str(i) + 's.png', sess.run(m.view_output[1], feed_dict={input: x, m.temp: t}))
            misc.imsave(im_dir + str(i) + 'o.png', sess.run(o[1], feed_dict={input: x, m.temp: t}))


        if i % logfreq == 0:
            writer.add_summary(summary, i+1)
        if i % chkptfreq == 0:
            chkpt.save(sess, snapshot_dir + 'checkpoint_' + str(i) + '.chkpt')


            # values = sess.run(m.softmax[0,  17, :, :],  feed_dict={input: x, m.temp: t})
            # for j in range(64):
                # log_histogram(writer,  'coeff' + str(j),  values[j, :], i, bins=NUM_TEMPLATES)
        ##############################################################

chkpt.save(sess, snapshot_dir + 'checkpoint_final.chkpt')
slack_msg = 'Experiment done on gpu #' + str(gpu) + ' on Delta'
slack_notify('nariman_saftarli', slack_msg)
