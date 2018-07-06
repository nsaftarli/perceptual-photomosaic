import os
import time
import argparse
import tensorflow as tf
from src.dataset import Dataset, get_templates
from src.model import MosaicNet
from src.utils import slack_notify


# COMMAND LINE ARGS
parser = argparse.ArgumentParser(description='training')
parser.add_argument('-g', '--gpu', default=1, type=int)
parser.add_argument('-b', '--batch_size', default=6, type=int)
parser.add_argument('-i', '--iterations', default=5000, type=int)
parser.add_argument('-tr', '--train', default=True, type=bool)
parser.add_argument('-lr', '--learning_rate', default=1e-6, type=float)
parser.add_argument('-t', '--init_temperature', default=1.0, type=float)
parser.add_argument('-n', '--notes', default=None, type=str)
parser.add_argument('-logf', '--log_freq', default=10, type=int)
parser.add_argument('-printf', '--print_freq', default=10, type=int)
parser.add_argument('-chkpt', '--chkpt_freq', default=500, type=int)
parser.add_argument('-valf', '--val_freq', default=500, type=int)
parser.add_argument('-tpath', '--train_path', default='data/coco_resized_512_train/', type=str)
parser.add_argument('-vpath', '--val_path', default='data/coco_resized_512_val/', type=str)
parser.add_argument('-folder', '--template_folder', default='black_ascii_8', type=str)
parser.add_argument('-id', '--run_id', default=time.strftime('%d%b-%X'), type=str)
args = parser.parse_args()


# SETTINGS
my_config = {}
my_config['batch_size'] = args.batch_size
my_config['train'] = args.train
my_config['learning_rate'] = args.learning_rate
my_config['init_temperature'] = args.init_temperature
my_config['chkpt_freq'] = args.chkpt_freq
my_config['val_freq'] = args.val_freq
my_config['print_freq'] = args.print_freq
my_config['log_freq'] = args.log_freq
my_config['run_id'] = args.run_id
my_config['iterations'] = args.iterations


# For Simplicity
gpu = args.gpu
batch_size = args.batch_size
iterations = args.iterations
train = args.train
base_lr = args.learning_rate
t = args.init_temperature
notes = args.notes
template_folder = args.template_folder
train_path = args.train_path
val_path = args.val_path
run_id = args.run_id


# GPU SETTINGS
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True


# DATA INPUT
d = Dataset(training_path=train_path, validation_path=val_path, config=my_config)
templates = get_templates(path='data/' + template_folder + '/')


# BUILD GRAPH
with tf.device('/gpu:' + str(gpu)):
    m = MosaicNet(d, templates, tf_config=config, my_config=my_config)


if train:
    # Set up folders for training
    log_dir = 'logs/' + run_id + '/'
    snapshot_dir = 'snapshots/' + run_id + '/'
    im_dir = 'data/out/' + run_id + '/'
    notes_dir = 'notes/' + run_id + '/'
    try:
        os.makedirs(log_dir)
        os.makedirs(snapshot_dir)
        os.makedirs(im_dir)
        os.makedirs(notes_dir)
    except OSError:
        pass

    # Write to notes file
    with open(notes_dir + 'info.txt', 'w') as fp:
        fp.write(run_id + '\n')
        fp.write('Hyperparameters' + '\n')
        fp.write('# Iterations: ' + str(iterations) + '\n')
        fp.write('Learning Rate: ' + str(base_lr) + '\n')
        fp.write('Initial Temperature: ' + str(t) + '\n')
        fp.write('Template Folder: ' + template_folder + '\n')
        fp.write('Notes: ' + '\n')
        if notes is not None:
            fp.write(notes + '\n')

    # Start Training
    m.train()
else:
    m.predict()

slack_msg = 'Experiment done on gpu #' + str(gpu) + ' on Delta'
slack_notify('nariman_saftarli', slack_msg)
