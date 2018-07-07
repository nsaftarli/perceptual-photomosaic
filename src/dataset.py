import os
from functools import partial
import numpy as np
import tensorflow as tf
from PIL import Image
import itertools
from scipy.misc import imread


class Dataset:

    def __init__(self, training_path, validation_path, prediction_path, config):
        self.train_path = training_path
        self.val_path = validation_path
        self.pred_path = prediction_path
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, (tf.float32, tf.int32, tf.int32))
        train_generator = partial(self.data_generator, path=self.train_path)
        val_generator = partial(self.data_generator, path=self.val_path, train=False)
        pred_generator = partial(self.data_generator, path=self.pred_path, train=False)
        self.train_dataset = \
            tf.data.Dataset.from_generator(train_generator, (tf.float32, tf.int32, tf.int32))
        self.val_dataset = \
            tf.data.Dataset.from_generator(val_generator, (tf.float32, tf.int32, tf.int32))
        self.pred_dataset = \
            tf.data.Dataset.from_generator(pred_generator, (tf.float32, tf.int32, tf.int32))
        self.config = config

    def get_training_handle(self):
        self.train_dataset = \
            self.train_dataset.prefetch(self.config['batch_size'] * 3) \
                              .batch(self.config['batch_size'])
        self.train_iterator = self.train_dataset.make_one_shot_iterator()
        return self.train_iterator.string_handle()

    def get_validation_handle(self):
        self.val_dataset = \
            self.val_dataset.prefetch(self.config['batch_size'] * 3) \
                            .batch(self.config['batch_size'])
        self.val_iterator = self.val_dataset.make_initializable_iterator()
        return self.val_iterator.string_handle()

    def get_prediction_handle(self):
        self.pred_dataset = \
            self.pred_dataset.prefetch(self.config['batch_size'] * 3) \
                             .batch(self.config['batch_size'])
        self.pred_iterator = self.pred_dataset.make_initializable_iterator()
        return self.pred_iterator.string_handle()

    def data_generator(self, path, train=True):
        files = sorted(os.listdir(path))
        num_files = len(files)
        for i in (itertools.count(1) if train else range(num_files)):
            img = imread(path + files[i % num_files], mode='RGB').astype('float32')
            yield img, i, num_files


def get_templates(path):
    files = sorted(os.listdir(path))
    num_templates = len(files)
    images = []
    for i in range(num_templates):
        im = imread(path + files[i], mode='RGB').astype('float32')
        images.append(im)
    images = np.expand_dims(np.stack(images, axis=-1), axis=0)
    return images


# Move to utils
def turn_im_into_templates(path, patch_size=8):
    im = imread(path)
    x = 0
    for i in range(im.shape[0] // patch_size):
        for j in range(im.shape[1] // patch_size):
            patch = im[i*patch_size:i*(patch_size+1), j*patch_size:j*(patch_size+1), :]
            out = Image.fromarray(patch.astype('uint8'))
            out.save('./assets/cam_templates_2/' + str(x) + '.png', 'PNG')
            x += 1
