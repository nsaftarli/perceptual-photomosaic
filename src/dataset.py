import os
from functools import partial
import numpy as np
import tensorflow as tf
import itertools
from scipy.misc import imread


class Dataset:

    def __init__(self, training_path, validation_path, prediction_path, config):
        self.train_path = training_path
        self.val_path = validation_path
        self.pred_path = prediction_path
        self.config = config
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, (tf.float32, tf.int32, tf.int32))

        train_generator = self.data_generator_gpu(self.train_path)
        self.train_dataset = tf.data.Dataset.from_tensor_slices(train_generator)

        val_generator = self.data_generator_gpu(self.val_path)
        self.val_dataset = tf.data.Dataset.from_tensor_slices(val_generator)

        if not config['train']:
            pred_generator = self.data_generator_gpu(self.pred_path)
            self.pred_dataset = tf.data.Dataset.from_tensor_slices(pred_generator)


    def get_training_handle(self):
        self.train_dataset = \
            self.train_dataset.map(self.read_file) \
                              .prefetch(self.config['batch_size'] * 3) \
                              .batch(self.config['batch_size'])
        self.train_iterator = self.train_dataset.make_one_shot_iterator()
        return self.train_iterator.string_handle()

    def get_validation_handle(self):
        self.val_dataset = \
            self.val_dataset.map(self.read_file) \
                            .prefetch(self.config['batch_size'] * 3) \
                            .batch(self.config['batch_size'])
        self.val_iterator = self.val_dataset.make_initializable_iterator()
        return self.val_iterator.string_handle()

    def get_prediction_handle(self):
        self.pred_dataset = \
            self.pred_dataset.map(self.read_file) \
                             .prefetch(self.config['batch_size'] * 3) \
                             .batch(self.config['batch_size'])
        self.pred_iterator = self.pred_dataset.make_initializable_iterator()
        return self.pred_iterator.string_handle()

    def data_generator_gpu(self, path):
        filenames = [os.path.join(path, f) for f in sorted(os.listdir(path))]
        num_files = len(filenames)
        indices = list(range(num_files))
        filenames = tf.constant(filenames)
        num_files = tf.constant([num_files] * num_files)
        indices = tf.constant(indices)
        return (filenames, indices, num_files)

    def read_file(self, filename, index, num_files):
        image_string = tf.read_file(filename)
        image_decoded = tf.to_float(tf.image.decode_image(image_string,
                                                          channels=3))
        image_decoded.set_shape([None, None, 3])
        return image_decoded, index, num_files




def get_templates(path):
    files = sorted(os.listdir(path))
    num_templates = len(files)
    images = []
    for i in range(num_templates):
        im = imread(path + files[i], mode='RGB').astype('float32')
        images.append(im)
    images = np.expand_dims(np.stack(images, axis=-1), axis=0)
    return images
