import sys
# sys.path.append('layers/')
# sys.path.append('utils/')
import os
import tensorflow as tf
import imdata
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]


class VGG16:
    def __init__(self, input, weight_path='../weights/vgg16.npy', trainable=False):
        self.vgg_weights = np.load(weight_path, encoding='latin1').item()
        self.vgg = self.build_vgg(input, trainable=trainable)

    def build_vgg(self, input, trainable):

        with tf.name_scope('mean_subtract'):
            r, g, b = tf.split(input, 3, axis=3)
            self.vgg_input = tf.concat([
                b - VGG_MEAN[0],
                g - VGG_MEAN[1],
                r - VGG_MEAN[2]], axis=3)


        #################Block 1################
        self.conv1_1 = self.conv_layer(self.vgg_input, name='conv1_1', trainable=trainable)
        self.conv1_2 = self.conv_layer(self.conv1_1, name='conv1_2', trainable=trainable)
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        #################Block 2################
        self.conv2_1 = self.conv_layer(self.pool1, name='conv2_1', trainable=trainable)
        self.conv2_2 = self.conv_layer(self.conv2_1, name='conv2_2', trainable=trainable)
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        #################Block 3################
        self.conv3_1 = self.conv_layer(self.pool2, name='conv3_1', trainable=trainable)
        self.conv3_2 = self.conv_layer(self.conv3_1, name='conv3_2', trainable=trainable)
        self.conv3_3 = self.conv_layer(self.conv3_2, name='conv3_3', trainable=trainable)
        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        #################Block 4################
        self.conv4_1 = self.conv_layer(self.pool3, name='conv4_1', trainable=trainable)
        self.conv4_2 = self.conv_layer(self.conv4_1, name='conv4_2', trainable=trainable)
        self.conv4_3 = self.conv_layer(self.conv4_2, name='conv4_3', trainable=trainable)
        self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
        #################Block 5################
        self.conv5_1 = self.conv_layer(self.pool4, name='conv5_1', trainable=trainable)
        self.conv5_2 = self.conv_layer(self.conv5_1, name='conv5_2', trainable=trainable)
        self.conv5_3 = self.conv_layer(self.conv5_2, name='conv5_3', trainable=trainable)
        self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

        self.output = self.pool5

        # self.print_vgg()

    def conv_layer(self, x, name, trainable):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            w = self.get_vgg_weights(name, trainable=trainable)
            b = self.get_vgg_biases(name, trainable=trainable)
            z = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
            activation = tf.nn.relu(z)
        return activation

    def get_vgg_weights(self, name, trainable):
        return tf.get_variable(initializer=self.vgg_weights[name][0],
                               name='filter',
                               trainable=trainable)

    def get_vgg_biases(self, name, trainable):
        return tf.get_variable(initializer=self.vgg_weights[name][1],
                               name='biases',
                               trainable=trainable)

    def print_vgg(self):
        print(self.conv1_1.get_shape())
        print(self.conv1_2.get_shape())
        print(self.conv2_1.get_shape())
        print(self.conv2_2.get_shape())
        print(self.conv3_1.get_shape())
        print(self.conv3_2.get_shape())
        print(self.conv3_3.get_shape())
        print(self.conv4_1.get_shape())
        print(self.conv4_2.get_shape())
        print(self.conv4_3.get_shape())
        print(self.conv5_1.get_shape())
        print(self.conv5_2.get_shape())
        print(self.conv5_3.get_shape())


if __name__ == '__main__':
    pebbles = imdata.get_pebbles()
    pebbles = tf.convert_to_tensor(pebbles, tf.float32)
    vgg = VGG16(pebbles)
