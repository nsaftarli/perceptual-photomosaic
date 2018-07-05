import tensorflow as tf
from Normalization import GroupNorm


def ConvLayer(x, name, out_channels, ksize=3, stride=1, activation='leaky_relu', trainable=True):

    in_channels = x.get_shape()[3].value
    shape_in = [ksize, ksize, in_channels, out_channels]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable('weight', initializer=tf.contrib.layers.xavier_initializer(), shape=shape_in, trainable=trainable)
        b = tf.get_variable('bias', initializer=tf.constant(0.0, shape=[out_channels], dtype=tf.float32), trainable=trainable)
        z = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME') + b

        if activation is not None:
            z = GroupNorm(z, G=2)
            z = tf.nn.leaky_relu(z)
        else:
            z = GroupNorm(z, G=1)

        return z
