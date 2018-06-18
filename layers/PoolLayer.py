import tensorflow as tf


def PoolLayer(x, name, trainable=True):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=name)
