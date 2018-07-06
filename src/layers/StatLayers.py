import tensorflow as tf
import numpy as np


def VarianceLayer(prob, num_bins):
    bins = tf.reshape(
        np.linspace(1, num_bins, num=num_bins).astype('float32'),
        [1, 1, 1, num_bins])
    mean = tf.reduce_sum(bins * prob, axis=3)
    mean_2 = tf.reduce_sum(bins ** 2 * prob, axis=3)
    variance = mean_2 - mean ** 2
    return tf.reduce_mean(variance)


def EntropyLayer(prob):
    return tf.reduce_mean(-1.0 * tf.reduce_sum(prob * tf.log(prob + 1e-8),
                          axis=3))
