import tensorflow as tf


def InstanceNorm(input, epsilon=1e-12):
    mean, var = tf.nn.moments(input, axes=[1, 2, 3])
    return (input - mean) / (tf.sqrt(var+epsilon))


def LayerNorm(input, eps=1e-12):
    mean, var = tf.nn.moments(input, axes=[0, 1, 2])
    return (input-mean)/tf.sqrt(var+eps)


def BatchNorm(x):
    y = tf.layers.batch_normalization(x, training=True)
    return y


def GroupNorm(x, G, eps=1e-12):
    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x-mean)/tf.sqrt(var+eps)
    x = tf.reshape(x, [N, C, H, W])
    return x
