import tensorflow as tf


def InstanceNormLayer(input, epsilon=1e-12):
    mean, var = tf.nn.moments(input, axes=[1, 2, 3])
    return (input - mean) / (tf.sqrt(var+epsilon))


def LayerNormLayer(input, eps=1e-12):
    mean, var = tf.nn.moments(input, axes=[0, 1, 2])
    return (input-mean)/tf.sqrt(var+eps)


def BatchNormLayer(x):
    y = tf.layers.batch_normalization(x, training=True)
    return y


def GroupNormLayer(x, G, eps=1e-12):
    N, H, W, C = x.shape
    x = tf.reshape(x, [N, G, H, W, C // G])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x-mean)/tf.sqrt(var+eps)
    x = tf.reshape(x, [N, H, W, C])
    return x
