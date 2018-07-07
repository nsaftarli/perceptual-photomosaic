import tensorflow as tf


def InstanceNormLayer(input, name, epsilon=1e-12):
    with tf.name_scope(name):
        mean, var = tf.nn.moments(input, axes=[1, 2, 3])
        return (input - mean) / (tf.sqrt(var+epsilon))


def LayerNormLayer(input, name, eps=1e-12):
    with tf.name_scope(name):
        mean, var = tf.nn.moments(input, axes=[0, 1, 2])
        return (input-mean)/tf.sqrt(var+eps)


def BatchNormLayer(input, name):
    with tf.name_scope(name):
        y = tf.layers.batch_normalization(input, training=True)
        return y


def GroupNormLayer(input, name, G, eps=1e-12):
    with tf.name_scope(name):
        input_shape = tf.shape(input)
        N, H, W, C = [input_shape[0], input_shape[1], input_shape[2],
                      input.get_shape().as_list()[3]]
        input = tf.reshape(input, [N, G, H, W, C // G])
        mean, var = tf.nn.moments(input, [2, 3, 4], keep_dims=True)
        input = (input-mean)/tf.sqrt(var+eps)
        input = tf.reshape(input, [N, H, W, C])
        return input
