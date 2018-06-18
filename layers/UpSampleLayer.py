import tensorflow as tf


def UpSampleLayer(x, scale_factor, name):
    with tf.variable_scope(name):
        in_shape = x.get_shape()
        upsampled = tf.image.resize_images(x,
            size=[in_shape[1].value * scale_factor, in_shape[2].value * scale_factor])
        return upsampled
