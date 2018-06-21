import tensorflow as tf


def TemplateLayer(templates, rgb=False):
    return tf.constant(value=templates, dtype=tf.float32, shape=templates.shape, name='templates')
