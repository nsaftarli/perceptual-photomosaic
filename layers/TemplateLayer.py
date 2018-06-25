import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]

def TemplateLayer(templates, rgb=False):
    if not rgb:
        return tf.constant(value=templates, dtype=tf.float32, shape=templates.shape, name='templates')
    else:
        temps = tf.constant(value=templates, dtype=tf.float32, shape=templates.shape, name='templates')
        print('AAAAAAAAAAAAAAA')
        print(temps.get_shape())
        r, g, b = tf.split(temps, 3, axis=3)
        r = tf.squeeze(r, axis=3) #- VGG_MEAN[2]
        g = tf.squeeze(g, axis=3) #- VGG_MEAN[1]
        b = tf.squeeze(b, axis=3) #- VGG_MEAN[0]
        # r = tf.tile(r, [])
        return (r, g, b)
