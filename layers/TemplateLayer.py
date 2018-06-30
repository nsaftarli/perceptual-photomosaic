import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]

def TemplateLayer(templates, rgb=False, new_size=8):
    if not rgb:
        temps = tf.constant(value=templates, dtype=tf.float32, shape=templates.shape, name='templates')
        print('BBBBBBBBBBBBBBBB')
        print(temps.get_shape())
        print(tf.reduce_mean(temps, axis=3).get_shape())
        return tf.reduce_mean(tf.constant(value=templates, dtype=tf.float32, shape=templates.shape, name='templates'), axis=3)
    else:
        temps = tf.constant(value=templates, dtype=tf.float32, shape=templates.shape, name='templates')
        temps = tf.image.resize_images(tf.transpose(temps,[0, 4, 1, 2, 3])[0],
                                       size=[new_size, new_size],
                                       method=tf.image.ResizeMethod.BILINEAR)
        temps = tf.expand_dims(tf.transpose(temps,[1, 2, 3, 0]), 0)
        print('AAAAAAAAAAAAAAA')
        print(temps.get_shape())
        r, g, b = tf.split(temps, 3, axis=3)
        r = tf.squeeze(r, axis=3) #- VGG_MEAN[2]
        g = tf.squeeze(g, axis=3) #- VGG_MEAN[1]
        b = tf.squeeze(b, axis=3) #- VGG_MEAN[0]
        # r = tf.tile(r, [])
        return (r, g, b)
