import tensorflow as tf

#(B, H, W, C)
def rgb_to_gray(input):
    print(input.get_shape())
    x = tf.expand_dims(
        input[..., 0] * 0.299 +
        input[..., 1] * 0.587 +
        input[..., 2] * 0.114, 3)
    print(x.get_shape())
    return tf.tile(x, [1, 1, 1, 3])
