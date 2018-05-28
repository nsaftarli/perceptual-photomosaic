import tensorflow as tf 

def EntropyRegularizer(y_pred):
	return tf.reduce_mean(-1.0 * tf.reduce_sum(y_pred * tf.log(y_pred + 1e-8), axis=3))