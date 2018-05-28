import tensorflow as tf 

def batch_norm_layer(x):
	y = tf.layers.batch_normalization(x, training=True)
	tf.add_to_collection('batch_layers',y)
	return y