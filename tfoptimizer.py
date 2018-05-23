import tensorflow as tf 


def optimize(loss):
	lr = tf.placeholder(tf.float32,shape=[])
	opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
	return [opt,lr] 