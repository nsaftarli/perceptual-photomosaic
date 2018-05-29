import tensorflow as tf 

def DenseLayer(self,x,name):
	with tf.variable_scope(name):
		shape = x.get_shape().as_list()
		dim = 1
		for d in shape[1:]:
			dim *= d
		x = tf.reshape(x,[-1,dim])

		weights = self.get_dense_weight(name)
		biases = self.get_vgg_biases(name)

		return tf.matmul(x, weights) + biases
def get_dense_weight(self, name):
	return tf.constant(self.vgg_weights[name][0], name='weights')