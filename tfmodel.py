import tensorflow as tf 
import numpy as np 

VGG_MEAN = [103.939, 116.779, 123.68]

class ASCIINet:

	def __init__(self, weight_path='./weights/vgg16.npy'):
		self.filter_shapes = []
		self.bias_shapes = []
		self.vgg_weights = np.load(weight_path, encoding='latin1').item()
		self.vgg = self.build_model(tf.placeholder(tf.float32, shape=(None,224,224,3)))
		
		print(self.filter_shapes)
		print(self.bias_shapes)

	def build_model(self, images):
		r,g,b = tf.split(images, 3, axis=3)
		images = tf.concat([
			r - VGG_MEAN[0],
			g - VGG_MEAN[1],
			b - VGG_MEAN[2]], axis=3)

		encoder = self.build_vgg(images)
		return encoder

	def build_vgg(self,images):
		self.conv1_1 = self.ConvLayer(images, name='conv1_1')
		self.conv1_2 = self.ConvLayer(self.conv1_1, name='conv1_2')
		self.pool1 = self.PoolLayer(self.conv1_2, name='pool1')

		self.conv2_1 = self.ConvLayer(self.pool1, name='conv2_1')
		self.conv2_2 = self.ConvLayer(self.conv2_1, name='conv2_2')
		self.pool2 = self.PoolLayer(self.conv2_2, name='pool2')

		self.conv3_1 = self.ConvLayer(self.pool2, name='conv3_1')
		self.conv3_2 = self.ConvLayer(self.conv3_1, name='conv3_2')
		self.conv3_3 = self.ConvLayer(self.conv3_2, name='conv3_3')
		self.pool3 = self.PoolLayer(self.conv3_3, name='pool3')

		self.conv4_1 = self.ConvLayer(self.pool3, name='conv4_1')
		self.conv4_2 = self.ConvLayer(self.conv4_1, name='conv4_2')
		self.conv4_3 = self.ConvLayer(self.conv4_2, name='conv4_3')
		self.pool4 = self.PoolLayer(self.conv4_3, name='pool4')

		self.conv5_1 = self.ConvLayer(self.pool4, name='conv5_1')
		self.conv5_2 = self.ConvLayer(self.conv5_1, name='conv5_2')
		self.conv5_3 = self.ConvLayer(self.conv5_2, name='conv5_3')
		self.pool5 = self.PoolLayer(self.conv5_3, name='pool5')
		# return self.pool5

		self.fc6 = self.DenseLayer(self.pool5, name='fc6')
		self.relu6 = tf.nn.relu(self.fc6)
		self.fc7 = self.DenseLayer(self.relu6,'fc7')
		self.relu7 = tf.nn.relu(self.fc7)

		self.fc8 = self.DenseLayer(self.relu7,'fc8')
		self.prob = tf.nn.softmax(self.fc8,name='prob')


	def DenseLayer(self,x,name):
		with tf.variable_scope(name):
			shape = x.get_shape().as_list()
			dim = 1
			for d in shape[1:]:
				dim *= d
			x = tf.reshape(x,[-1,dim])

			weights = self.get_dense_weight(name)
			biases = self.get_biases(name)

			return tf.matmul(x, weights) + biases
	def get_dense_weight(self, name):
		return tf.constant(self.vgg_weights[name][0], name='weights')

	def ConvLayer(self,x,name,type='VGG16'):
		with tf.variable_scope(name):
			w = self.get_weights(name)
			b = self.get_biases(name)

			z = tf.nn.conv2d(x,w, [1,1,1,1],padding='SAME')
			z = z + b 

			return tf.nn.relu(z)

	def get_weights(self, name):
		self.filter_shapes.append(self.vgg_weights[name][0].shape)
		return tf.constant(self.vgg_weights[name][0], name='filter')

	def get_biases(self, name):
		self.bias_shapes.append(self.vgg_weights[name][1].shape)
		return tf.constant(self.vgg_weights[name][1], name='biases')

	def PoolLayer(self, x, name):
		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)



if __name__ == '__main__':
	m = ASCIINet()