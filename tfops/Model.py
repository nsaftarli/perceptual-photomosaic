import tensorflow as tf 
import numpy as np 
import h5py
from ops import Conv2D, MaxPool

VGG_MEAN = [103.939, 116.779, 123.68]

class Model:
	def __init__(self,train_data,weight_file):
		self.weight_data = np.load(weight_file, encoding='latin1').item()
		self.model = self.build_model(train_data)



	def build_model(self,data):
		encoder = self.build_vgg(data)
		sess = tf.Session()
		# encoder = self.load_weights('weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',encoder)
		# decoder = build_decoder(encoder)
		return encoder

	def build_vgg(self,data):

		with tf.name_scope('pre-process'):
			tf.subtract(tf.constant(VGG_MEAN),data)

		with tf.variable_scope('VGG'):

			# print('NETSHAPE:' + str(data.get_shape()))

			# self.conv1_1 = Conv2D(data, k_h=3, k_w=3, k_out=64, stride=1, name='conv1_1')
			# self.conv1_2 = Conv2D(conv1_1, k_h=3, k_w=3, k_out=64, stride=1, name='conv1_2')
			# self.pool1 = MaxPool(conv1_2, k_h=2,  k_w=2, stride=1, name='pool1')

			# self.conv2_1 = Conv2D(pool1, k_h=3, k_w=3, k_out=128, stride=1, name='conv2_1')
			# self.conv2_2 = Conv2D(conv2_1, k_h=3, k_w=3, k_out=128, stride=1, name='conv2_2')
			# self.pool2 = MaxPool(conv2_2, k_h=2,  k_w=2, stride=1, name='pool2')

			# self.conv3_1 = Conv2D(pool2, k_h=3, k_w=3, k_out=256, stride=1, name='conv3_1')
			# self.conv3_2 = Conv2D(conv3_1, k_h=3, k_w=3, k_out=256, stride=1, name='conv3_2')
			# self.conv3_3 = Conv2D(conv3_2, k_h=3, k_w=3, k_out=256, stride=1, name='conv3_3')
			# self.pool3 = MaxPool(conv3_3, k_h=2,  k_w=2, stride=1, name='pool3')



			# self.conv4_1 = Conv2D(pool3, k_h=3, k_w=3, k_out=512, stride=1, name='conv4_1')
			# self.conv4_2 = Conv2D(conv4_1, k_h=3, k_w=3, k_out=512, stride=1, name='conv4_2')
			# self.conv4_3 = Conv2D(conv4_2, k_h=3, k_w=3, k_out=512, stride=1, name='conv4_3')
			# self.pool4 = MaxPool(conv4_3, k_h=2,  k_w=2, stride=1, name='pool4')

			# self.conv5_1 = Conv2D(pool4, k_h=3, k_w=3, k_out=512, stride=1, name='conv5_1')
			# self.conv5_2 = Conv2D(conv5_1, k_h=3, k_w=3, k_out=512, stride=1, name='conv5_2')
			# self.conv5_3 = Conv2D(conv5_2, k_h=3, k_w=3, k_out=512, stride=1, name='conv5_3')
			# self.pool5 = MaxPool(conv5_3, k_h=2,  k_w=2, stride=1, name='pool5')
			self.conv1_1 = self.VGGConvLayer(data, name='conv1_1')
			self.conv1_2 = self.VGGConvLayer(self.conv1_1, name='conv1_2')
			self.pool1 = self.MaxPoolLayer(self.conv1_2, k_h=2,  k_w=2, stride=1, name='pool1')

			self.conv2_1 = self.VGGConvLayer(self.pool1, name='conv2_1')
			self.conv2_2 = self.VGGConvLayer(self.conv2_1, name='conv2_2')
			self.pool2 = self.MaxPoolLayer(self.conv2_2, k_h=2,  k_w=2, stride=1, name='pool2')

			self.conv3_1 = self.VGGConvLayer(self.pool2, name='conv3_1')
			self.conv3_2 = self.VGGConvLayer(self.conv3_1, name='conv3_2')
			self.conv3_3 = self.VGGConvLayer(self.conv3_2, name='conv3_3')
			self.pool3 = self.MaxPoolLayer(self.conv3_3, k_h=2,  k_w=2, stride=1, name='pool3')



			self.conv4_1 = self.VGGConvLayer(self.pool3, name='conv4_1')
			self.conv4_2 = self.VGGConvLayer(self.conv4_1, name='conv4_2')
			self.conv4_3 = self.VGGConvLayer(self.conv4_2, name='conv4_3')
			self.pool4 = self.MaxPoolLayer(self.conv4_3, k_h=2,  k_w=2, stride=1, name='pool4')

			self.conv5_1 = self.VGGConvLayer(self.pool4, name='conv5_1')
			self.conv5_2 = self.VGGConvLayer(self.conv5_1, name='conv5_2')
			self.conv5_3 = self.VGGConvLayer(self.conv5_2, name='conv5_3')
			self.pool5 = self.MaxPoolLayer(self.conv5_3, k_h=2,  k_w=2, stride=1, name='pool5')
			self.net = self.pool5
			return self.net

	def load_weights(self,wfile,encoder):
		# weights = np.load(wfile)
		weights = h5py.File(wfile)
		# weights = np.load(wfile, encoding='bytes')
		print('AAAAAAAAAAAAAAAAAAAAAAA')
		# print(weights.shape)
		for key in weights.keys():
			print(key)
			group = weights[key]
			for key2 in group.keys():
				print(group[key2].value.shape)
			break
		print(tf.GraphKeys.WEIGHTS)
		# keys = sorted(weights.keys())
		# for i, j in enumerate(keys):
			# sess.run(self.parameters[i].assign(weights[j]))
			# continue

	def build_decoder(self, x):
		net = Conv2D(x, k_h=3, k_w=3, k_out=512, stride=1, name='deconv5_1')
		net = Conv2D(x, k_h=3, k_w=3, k_out=512, stride=1, name='deconv5_2')
		net = Conv2D(x, k_h=3, k_w=3, k_out=512, stride=1, name='deconv5_3')

	def get_vgg_filter(self, name):
		init = tf.constant_initializer(value=self.weight_data[name][0], dtype=tf.float32)
		shape = self.weight_data[name][0].shape
		print(shape)
		with tf.variable_scope("weights",reuse=tf.AUTO_REUSE):
			filt = tf.get_variable(name="filter",initializer=init,shape=shape)
			return filt 

	def get_vgg_bias(self, name):
		init = tf.constant_initializer(value=self.weight_data[name][1], dtype=tf.float32)
		shape = self.weight_data[name][0].shape
		print(shape)
		with tf.variable_scope("biases", reuse=tf.AUTO_REUSE):
			bias = tf.get_variable(name="bias",initializer=init,shape=shape)
			return bias

	def VGGConvLayer(self,x,name):
		w = self.get_vgg_filter(name)
		b = self.get_vgg_bias(name)
		print(w)
		print(b)
		print('AAAAAAAAAAAAAAA')
		z = tf.nn.conv2d(x,w,[1,1,1,1],padding='SAME')

		tf.add_to_collection('WEIGHTS',w)

		return tf.nn.relu(z)

	def MaxPoolLayer(self,x,name):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)



if __name__ == '__main__':
	x = tf.placeholder(tf.float32, [None, 224, 224, 3])
	m = Model(x,'../weights/vgg16.npy')
	# net = m.build_model(x)