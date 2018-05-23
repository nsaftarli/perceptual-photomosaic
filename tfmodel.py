import tensorflow as tf 
import numpy as np 
from math import ceil

VGG_MEAN = [103.939, 116.779, 123.68]
NUM_TEMPLATES = 16
PATCH_SIZE=8
IM_SHAPE = 224

class ASCIINet:

	def __init__(self, images, templates, weight_path='./weights/vgg16.npy',batch_size=32):
		self.vgg_weights = np.load(weight_path, encoding='latin1').item()
		print(images)
		print(templates)

		self.vgg = self.build_network(images,templates,batch_size=batch_size)


	def build_network(self,input, templates, batch_size):
		self.input = input
		self.gray_im = tf.reduce_mean(self.input,axis=-1, keep_dims=True)

		r,g,b = tf.split(self.input, 3, axis=3)
		self.vgg_input = tf.concat([
			r - VGG_MEAN[0],
			g - VGG_MEAN[1],
			b - VGG_MEAN[2]], axis=3) 


		# self.max_val = tf.reduce_max(self.gray_im,axis=-1)
		# self.min_val = tf.reduce_min(self.gray_im, axis=-1)

		#Encoder (VGG16 by default)
		self.conv1_1 = self.ConvLayer(self.vgg_input, name='conv1_1', trainable=False)
		self.conv1_2 = self.ConvLayer(self.conv1_1, name='conv1_2', trainable=False)
		self.pool1 = self.PoolLayer(self.conv1_2, name='pool1', trainable=False)

		self.conv2_1 = self.ConvLayer(self.pool1, name='conv2_1', trainable=False)
		self.conv2_2 = self.ConvLayer(self.conv2_1, name='conv2_2', trainable=False)
		self.pool2 = self.PoolLayer(self.conv2_2, name='pool2', trainable=False)

		self.conv3_1 = self.ConvLayer(self.pool2, name='conv3_1', trainable=False)
		self.conv3_2 = self.ConvLayer(self.conv3_1, name='conv3_2', trainable=False)
		self.conv3_3 = self.ConvLayer(self.conv3_2, name='conv3_3', trainable=False)
		self.pool3 = self.PoolLayer(self.conv3_3, name='pool3', trainable=False)

		self.conv4_1 = self.ConvLayer(self.pool3, name='conv4_1', trainable=False)
		self.conv4_2 = self.ConvLayer(self.conv4_1, name='conv4_2', trainable=False)
		self.conv4_3 = self.ConvLayer(self.conv4_2, name='conv4_3', trainable=False)
		self.pool4 = self.PoolLayer(self.conv4_3, name='pool4', trainable=False)

		self.conv5_1 = self.ConvLayer(self.pool4, name='conv5_1', trainable=False)
		self.conv5_2 = self.ConvLayer(self.conv5_1, name='conv5_2', trainable=False)
		self.conv5_3 = self.ConvLayer(self.conv5_2, name='conv5_3', trainable=False)
		self.pool5 = self.PoolLayer(self.conv5_3, name='pool5', trainable=False)

		#Decoder
		self.up6 = self.UpSampleLayer(self.pool5,scale_factor=2,name='up6')
		self.conv6_1 = self.ConvLayer(self.up6, name='conv6_1', layer_type='Decoder', out_channels=512)
		self.conv6_2 = self.ConvLayer(self.conv6_1, name='conv6_2', layer_type='Decoder', out_channels=512)
		self.conv6_3 = self.ConvLayer(self.conv6_2, name='conv6_3', layer_type='Decoder', out_channels=512)
		self.add6 = tf.add(self.conv6_3,self.conv5_3, name='add6')

		self.up7 = self.UpSampleLayer(self.add6,scale_factor=2,name='up7')
		self.conv7_1 = self.ConvLayer(self.up7, name='conv7_1', layer_type='Decoder', out_channels=512)
		self.conv7_2 = self.ConvLayer(self.conv7_1, name='conv7_2', layer_type='Decoder', out_channels=512)
		self.conv7_3 = self.ConvLayer(self.conv7_2, name='conv7_3', layer_type='Decoder', out_channels=512)
		self.add7 = tf.add(self.conv7_3, self.conv4_3, name='add7')

		self.up8 = self.UpSampleLayer(self.add7,scale_factor=2,name='up8')
		self.conv8_1 = self.ConvLayer(self.up8, name='conv8_1', layer_type='Decoder', out_channels=256)
		self.conv8_2 = self.ConvLayer(self.conv8_1, name='conv8_2', layer_type='Decoder', out_channels=256)
		self.conv8_3 = self.ConvLayer(self.conv8_2, name='conv8_3', layer_type='Decoder', out_channels=256)
		self.add8 = tf.add(self.conv8_3, self.conv3_3, name='add8')

		self.up9 = self.UpSampleLayer(self.add8,scale_factor=2,name='up9')
		self.conv9_1 = self.ConvLayer(self.up9, name='conv9_1', layer_type='Decoder', out_channels=128)
		self.conv9_2 = self.ConvLayer(self.conv9_1, name='conv9_2', layer_type='Decoder', out_channels=128)
		self.add9 = tf.add(self.conv9_2, self.conv2_2, name='add9')

		self.up10 = self.UpSampleLayer(self.add9,scale_factor=2,name='up10')
		self.conv10_1 = self.ConvLayer(self.up10, name='conv10_1', layer_type='Decoder', out_channels=64)
		self.conv10_2 = self.ConvLayer(self.conv10_1, name='conv10_2', layer_type='Decoder', out_channels=64)
		self.add10 = tf.add(self.conv10_2, self.conv1_2, name='add10')

		self.softmax = self.ConvLayer(self.add10, name='softmax', layer_type='Softmax', out_channels=NUM_TEMPLATES)

		self.flat_softmax = tf.reshape(self.softmax,
			shape=tf.convert_to_tensor([1, 1, NUM_TEMPLATES, batch_size * self.softmax.get_shape()[1].value * self.softmax.get_shape()[2].value], dtype=tf.int32),
			name='flat_softmax')

		self.template_tensor = self.TemplateLayer(templates,rgb=False)
		print(self.template_tensor)

		# self.conv11 = tf.nn.conv2d(self.template_tensor,self.flat_softmax,strides=[1,1,1,1],padding='SAME',name='conv11')
		self.conv11 = tf.nn.conv2d(self.flat_softmax,self.template_tensor,strides=[1,1,1,1],padding='SAME',name='conv11')

		self.output = tf.reshape(self.conv11,shape=[batch_size, IM_SHAPE, IM_SHAPE, 1], name='output')

		# self.conv11 = tf.reshape(
		# 	tf.nn.conv2d(self.template_tensor,self.flat_softmax,strides=[1,1,1,1],padding='SAME'),
		# 	shape=[batch_size, IM_SHAPE, IM_SHAPE], name='conv11')

		# self.tloss = self.ToyLoss(self.gray_im,self.conv11)
		self.build_summaries()


		self.print_architecture()


	def build_summaries(self):
		tf.summary.image('target', self.gray_im, max_outputs=1)
		tf.summary.image('output', self.output, max_outputs=1)
		for i in range(16):
			tf.summary.image('templates', self.template_tensor[..., i:i+1])
		with tf.variable_scope('conv1_1', reuse=True):
			conv1_1_weights = tf.get_variable('filter')
			tf.summary.scalar('conv1_1', tf.reduce_mean(conv1_1_weights))
		with tf.variable_scope('conv6_1', reuse=True):
			conv6_1_weights = tf.get_variable('weight')
			tf.summary.scalar('conv6_1', tf.reduce_mean(conv6_1_weights))
		self.summaries = tf.summary.merge_all()



	def TemplateLayer(self,templates, rgb=False):
		return tf.constant(value=templates, dtype=tf.float32, shape=templates.shape, name='templates')


	def ConvLayer(self,x,name,layer_type='VGG16',out_channels=None, trainable=True):
		if layer_type == 'VGG16':
			with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
				w = self.get_vgg_weights(name, trainable=trainable)
				b = self.get_vgg_biases(name, trainable=trainable)
				z = tf.nn.conv2d(x,w, [1,1,1,1],padding='SAME')
				z = z + b 
				return tf.nn.relu(z)
		elif layer_type == 'Softmax':
			with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
				in_channels = x.get_shape()[3].value
				shape_in = [PATCH_SIZE, PATCH_SIZE, in_channels, NUM_TEMPLATES]
				w = tf.get_variable('weight', initializer=tf.truncated_normal(shape_in), trainable=trainable)
				b = tf.get_variable('bias', initializer=tf.constant(0.0,dtype=tf.float32), trainable=trainable)
				return tf.nn.softmax(tf.nn.conv2d(
					x,w,strides=[1, PATCH_SIZE, PATCH_SIZE, 1], padding='SAME') + b)
		else:
			with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
				in_channels = x.get_shape()[3].value
				shape_in = [3,3, in_channels, out_channels]
				w = tf.get_variable('weight',initializer=tf.truncated_normal(shape_in), trainable=trainable)
				b = tf.get_variable('bias',initializer=tf.constant(0.0,dtype=tf.float32), trainable=trainable)
				return tf.nn.relu(tf.nn.conv2d(
					x,w,strides=[1,1,1,1],padding='SAME')+b)

	def PoolLayer(self, x, name, trainable=True):
		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)


	def UpSampleLayer(self,x,scale_factor,name):
		with tf.variable_scope(name):
			in_shape = x.get_shape()
			upsampled = tf.image.resize_images(x,
				size=[in_shape[1].value * scale_factor, in_shape[2].value * scale_factor])
			return upsampled

	def get_vgg_weights(self, name, trainable):
		return tf.get_variable(initializer=self.vgg_weights[name][0], name='filter', trainable=trainable)


	def get_vgg_biases(self, name, trainable):
		return tf.get_variable(initializer=self.vgg_weights[name][1], name='biases', trainable=trainable)


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

	def print_architecture(self):
		print(self.conv1_1.get_shape())
		print(self.conv1_2.get_shape())
		# print(self.pool1.get_shape())
		print(self.conv2_1.get_shape())
		print(self.conv2_2.get_shape())
		# print(self.pool2.get_shape())
		print(self.conv3_1.get_shape())
		print(self.conv3_2.get_shape())
		print(self.conv3_3.get_shape())
		# print(self.pool3.get_shape())
		print(self.conv4_1.get_shape())
		print(self.conv4_2.get_shape())
		print(self.conv4_3.get_shape())
		# print(self.pool4.get_shape())
		print(self.conv5_1.get_shape())
		print(self.conv5_2.get_shape())
		print(self.conv5_3.get_shape())
		# print(self.pool5.get_shape())

		print('################################')
		# print(self.up6.get_shape())
		print(self.conv6_1.get_shape())
		print(self.conv6_2.get_shape())
		print(self.conv6_3.get_shape())
		print(self.conv7_3.get_shape())
		print(self.conv7_2.get_shape())
		print(self.conv7_3.get_shape())
		print(self.conv8_1.get_shape())
		print(self.conv8_2.get_shape())
		print(self.conv8_3.get_shape())
		print(self.conv9_1.get_shape())
		print(self.conv9_2.get_shape())
		# print(self.conv9_3.get_shape())
		print(self.conv10_1.get_shape())
		print(self.conv10_2.get_shape())
		print(self.softmax.get_shape())
		# print(self.prediction.get_shape())
		print(self.flat_softmax.get_shape())
		print(self.template_tensor.get_shape())
		print(self.conv11.get_shape())
		# print(self.tloss.get_shape())
		print(tf.trainable_variables())

if __name__ == '__main__':
	m = ASCIINet()