import tensorflow as tf 
import numpy as np 
from layers import *


VGG_MEAN = [103.939, 116.779, 123.68]
NUM_TEMPLATES = 16
PATCH_SIZE=8
IM_SHAPE = 224

class ASCIINet:

	def __init__(self, images, templates, weight_path='./weights/vgg16.npy',batch_size=1):
		self.vgg_weights = np.load(weight_path, encoding='latin1').item()
		self.vgg = self.build_network(images,templates,batch_size=batch_size)


	def build_network(self,input, templates, batch_size):
		
		with tf.name_scope('input'):
			self.input = input

		with tf.name_scope('grayscale_input'):
			self.gray_im = tf.reduce_mean(self.input,axis=-1, keep_dims=True)

		with tf.name_scope('mean_subtract'):
			r,g,b = tf.split(self.input, 3, axis=3)
			self.vgg_input = tf.concat([
				b - VGG_MEAN[0],
				g - VGG_MEAN[1],
				r - VGG_MEAN[2]], axis=3) 

		#Encoder (VGG16 by default)
		self.conv1_1,_ = ConvLayer(self.vgg_input, name='conv1_1', trainable=False)
		self.conv1_2,_ = ConvLayer(self.conv1_1, name='conv1_2', trainable=False)
		self.pool1 = PoolLayer(self.conv1_2, name='pool1', trainable=False)

		self.conv2_1,_ = ConvLayer(self.pool1, name='conv2_1', trainable=False)
		self.conv2_2,_ = ConvLayer(self.conv2_1, name='conv2_2', trainable=False)
		self.pool2 = PoolLayer(self.conv2_2, name='pool2', trainable=False)


		self.conv3_1,_ = ConvLayer(self.pool2, name='conv3_1', trainable=False)
		self.conv3_2,_ = ConvLayer(self.conv3_1, name='conv3_2', trainable=False)
		self.conv3_3,_ = ConvLayer(self.conv3_2, name='conv3_3', trainable=False)
		self.pool3 = PoolLayer(self.conv3_3, name='pool3', trainable=False)


		self.conv4_1,_ = ConvLayer(self.pool3, name='conv4_1', trainable=False)
		self.conv4_2,_ = ConvLayer(self.conv4_1, name='conv4_2', trainable=False)
		self.conv4_3,_ = ConvLayer(self.conv4_2, name='conv4_3', trainable=False)
		self.pool4 = PoolLayer(self.conv4_3, name='pool4', trainable=False)


		self.conv5_1,_ = ConvLayer(self.pool4, name='conv5_1', trainable=False)
		self.conv5_2,_ = ConvLayer(self.conv5_1, name='conv5_2', trainable=False)
		self.conv5_3,self.w2 = ConvLayer(self.conv5_2, name='conv5_3', trainable=False)
		self.pool5 = PoolLayer(self.conv5_3, name='pool5', trainable=False)

		#Decoder
		self.up6 = UpSampleLayer(self.pool5,scale_factor=2,name='up6')
		self.conv6_1,_ = ConvLayer(self.up6, name='conv6_1', layer_type='Decoder', out_channels=512)
		self.conv6_2,_ = ConvLayer(self.conv6_1, name='conv6_2', layer_type='Decoder', out_channels=512)
		self.conv6_3,_ = ConvLayer(self.conv6_2, name='conv6_3', layer_type='Decoder', out_channels=512)
		self.add6 = tf.add(self.conv6_3,self.conv5_3, name='add6')
		self.batch6 = batch_norm_layer(self.add6)


		self.up7 = UpSampleLayer(self.batch6,scale_factor=2,name='up7')
		self.conv7_1,_ = ConvLayer(self.up7, name='conv7_1', layer_type='Decoder', out_channels=512)
		self.conv7_2,_ = ConvLayer(self.conv7_1, name='conv7_2', layer_type='Decoder', out_channels=512)
		self.conv7_3,_ = ConvLayer(self.conv7_2, name='conv7_3', layer_type='Decoder', out_channels=512)
		self.add7 = tf.add(self.conv7_3, self.conv4_3, name='add7')
		self.batch7 = batch_norm_layer(self.add7)


		self.up8 = UpSampleLayer(self.batch7,scale_factor=2,name='up8')
		self.conv8_1,_ = ConvLayer(self.up8, name='conv8_1', layer_type='Decoder', out_channels=256)
		self.conv8_2,_ = ConvLayer(self.conv8_1, name='conv8_2', layer_type='Decoder', out_channels=256)
		self.conv8_3,_ = ConvLayer(self.conv8_2, name='conv8_3', layer_type='Decoder', out_channels=256)
		self.add8 = tf.add(self.conv8_3, self.conv3_3, name='add8')
		self.batch8 = batch_norm_layer(self.add8)


		self.up9 = UpSampleLayer(self.batch8,scale_factor=2,name='up9')
		self.conv9_1,_ = ConvLayer(self.up9, name='conv9_1', layer_type='Decoder', out_channels=128)
		self.conv9_2,_ = ConvLayer(self.conv9_1, name='conv9_2', layer_type='Decoder', out_channels=128)
		self.add9 = tf.add(self.conv9_2, self.conv2_2, name='add9')
		self.batch9 = batch_norm_layer(self.add9)


		self.up10 = UpSampleLayer(self.batch9,scale_factor=2,name='up10')
		self.conv10_1,_ = ConvLayer(self.up10, name='conv10_1', layer_type='Decoder', out_channels=64)
		self.conv10_2, _ = ConvLayer(self.conv10_1, name='conv10_2', layer_type='Decoder', out_channels=64)
		self.add10 = tf.add(self.conv10_2, self.conv1_2, name='add10')
		self.batch10 = batch_norm_layer(self.add10)
		
		self.conv11, self.w = ConvLayer(self.batch10, name='softmax', layer_type='Softmax', out_channels=NUM_TEMPLATES, patch_size=PATCH_SIZE)

		# with tf.variable_scope('abc',reuse=tf.AUTO_REUSE):
		# 	self.w2 = tf.get_variable('w',initializer=tf.constant(5.0,dtype=tf.float32))

		# self.temp = tf.placeholder(tf.float32,shape=[])
		self.template_tensor = TemplateLayer(templates,rgb=False)
		# self.template_tensor = tf.constant(templates,dtype=tf.float32)
		self.softmax = (tf.nn.softmax(self.conv11)) #* self.temp

		with tf.name_scope('reshaped_templates'):
			reshaped_templates = tf.transpose(tf.reshape(self.template_tensor,[-1, PATCH_SIZE ** 2, NUM_TEMPLATES]),perm=[0,2,1])

		with tf.name_scope('reshaped_softmax'):
			reshaped_softmax = tf.reshape(self.softmax,[-1, (IM_SHAPE//PATCH_SIZE) ** 2, 16])

		with tf.name_scope('output'):
			self.output = tf.matmul(reshaped_softmax,reshaped_templates)

		with tf.name_scope('reshaped_output'):
			self.reshaped_output = tf.reshape(tf.transpose(tf.reshape(
				self.output, [batch_size,28,28,8,8]),
				perm=[0,1,3,2,4]), [batch_size,224,224,1])

		print(self.gray_im.get_shape())
		print(self.reshaped_output.get_shape())

		##################Regularizers#####################################
		# self.entropy = EntropyRegularizer(self.softmax) * 5e4
		# self.variance = VarianceRegularizer(self.softmax) * 3e4
		# self.loss = LossLayer(self.gray_im,self.reshaped_output) + self.entropy + self.variance
		# self.loss = LossLayer(self.gray_im,self.reshaped_output)
		###########################################################

		# self.build_summaries()




	def build_summaries(self):
		tf.summary.image('target', self.gray_im, max_outputs=1)
		tf.summary.image('output', self.reshaped_output, max_outputs=1)

		for i in range(16):
			tf.summary.image('templates', self.template_tensor[..., i:i+1])
		with tf.variable_scope('conv1_1', reuse=True):
			conv1_1_weights = tf.get_variable('filter')
			tf.summary.scalar('conv1_1', tf.reduce_mean(conv1_1_weights))
		with tf.variable_scope('conv6_1', reuse=True):
			conv6_1_weights = tf.get_variable('weight')
			tf.summary.scalar('conv6_1', tf.reduce_mean(conv6_1_weights))

		with tf.variable_scope('softmax', reuse=True):
			softmax_weights = tf.get_variable('weight')
			tf.summary.histogram('weight',softmax_weights)
		with tf.variable_scope('add10',reuse=True):
			tf.summary.histogram('add10',self.add10)

		with tf.variable_scope('conv6_2', reuse=True):
			conv6_2_weights = tf.get_variable('weight')
			tf.summary.scalar('conv6_2', tf.reduce_mean(conv6_2_weights))

		with tf.variable_scope('conv6_2', reuse=True):
			conv6_2_biases = tf.get_variable('bias')
			tf.summary.scalar('conv6_2_b', tf.reduce_mean(conv6_2_biases))


		for i,el in enumerate(tf.get_collection('pre-act')):
			print(el)
			el_mean = tf.reduce_mean(el[0,:,:,:])


			if i == 26 :
				tf.summary.scalar('final_encoder_pre',el_mean)




		for i,el in enumerate(tf.get_collection('activations')):
			print(el)
			el_mean = tf.reduce_mean(el[0,:,:,:])

			if i == 26 :
				tf.summary.scalar('final_encoder_act',el_mean)


		for i,el in enumerate(tf.get_collection('conv_biases')):
			print(el)
			el_mean = tf.reduce_mean(el)

			if i == 26 :
				tf.summary.scalar('final_encoder_bias',el_mean)

		for i,el in enumerate(tf.get_collection('conv_weights')):
			print(el)
			el_mean = tf.reduce_mean(el)

			if i == 26 :
				tf.summary.scalar('final_encoder_weight',el_mean)


		# tf.summary.scalar('entropy',self.entropy)
		# tf.summary.scalar('variance',self.variance)
		# tf.summary.scalar('temperature',self.temp)
		# tf.summary.scalar('loss',self.loss)

		self.summaries = tf.summary.merge_all()




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