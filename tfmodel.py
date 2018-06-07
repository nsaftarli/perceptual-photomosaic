import tensorflow as tf 
import numpy as np 
from layers import *
from VGG16 import *


VGG_MEAN = [103.939, 116.779, 123.68]
NUM_TEMPLATES = 16
PATCH_SIZE=8
IM_SHAPE = 224
norm_type='group'

class ASCIINet:

	def __init__(self, images, templates, weight_path='./weights/vgg16.npy',batch_size=1):
		self.vgg_weights = np.load(weight_path, encoding='latin1').item()
		self.net = self.build_network(images,templates,batch_size=batch_size)


	def build_network(self,input, templates, batch_size):

		with tf.name_scope('input'):
			self.input = input

		with tf.name_scope('grayscale_input'):
			self.gray_im = tf.reduce_mean(self.input,axis=-1, keep_dims=True)

		with tf.name_scope('VGG_Encoder'):
			self.encoder = VGG16(input=input)

		# with tf.name_scope('norm_enc'):
		# 	if norm_type=='instance':
		# 		self.decoder_input = InstanceNorm(self.encoder.output)
		# 	elif norm_type=='layer':
		# 		self.decoder_input = LayerNorm(self.encoder.output)
		self.decoder_input = self.encoder.output


		#################Decoder##################################################################################
		#################Block 6#################
		with tf.name_scope('block_6'):
				self.up6 = UpSampleLayer(self.decoder_input,scale_factor=2,name='up6')
				self.conv6_1,_ = ConvLayer(self.up6, name='conv6_1', ksize=3, layer_type='Decoder', out_channels=512, norm_type=norm_type)
				self.conv6_2,_ = ConvLayer(self.conv6_1, name='conv6_2', ksize=3, layer_type='Decoder', out_channels=512, norm_type=norm_type)
				self.conv6_3,_ = ConvLayer(self.conv6_2, name='conv6_3', ksize=3, layer_type='Decoder', out_channels=512, norm_type=norm_type)
				self.add6 = tf.add(self.conv6_3,self.encoder.conv5_3, name='add6')
		#################Block 7#################
		with tf.name_scope('block_7'):
				self.up7 = UpSampleLayer(self.add6,scale_factor=2,name='up7')
				self.conv7_1,_ = ConvLayer(self.up7, name='conv7_1', ksize=3, layer_type='Decoder', out_channels=512, norm_type=norm_type)
				self.conv7_2,_ = ConvLayer(self.conv7_1, name='conv7_2', ksize=3, layer_type='Decoder', out_channels=512, norm_type=norm_type)
				self.conv7_3,_ = ConvLayer(self.conv7_2, name='conv7_3', ksize=3, layer_type='Decoder', out_channels=512, norm_type=norm_type)
				self.add7 = tf.add(self.conv7_3, self.encoder.conv4_3, name='add7')
		#################Block 8#################
		with tf.name_scope('block_8'):
			self.up8 = UpSampleLayer(self.add7,scale_factor=2,name='up8')
			self.conv8_1,_ = ConvLayer(self.up8, name='conv8_1', ksize=3, layer_type='Decoder', out_channels=256, norm_type=norm_type)
			self.conv8_2,_ = ConvLayer(self.conv8_1, name='conv8_2', ksize=3, layer_type='Decoder', out_channels=256, norm_type=norm_type)
			self.conv8_3,_ = ConvLayer(self.conv8_2, name='conv8_3', ksize=3, layer_type='Decoder', out_channels=256, norm_type=norm_type)
			self.add8 = tf.add(self.conv8_3, self.encoder.conv3_3, name='add8')
		#################Block 9#################
		with tf.name_scope('block_9'):
			self.up9 = UpSampleLayer(self.add8,scale_factor=2,name='up9')
			self.conv9_1,_ = ConvLayer(self.up9, name='conv9_1', ksize=3, layer_type='Decoder', out_channels=128, norm_type=norm_type)
			self.conv9_2,_ = ConvLayer(self.conv9_1, name='conv9_2', ksize=3, layer_type='Decoder', out_channels=128, norm_type=norm_type)
			self.add9 = tf.add(self.conv9_2, self.encoder.conv2_2, name='add9')
		#################Block 10################
		with tf.name_scope('block_10'):
			self.up10 = UpSampleLayer(self.add9,scale_factor=2,name='up10')
			self.conv10_1,_ = ConvLayer(self.up10, name='conv10_1', ksize=3, layer_type='Decoder', out_channels=64, norm_type=norm_type)
			self.conv10_2, _ = ConvLayer(self.conv10_1, name='conv10_2', ksize=3, layer_type='Decoder', out_channels=64, norm_type=norm_type)
			self.add10 = tf.add(self.conv10_2, self.encoder.conv1_2, name='add10')
		##########################################################################################################


		#################Other Inputs#############################################################################
		self.temp = tf.placeholder(tf.float32,shape=[])
		self.template_tensor = TemplateLayer(templates,rgb=False)
		self.template_tensor = tf.transpose(tf.reshape(self.template_tensor,[-1, PATCH_SIZE ** 2, NUM_TEMPLATES]),perm=[0,2,1])
		##########################################################################################################

		################Softmax###################################################################################
		self.conv11, self.w = ConvLayer(self.add10, name='softmax', ksize=PATCH_SIZE, stride=PATCH_SIZE, layer_type='Softmax', 
										out_channels=NUM_TEMPLATES, patch_size=PATCH_SIZE, norm_type=norm_type)

		self.softmax = tf.nn.softmax(self.conv11 * self.temp)
		self.reshaped_softmax = tf.reshape(self.softmax,[-1, (IM_SHAPE//PATCH_SIZE) ** 2, 16])
		##########################################################################################################

		###############Output#####################################################################################
		with tf.name_scope('output_and_tile'):
			self.output = tf.matmul(self.reshaped_softmax,self.template_tensor)
			self.output = tf.reshape(tf.transpose(tf.reshape(
				self.output, [batch_size,28,28,8,8]),
				perm=[0,1,3,2,4]), [batch_size,224,224,1])
			self.output = tf.tile(self.output,[1,1,1,3])
		##########################################################################################################

		###############Loss and Regularizers######################################################################
		with tf.name_scope('VGG16_loss'):
			self.vgg2 = VGG16(input=self.output)

		self.feature_dict = {
								'conv1_1_1':self.encoder.conv1_1, 'conv1_1_2':self.vgg2.conv1_1,
								'conv1_2_1':self.encoder.conv1_2, 'conv1_2_2':self.vgg2.conv1_2,

								'conv2_1_1':self.encoder.conv2_1, 'conv2_1_2':self.vgg2.conv2_1,
								'conv2_2_1':self.encoder.conv2_2, 'conv2_2_2':self.vgg2.conv2_2,

								'conv3_3_1':self.encoder.conv3_3, 'conv3_3_2':self.vgg2.conv3_3,
								'conv4_3_1':self.encoder.conv4_3, 'conv4_3_2':self.vgg2.conv4_3,
								'conv5_3_1':self.encoder.conv5_3, 'conv5_3_2':self.vgg2.conv5_3
							}

		self.entropy = EntropyRegularizer(self.softmax) 
		self.variance = VarianceRegularizer(self.softmax)

		self.f_loss1 = tf.losses.mean_squared_error(self.feature_dict['conv1_2_1'],self.feature_dict['conv1_2_2'])
		self.loss = self.f_loss1
		##########################################################################################################

		self.build_summaries()




	def build_summaries(self):
		tf.summary.image('target', self.gray_im, max_outputs=1)
		tf.summary.image('output', self.output, max_outputs=1)

		tf.summary.scalar('entropy',self.entropy)
		tf.summary.scalar('variance',self.variance)
		tf.summary.scalar('temperature',self.temp)
		tf.summary.scalar('total_loss',self.loss)

		# for i in range(16):
		# 	tf.summary.image('templates', self.template_tensor[..., i:i+1])
		# with tf.variable_scope('conv1_1', reuse=True):
		# 	conv1_1_weights = tf.get_variable('filter')
		# 	tf.summary.scalar('conv1_1', tf.reduce_mean(conv1_1_weights))
		# with tf.variable_scope('conv6_1', reuse=True):
		# 	conv6_1_weights = tf.get_variable('weight')
		# 	tf.summary.scalar('conv6_1', tf.reduce_mean(conv6_1_weights))

		# with tf.variable_scope('softmax', reuse=True):
		# 	softmax_weights = tf.get_variable('weight')
		# 	tf.summary.histogram('weight',softmax_weights)
		# with tf.variable_scope('add10',reuse=True):
		# 	tf.summary.histogram('add10',self.add10)

		# with tf.variable_scope('conv6_2', reuse=True):
		# 	conv6_2_weights = tf.get_variable('weight')
		# 	tf.summary.scalar('conv6_2', tf.reduce_mean(conv6_2_weights))

		# with tf.variable_scope('conv6_2', reuse=True):
		# 	conv6_2_biases = tf.get_variable('bias')
		# 	tf.summary.scalar('conv6_2_b', tf.reduce_mean(conv6_2_biases))


		# for i,el in enumerate(tf.get_collection('pre-act')):
		# 	print(el)
		# 	el_mean = tf.reduce_mean(el[0,:,:,:])


		# 	if i == 26 :
		# 		tf.summary.scalar('final_encoder_pre',el_mean)




		# for i,el in enumerate(tf.get_collection('activations')):
		# 	print(el)
		# 	el_mean = tf.reduce_mean(el[0,:,:,:])

		# 	if i == 26 :
		# 		tf.summary.scalar('final_encoder_act',el_mean)


		# for i,el in enumerate(tf.get_collection('conv_biases')):
		# 	print(el)
		# 	el_mean = tf.reduce_mean(el)

		# 	if i == 26 :
		# 		tf.summary.scalar('final_encoder_bias',el_mean)

		# for i,el in enumerate(tf.get_collection('conv_weights')):
		# 	print(el)
		# 	el_mean = tf.reduce_mean(el)

		# 	if i == 26 :
		# 		tf.summary.scalar('final_encoder_weight',el_mean)


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