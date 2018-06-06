import tensorflow as tf
import numpy as np 

from EntropyRegularizer import *
from VarianceRegularizer import *

class LossLayer:
	def __init__(self,y,y_pred):
		# self.entropy = 5e-3 * self.entropy_loss(y_pred)
		self.mse = (self.MSE2(y,y_pred))  #+ (3e-5*self.variance_loss(y_pred))

		# tf.summary.scalar('entropy',self.entropy)
		tf.summary.scalar('loss',self.mse)
		self.summaries = tf.summary.merge_all()

	def entropy_loss(self,y_pred):
		return tf.reduce_mean(-1.0 * tf.reduce_sum(y_pred * tf.log(y_pred + 1e-10), axis=3))

	def MSE2(self,y,y_pred):
		# tf.summary.image('output',y)
		# tf.summary.image('predicted',y_pred)
		return tf.losses.mean_squared_error(y,y_pred)
		# return tf.reduce_mean(tf.abs(y-y_pred))*1e2

	def MSE(self,y, y_pred):
		# 1x128x128x1
		with tf.name_scope('downsample1'):
			self.e_1 = self.laplacian(y)
			self.p_1 = self.laplacian(y_pred)
		# 1x64x64x1
		with tf.name_scope('downsample2'):
			self.e_2 = self.laplacian(self.e_1)
			self.p_2 = self.laplacian(self.p_1)
		# 1x32x32x1
		with tf.name_scope('downsample3'):
			self.e_3 = self.laplacian(self.e_2)
			self.p_3 = self.laplacian(self.p_2)

		# with tf.name_scope('laplace1'):
		# 	self.l1 = self.laplacian(y)

		# tf.summary.image('laplace',self.l1)

		tf.summary.image('output',y)
		tf.summary.image('predicted',y_pred)
		tf.summary.image('im_down1',self.e_1)
		tf.summary.image('p_down1',self.p_1)
		tf.summary.image('im_down2',self.e_2)
		tf.summary.image('p_down2',self.p_2)
		tf.summary.image('im_down3',self.e_3)
		tf.summary.image('p_down3',self.p_3)

		with tf.name_scope('ms_mse'):
			totalLoss = (tf.reduce_mean(tf.square(self.e_1 - self.p_1)) + \
						tf.reduce_mean(tf.square(self.e_2 - self.p_2)) + \
						tf.reduce_mean(tf.square(self.e_3 - self.p_3)))
		return totalLoss


	def variance_loss(self,y_pred):
		bins = tf.reshape(np.linspace(1,16,num=16).astype('float32'),[1,1,1,16])
		mean = tf.reduce_sum(bins * y_pred, axis=3)
		mean_2 = tf.reduce_sum(bins ** 2 * y_pred,axis=3)
		variance = mean_2 - mean ** 2
		return tf.reduce_mean(variance)


	def gauss2d_kernel(self,shape=(3, 3), sigma=0.5):
	    """
	    2D gaussian mask - should give the same result as MATLAB's
	    fspecial('gaussian',[shape],[sigma])
	    """
	    m, n = [(ss-1.)/2. for ss in shape]
	    y, x = np.ogrid[-m:m+1, -n:n+1]
	    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
	    h[h < np.finfo(h.dtype).eps*h.max()] = 0
	    sumh = h.sum()
	    if sumh != 0:
	        h /= sumh
	    return h


	def downsample(self,input):
	    w = tf.reshape(tf.constant(self.gauss2d_kernel(), dtype=tf.float32),
	                   [3, 3, 1, 1]) 
	    return tf.nn.conv2d(input, w, strides=[1, 2, 2, 1], padding='SAME')

	def laplacian(self,input):
		w = tf.reshape(tf.constant(self.gauss2d_kernel(), dtype=tf.float32),
						[3, 3, 1, 1])
		#Downsampled input
		x = tf.nn.conv2d(self.downsample(input),w,strides=[1,1,1,1], padding='SAME')
		# tf.summary.image('blurred_img',x)
		# tf.summary.image('regular_img',self.downsample(input))
		y = self.downsample(input) - x
		# tf.summary.image('laplace',y)
		return y
