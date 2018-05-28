import tensorflow as tf
import numpy as np 

class LossLayer:
	def __init__(self,y,y_pred):
		self.loss = self.MSE(y,y_pred)
		tf.summary.scalar('loss',self.loss)
		self.summaries = tf.summary.merge_all()
		#Entropy loss

		#Perceptual loss

	def entropy_loss(self,y_pred):
		return tf.reduce_mean(-1.0 * tf.reduce_sum(y_pred * tf.log(y_pred + 1e-8), axis=3))

	# def MSE(self,y,y_pred):
	# 	return tf.losses.mean_squared_error(y,y_pred)

	def MSE(self,y, y_pred):
	    # 1x128x128x1
	    e_1 = self.downsample(expected)
	    p_1 = self.downsample(predicted)
	    # 1x64x64x1
	    e_2 = self.downsample(e_1)
	    p_2 = self.downsample(p_1)
	    # 1x32x32x1
	    e_3 = self.downsample(e_2)
	    p_3 = self.downsample(p_2)
	    return \
	        tf.reduce_mean(tf.square(e_1 - p_1)) + \
	        tf.reduce_mean(tf.square(e_2 - p_2)) + \
	        tf.reduce_mean(tf.square(e_3 - p_3))

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
