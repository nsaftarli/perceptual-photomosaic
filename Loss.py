import tensorflow as tf
import numpy as np 

from layers import *

class Loss:
	def __init__(self,y,y_pred):
		self.loss = self.MSE(y,y_pred)
		#Entropy loss

		#Perceptual loss

	def entropy_loss(self,y_pred):
		return tf.reduce_mean(-1.0 * tf.reduce_sum(y_pred * tf.log(y_pred + 1e-8), axis=3))

	def MSE(self,y,y_pred):
		return tf.losses.mean_squared_error(y,y_pred)

	def variance_loss(self,y_pred):
		bins = tf.reshape(np.linspace(1,16,num=16).astype('float32'),[1,1,1,16])
		mean = tf.reduce_sum(bins * y_pred, axis=3)
		mean_2 = tf.reduce_sum(bins ** 2 * y_pred,axis=3)
		variance = mean_2 - mean ** 2
		return tf.reduce_mean(variance)
