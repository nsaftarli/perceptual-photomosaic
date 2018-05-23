import tensorflow as tf

class Loss:
	def __init__(self,y,y_pred):
		self.loss = self.MSE(y,y_pred)
		#Entropy loss

		#Perceptual loss

	def entropy_loss(self,y_pred):
		num_templates = y_pred.get_shape()[-1].value
		tf.contrib.bayesflow.entropy.entropy_shannon(y_pred)

	def MSE(self,y,y_pred):
		return tf.losses.mean_squared_error(y,y_pred)
