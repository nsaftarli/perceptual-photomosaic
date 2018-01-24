import tensorflow as tf 
import numpy as np
from keras import callbacks

class AccuracyHistory(callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.accuracy = []

	def on_batch_end(self, batch, logs={}):
		self.accuracy.append(logs.get('accuracy'))