import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from keras import callbacks
from constants import Constants
import os
import utils


const = Constants()
experiments_dir = const.experiments_dir

class SettingLogs(callbacks.Callback):

	def __init__(self, logs={}):
		self.logs = logs 

	def on_train_begin(self, logs={}):
		batch_size = self.logs.get('batch_size')
		lr = self.logs.get('lr')
		flip = self.logs.get('flip')
		epochs = self.logs.get('epochs')
		weights = self.logs.get('weights')
		exp_file = self.logs.get('exp_file')
		f_dir = exp_file + 'settings.txt'
		os.mkdir(exp_file)
		print("AAAAAAAAAAAAAAA")
		with open(f_dir,'w') as f:
			f.write(
				'Batch Size: ' + str(batch_size) + '\n' +
				'Flip Images: ' + str(flip) + '\n' +
				'Epochs: ' + str(epochs) + '\n' + 
				'Weights: ' + str(weights) + '\n')

class ClassAccs(callbacks.Callback):

	def __init__(self, logs={}):
		self.logs = logs

	def on_epoch_end(self, epoch, logs=None):
		classes = self.logs.get('classes')
		model = self.logs.get('model_dir')
		utils.per_class_acc(model)


class LRHistory(callbacks.Callback):
	def __init__(self, logs={}):
		self.logs = logs

	def on_epoch_begin(self, epoch, logs=None):
		optimizer = self.model.optimizer
		lr = K.eval(optimizer.lr)