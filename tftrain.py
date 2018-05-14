import tensorflow as tf
import numpy as np 
import sklearn
import imdata
from constants import Constants 

from tfops import Model


'''Data constants'''
const = Constants()
img_data_dir = const.img_data_dir
ascii_data_dir = const.ascii_data_dir
val_data_dir = const.val_data_dir
char_array = const.char_array
char_dict = const.char_dict
img_rows = const.img_rows
img_cols = const.img_cols
text_rows = const.text_rows
text_cols = const.text_cols
dims = const.char_count
experiments_dir = const.experiments_dir


dataset = tf.data.Dataset.from_generator(imdata.load_data, (tf.float32,tf.float32))
it = dataset.make_one_shot_iterator()
next_batch = it.get_next()



with tf.Session() as sess:
	for i in range(1):
		s,y = sess.run(next_batch)
		s = tf.convert_to_tensor(s,tf.float32)
		y = tf.convert_to_tensor(y,tf.float32)
		m = Model.Model(s)
		


