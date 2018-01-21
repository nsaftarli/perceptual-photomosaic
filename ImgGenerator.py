import imgdata

class ImgGenerator():
	def get_batch(size=32):
		while True:
			(x_train, y_train) = imgdata.load_data(batch_size=size)
			yield (x_train, y_train)