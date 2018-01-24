import numpy as np

class Constants:
	def __init__(self):
		self.img_data_dir = '/home/nsaftarl/Documents/ascii-art/ASCIIArtNN/assets/rgb_in/img_celeba/'
		self.ascii_data_dir = '/home/nsaftarl/Documents/ascii-art/ASCIIArtNN/assets/ssim_imgs/'
		self.val_data_dir = '/home/nsaftarl/Documents/ascii-art/ASCIIArtNN/assets/ssim_imgs_val/'
		self.experiments_dir = '/home/nsaftarl/Documents/ascii-art/ASCIIArtNN/experiments/'
		self.char_array = np.asarray(['M','N','H','Q', '$', 'O','C', '?','7','>','!',':','-',';','.',' '])
		self.char_dict = {'M':0,'N':1,'H':2,'Q':3,'$':4,'O':5,'C':6,'?':7,'7':8,'>':9,'!':10,':':11,'-':12,';':13,'.':14,' ':15}
		self.img_rows = 224
		self.img_cols = 224
		self.text_rows = 28
		self.text_cols = 28
		self.char_count = len(self.char_array)
		self.train_set_size = 30000
		self.val_set_size = 1000