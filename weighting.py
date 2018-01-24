import predict
import numpy as np
from constants import Constants 


const = Constants()
text_rows = const.text_rows
text_cols = const.text_cols
train_set_size = const.train_set_size

def median_freq_balancing():
	total_counts, appearances = predict.char_counts(size=train_set_size, textrows=text_rows, textcols=text_cols)
	total_counts = np.asarray(total_counts, dtype='float32')
	appearances = np.asarray(appearances, dtype='float32')

	sum_total = np.sum(total_counts)
	sum_appearances = np.sum(appearances)
	freq_counts = total_counts / appearances
	median_freqs = np.median(total_counts) / freq_counts
	median_freqs /= np.sum(median_freqs)
	#Force higher weighting for M by hand
	# median_freqs[0] * 3


	print("TOTAL COUNTS: ")
	print(total_counts)
	print("FREQUENCY COUNTS: ")
	print(freq_counts)
	print("SUM OF CHARACTERS: ")
	print(sum_total)
	print("SUM OF CHARACTER APPEARANCES")
	print(sum_appearances)

	return median_freqs

def plain_weighting():
	total_counts, _ = predict.char_counts(size=train_set_size,textrows=28,textcols=28)
	total_counts = np.asarray(total_counts,dtype='float32')

	sum_total = np.sum(total_counts)
	total_counts /= sum_total
	weights = 1. - total_counts

	print("NORMALIZED: ")
	print(total_counts)
	print("WEIGHTED: ")
	print(total_counts)

	return weights