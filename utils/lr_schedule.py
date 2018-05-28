def lr_schedule(lr, it):
	if it < 4000:
		return lr
	elif it < 10000:
		return lr/4
	elif it < 20000:
		return lr/8
	else:
		return lr/16