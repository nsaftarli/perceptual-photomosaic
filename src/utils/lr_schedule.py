def lr_schedule(lr, it):
    if it < 8000:
        return lr
    else:
        return lr / 1.1