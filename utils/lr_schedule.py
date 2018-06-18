def lr_schedule(lr, it):

    # return lr
    if it < 10000:
        return lr
    elif it < 30000:
        return lr/2
    elif it < 50000:
        return lr/4
    elif it < 80000:
        return lr/8
    else:
        return lr/16
