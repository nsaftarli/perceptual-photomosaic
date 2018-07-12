

def temperature_schedule(init_temperature,
                         temperature_cap,
                         update_freq,
                         dataset_size,
                         batch_size):
    num_iters_per_epoch = dataset_size / batch_size

    # Cap halfway through the epoch
    num_iters_per_half_epoch = num_iters_per_epoch / 2
    total_num_temperature_updates = num_iters_per_half_epoch / update_freq

    increment_rate = (temperature_cap - init_temperature) / \
        total_num_temperature_updates
    return increment_rate, total_num_temperature_updates
