import os
import tensorflow as tf


# From https://github.com/ryersonvisionlab/two-stream-dyntex-synth/blob/master/src/utilities.py#L50
def check_snapshots(run_id):
    snapshots_folder = 'snapshots/' + run_id + '/'
    logs_folder = 'logs/' + run_id + '/'

    checkpoint = tf.train.latest_checkpoint(snapshots_folder)

    resume = False
    start_iteration = 1

    if checkpoint:
        choice = ''
        while choice != 'y' and choice != 'n':
            print('Snapshot file detected (' + checkpoint +
                  ') would you like to resume? (y/n)')
            choice = input().lower()

            if choice == 'y':
                resume = checkpoint
                start_iteration = int(checkpoint.split(snapshots_folder)
                                      [1][5:-5])
                print('resuming from iteration ' + str(start_iteration))
            else:
                print('removing old snapshots and logs, training from scratch')
                resume = False
                for file in os.listdir(snapshots_folder):
                    if file == '.gitignore':
                        continue
                    os.remove(snapshots_folder + file)
                for file in os.listdir(logs_folder):
                    if file == '.gitignore':
                        continue
                    os.remove(logs_folder + file)
    else:
        print("No snapshots found, training from scratch")

    return resume, start_iteration
