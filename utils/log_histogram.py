import tensorflow as tf
import numpy as np


def log_histogram(writer, tag, values, step, bins=62):
    values = np.array(values)
    bin_edges = np.linspace(0, 61, num=bins)

    hist = tf.HistogramProto()
    hist.min = float(0)
    hist.max = float(bins-1)
    hist.num = int(bins)
    hist.sum = float(1)
    hist.sum_squares = float(1)
    for edge in bin_edges:
        hist.bucket_limit.append(edge)

    for v in values:
        hist.bucket.append(v)

    # print(step)
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)
