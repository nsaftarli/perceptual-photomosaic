import os
import tensorflow as tf
from scipy.misc import imread, imsave
from VGG16 import VGG16
from utils import GaussianBlurLayer
names = ['conv1_1', 'conv1_2',
         'conv2_1', 'conv2_2',
         'conv3_1', 'conv3_2', 'conv3_3',
         'conv4_1', 'conv4_2', 'conv4_3',
         'conv5_1', 'conv5_2', 'conv5_3']


def conv_avg_test(path='../data/tests/conv_avg_test/',
                  file='starry_night.jpg'):

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + 'feats_x1/'):
        os.makedirs(path + 'feats_x1/')
    if not os.path.exists(path + 'feats_x2/'):
        os.makedirs(path + 'feats_x2/')
    if not os.path.exists(path + 'feats_x4/'):
        os.makedirs(path + 'feats_x4/')

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    img = tf.expand_dims(tf.to_float(imread(path + file, mode='RGB')), axis=0)
    print(img)
    img_downsample_x2 = GaussianBlurLayer(img, 'Target_Downsampled_x2', 3, 3, 2)
    img_downsample_x4 = GaussianBlurLayer(img_downsample_x2, 'Target_Downsampled_x4', 3, 3, 2)

    means = get_means(img)
    means_x2 = get_means(img_downsample_x2)
    means_x4 = get_means(img_downsample_x4)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        results_x1 = sess.run(means)
        results_x2 = sess.run(means_x2)
        results_x4 = sess.run(means_x4)
        write_images(results_x1, path=(path + 'feats_x1'))
        write_images(results_x2, path=(path + 'feats_x2'))
        write_images(results_x4, path=(path + 'feats_x4'))


def write_images(arr, path):
    for i, output in enumerate(arr):
        print(output.shape)
        imsave(path + arr[i] + '.png', output)


def get_means(img):
    vgg = VGG16(img, weight_path='../data/models/vgg16.npy')

    conv1_1_mean = tf.squeeze(tf.reduce_mean(vgg.conv1_1, axis=-1), axis=0)
    conv1_2_mean = tf.squeeze(tf.reduce_mean(vgg.conv1_2, axis=-1), axis=0)

    conv2_1_mean = tf.squeeze(tf.reduce_mean(vgg.conv2_1, axis=-1), axis=0)
    conv2_2_mean = tf.squeeze(tf.reduce_mean(vgg.conv2_2, axis=-1), axis=0)

    conv3_1_mean = tf.squeeze(tf.reduce_mean(vgg.conv3_1, axis=-1), axis=0)
    conv3_2_mean = tf.squeeze(tf.reduce_mean(vgg.conv3_2, axis=-1), axis=0)
    conv3_3_mean = tf.squeeze(tf.reduce_mean(vgg.conv3_3, axis=-1), axis=0)

    conv4_1_mean = tf.squeeze(tf.reduce_mean(vgg.conv4_1, axis=-1), axis=0)
    conv4_2_mean = tf.squeeze(tf.reduce_mean(vgg.conv4_2, axis=-1), axis=0)
    conv4_3_mean = tf.squeeze(tf.reduce_mean(vgg.conv4_3, axis=-1), axis=0)

    conv5_1_mean = tf.squeeze(tf.reduce_mean(vgg.conv5_1, axis=-1), axis=0)
    conv5_2_mean = tf.squeeze(tf.reduce_mean(vgg.conv5_2, axis=-1), axis=0)
    conv5_3_mean = tf.squeeze(tf.reduce_mean(vgg.conv5_3, axis=-1), axis=0)

    means = [conv1_1_mean,
             conv1_2_mean,
             conv2_1_mean,
             conv2_2_mean,
             conv3_1_mean,
             conv3_2_mean,
             conv3_3_mean,
             conv4_1_mean,
             conv4_2_mean,
             conv4_3_mean,
             conv5_1_mean,
             conv5_2_mean,
             conv5_3_mean]
    return means

if __name__ == '__main__':
    conv_avg_test()
