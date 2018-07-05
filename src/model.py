import tensorflow as tf
import numpy as np
from layers import *
from utils import *
from VGG16 import *

w = tf.reshape(tf.constant(gauss2d_kernel(shape=(patch_size, patch_size), sigma=3), dtype=tf.float32),
               [patch_size, patch_size, 1, 1])


class MosaicNet:
    def __init__(self,
                 dataset,
                 templates,
                 tf_config,
                 my_config):
        self.dataset = dataset
        self.templates = tf.to_float(templates)
        self.tf_config = tf_config
        self.my_config = my_config
        self.template_build_graph()

    def build_graph(self):

        # PLACEHOLDERS
        self.temperature = tf.placeholder(tf.float32, shape=[])
        self.next_batch = self.dataset.make_one_shot_iterator().get_next()
        self.input = self.next_batch[0]
        self.index = self.next_batch[1]

        input_shape = tf.shape(self.input)
        batch_size, input_h, input_w, input_c = [input_shape[0], input_shape[1], input_shape[2], input_shape[3]]
        _, template_h, template_w, template_c, num_templates = tf.shape(self.templates)

        # ################ Get Templates #############################################################################
        self.template_r, self.template_g, self.template_b = tf.unstack(self.templates, axis=3)

        self.template_r = tf.transpose(tf.reshape(self.template_r, [1, -1, num_templates]), perm=[0, 2, 1])
        self.template_r = tf.tile(self.template_r, [batch_size, 1, 1])

        self.template_g = tf.transpose(tf.reshape(self.template_g, [1, -1, num_templates]), perm=[0, 2, 1])
        self.template_g = tf.tile(self.template_g, [batch_size, 1, 1])

        self.template_b = tf.transpose(tf.reshape(self.template_b, [1, -1, num_templates]), perm=[0, 2, 1])
        self.template_b = tf.tile(self.template_b, [batch_size, 1, 1])

        # ENCODER
        with tf.name_scope('Encoder'):
            self.encoder = VGG16(input=self.input)
            self.decoder_in = self.encoder.pool3

        # DECODER
        with tf.name_scope("Decoder"):
            self.conv6 = ConvLayer(self.decoder_in, 'conv6', 4096, 1, trainable)
            self.conv7 = ConvLayer(self.conv6, 'conv7', 1024, 1, trainable)
            self.conv8 = ConvLayer(self.conv7, 'conv8', 512, 1, trainable)
            self.conv9 = ConvLayer(self.conv8, 'conv9', 256, 1, trainable)
            self.conv10 = ConvLayer(self.conv9, 'conv10', 128, 1, trainable)
            self.conv11 = ConvLayer(self.conv10, 'conv11', 64, 1, trainable)
            self.conv12 = ConvLayer(self.conv11, 'conv12', num_templates, 1, trainable, activation=None)
        ##########################################################################################################

        # ###############Softmax###################################################################################
        self.softmax = tf.nn.softmax(self.conv12 * self.temp)
        _, self.softmax_h, self.softmax_w, _ = self.softmax.get_shape().as_list()
        self.template_reshaped_softmax = tf.reshape(self.softmax, [-1, self.softmax_h * self.softmax_w, num_templates])
        ##########################################################################################################

        ###############Output#####################################################################################
        with tf.name_scope('output_and_tile'):
            self.output_r = tf.matmul(self.template_reshaped_softmax, self.template_r)
            self.output_r = tf.reshape(tf.transpose(tf.reshape(
                self.output_r, [batch_size, self.softmax_h, self.softmax_w, patch_size, patch_size]),
                perm=[0, 1, 3, 2, 4]), [batch_size, 376, img_new_size, 1])

            self.output_g = tf.matmul(self.template_reshaped_softmax, self.template_g)
            self.output_g = tf.reshape(tf.transpose(tf.reshape(
                self.output_g, [batch_size, self.softmax_h, self.softmax_w, patch_size, patch_size]),
                perm=[0, 1, 3, 2, 4]), [batch_size, 376, img_new_size, 1])

            self.output_b = tf.matmul(self.template_reshaped_softmax, self.template_b)
            self.output_b = tf.reshape(tf.transpose(tf.reshape(
                self.output_b, [batch_size, self.softmax_h, self.softmax_w, patch_size, patch_size]),
                perm=[0, 1, 3, 2, 4]), [batch_size, 376, img_new_size, 1])

        with tf.name_scope('soft_output'):
            self.view_output = tf.concat([self.output_r, self.output_g, self.output_b], axis=3)

        with tf.name_scope('blurred_out'):
            self.template_blurred_out = self.template_blur_recombine(self.view_output, w, stride=1)

        ##########################################################################################################
        if trainable:
            ################Gaussian Pyramid###########################################################################
            self.in_d1 = self.template_blur_recombine(self.input, w, stride=2)
            self.in_d2 = self.template_blur_recombine(self.in_d1, w, stride=2)
            self.out_d1 = self.template_blur_recombine(self.template_blurred_out, w, stride=2)
            self.out_d2 = self.template_blur_recombine(self.out_d1, w, stride=2)

            # ##############Loss and Regularizers######################################################################
            with tf.name_scope('multiscale_structure_features'):
                self.vgg2 = VGG16(input=self.template_blurred_out, trainable=False)

                self.vgg_in_d1 = VGG16(input=self.in_d1, trainable=False)
                self.vgg_in_d2 = VGG16(input=self.in_d2, trainable=False)
                self.vgg_out_d1 = VGG16(input=self.out_d1, trainable=False)
                self.vgg_out_d2 = VGG16(input=self.out_d2, trainable=False)


            ################Structure Loss############################################################################
            self.f_loss1 = tf.losses.mean_squared_error(self.encoder.conv1_1, self.vgg2.conv1_1)
            self.f_loss2 = tf.losses.mean_squared_error(self.encoder.conv2_1, self.vgg2.conv2_1)
            self.f_loss3 = tf.losses.mean_squared_error(self.encoder.conv3_1, self.vgg2.conv3_1)
            self.f_loss4 = tf.losses.mean_squared_error(self.encoder.conv4_1, self.vgg2.conv4_1)
            self.f_loss5 = tf.losses.mean_squared_error(self.encoder.conv5_1, self.vgg2.conv5_1)

            self.template_blur_loss = (tf.losses.mean_squared_error(self.vgg_in_d1.conv1_1, self.vgg_out_d1.conv1_1)) + \
                             (tf.losses.mean_squared_error(self.vgg_in_d2.conv1_1, self.vgg_out_d2.conv1_1)) + \
                             (tf.losses.mean_squared_error(self.vgg_in_d1.conv2_1, self.vgg_out_d1.conv2_1)) + \
                             (tf.losses.mean_squared_error(self.vgg_in_d2.conv2_1, self.vgg_out_d2.conv2_1))

            self.structure_loss = self.f_loss1 + self.f_loss2 + self.f_loss3 #+ self.f_loss4 + self.f_loss5 #+ self.template_blur_loss
            ###########################################################################################################
            self.tLoss = self.structure_loss + self.template_blur_loss
            ##########################################################################################################

            self.entropy = EntropyRegularizer(self.softmax) * 1e3
            self.variance = VarianceRegularizer(self.softmax, num_temps=NUM_TEMPLATES) * 1e2
            self.template_build_summaries()




    def get_avg_colour(self, input):
        return tf.nn.avg_pool(input, ksize=[1, patch_size, patch_size, 1], strides=[1, patch_size, patch_size, 1], padding='VALID')


    def build_summaries(self):
        tf.summary.image('target', tf.cast(self.input, tf.uint8), max_outputs=6)
        tf.summary.image('output', tf.cast(self.view_output, tf.uint8), max_outputs=6)

        tf.summary.scalar('entropy', self.entropy)
        tf.summary.scalar('variance', self.variance)
        tf.summary.scalar('temperature', self.temp)
        tf.summary.scalar('total_loss', self.tLoss)
        tf.summary.scalar('Struct_loss', self.structure_loss)

        self.summaries = tf.summary.merge_all()

    def blur_recombine(self, input, w, stride=1):
        with tf.name_scope('input'):
            r, g, b = tf.split(input, 3, axis=3)
            r = tf.nn.conv2d(r, w, strides=[1, stride, stride, 1], padding='SAME')
            g = tf.nn.conv2d(g, w, strides=[1, stride, stride, 1], padding='SAME')
            b = tf.nn.conv2d(b, w, strides=[1, stride, stride, 1], padding='SAME')
            return tf.concat([r, g, b], axis=3)


    # TODO: Implement
    def train(self):
        pass

    # TODO: Implement
    def predict(self):
        pass

    def optimize(self, loss):
        lr = tf.placeholder(tf.float32,shape=[])
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
        return opt, lr


    def print_architecture(self):
        print(self.conv1_1.get_shape())
        print(self.conv1_2.get_shape())
        print(self.conv2_1.get_shape())
        print(self.conv2_2.get_shape())
        print(self.conv3_1.get_shape())
        print(self.conv3_2.get_shape())
        print(self.conv3_3.get_shape())
        print(self.conv4_1.get_shape())
        print(self.conv4_2.get_shape())
        print(self.conv4_3.get_shape())
        print(self.conv5_1.get_shape())
        print(self.conv5_2.get_shape())
        print(self.conv5_3.get_shape())
        print(self.conv6_1.get_shape())
        print(self.conv6_2.get_shape())
        print(self.conv6_3.get_shape())
        print(self.conv7_3.get_shape())
        print(self.conv7_2.get_shape())
        print(self.conv7_3.get_shape())
        print(self.conv8_1.get_shape())
        print(self.conv8_2.get_shape())
        print(self.conv8_3.get_shape())
        print(self.conv9_1.get_shape())
        print(self.conv9_2.get_shape())
        print(self.conv10_1.get_shape())
        print(self.conv10_2.get_shape())
        print(self.softmax.get_shape())
        print(self.flat_softmax.get_shape())
        print('Num Variables: ', np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.all_variables()]))
