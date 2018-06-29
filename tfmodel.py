import tensorflow as tf
import numpy as np
from layers import *
from utils import *
from VGG16 import *


'''Data constants'''
const = constants.Constants()
img_data_dir = const.img_data_dir
ascii_data_dir = const.ascii_data_dir
val_data_dir = const.val_data_dir
char_array = const.char_array
char_dict = const.char_dict
img_rows = const.img_rows
img_cols = const.img_cols
text_rows = const.text_rows
text_cols = const.text_cols
dims = const.char_count
experiments_dir = const.experiments_dir
coco_dir = const.coco_dir
img_size = const.img_size
patch_size = const.patch_size
num_patches = const.num_patches





VGG_MEAN = [103.939, 116.779, 123.68]
NUM_TEMPLATES = 62
PATCH_SIZE = 8
IM_SHAPE = 512
norm_type = 'group'

w = tf.reshape(tf.constant(gauss2d_kernel(shape=(PATCH_SIZE, PATCH_SIZE), sigma=3), dtype=tf.float32),
               [PATCH_SIZE, PATCH_SIZE, 1, 1])


class ASCIINet:

    def __init__(self,
                 images,
                 templates,
                 weight_path='./weights/vgg16.npy',
                 batch_size=6,
                 trainable=True,
                 rgb=False):
        self.net = self.build_network(images, templates, batch_size=batch_size, trainable=trainable, rgb=rgb)

    def build_network(self, input, templates, batch_size, trainable, rgb):


        self.input = input

        # Get grayscale version of image
        with tf.name_scope('grayscale_input'):
            self.gray_im = tf.tile(tf.reduce_mean(self.input, axis=-1, keep_dims=True),[1, 1, 1, 3])

        # ################Encoder##################################################################################
        with tf.name_scope('VGG_Encoder'):
            self.encoder = VGG16(input=self.gray_im)
            self.decoder_in = self.encoder.pool3

        # ################Decoder##################################################################################
        with tf.name_scope("Decoder"):
            self.conv6, _ = ConvLayer(self.decoder_in, name='conv6', ksize=1, stride=1, out_channels=4096,  norm_type=norm_type, trainable=trainable)
            self.conv7, _ = ConvLayer(self.conv6, name='conv7', ksize=1, stride=1, out_channels=1024,  norm_type=norm_type, trainable=trainable)
            self.conv8, _ = ConvLayer(self.conv7, name='conv8', ksize=1, stride=1, out_channels=512,  norm_type=norm_type, trainable=trainable)
            self.conv9, _ = ConvLayer(self.conv8, name='conv9', ksize=1, stride=1, out_channels=256,  norm_type=norm_type, trainable=trainable)
            self.conv10, _ = ConvLayer(self.conv9, name='conv10', ksize=1, stride=1, out_channels=128,  norm_type=norm_type, trainable=trainable)
            self.conv11, _ = ConvLayer(self.conv10, name='conv11', ksize=1, stride=1, out_channels=64,  norm_type=norm_type, trainable=trainable)
            self.conv12, _ = ConvLayer(self.conv11, name='conv12', ksize=1, stride=1, out_channels=NUM_TEMPLATES,  norm_type=norm_type, trainable=trainable, layer_type='Softmax')

        # ################Other Inputs#############################################################################
        self.temp = tf.placeholder(tf.float32, shape=[])
        if not rgb:
            self.template_tensor = TemplateLayer(templates, rgb=False)
        else:
            self.gray_template_tensor = TemplateLayer(templates, rgb=False)
            self.r, self.g, self.b = TemplateLayer(templates, rgb=True)
            r = tf.expand_dims(self.r, axis=3)
            g = tf.expand_dims(self.g, axis=3)
            b = tf.expand_dims(self.b, axis=3)
        self.temps = tf.concat([r, g, b], axis=3)
        # #################Colour templates##############################
        self.r = tf.transpose(tf.reshape(self.r, [-1, PATCH_SIZE ** 2, NUM_TEMPLATES]), perm=[0, 2, 1])
        self.r = tf.tile(self.r, [batch_size, 1, 1])

        self.g = tf.transpose(tf.reshape(self.g, [-1, PATCH_SIZE ** 2, NUM_TEMPLATES]), perm=[0, 2, 1])
        self.g = tf.tile(self.g, [batch_size, 1, 1])

        self.b = tf.transpose(tf.reshape(self.b, [-1, PATCH_SIZE ** 2, NUM_TEMPLATES]), perm=[0, 2, 1])
        self.b = tf.tile(self.b, [batch_size, 1, 1])

        self.gr = tf.transpose(tf.reshape(self.gray_template_tensor, [-1, PATCH_SIZE ** 2, NUM_TEMPLATES]), perm=[0, 2, 1])
        self.gr = tf.tile(self.gr, [batch_size, 1, 1])

        ##########################################################################################################

        # ###############Softmax###################################################################################
        self.softmax = tf.nn.softmax(self.conv12 * self.temp)
        self.reshaped_softmax = tf.reshape(self.softmax,[-1, (IM_SHAPE//PATCH_SIZE) ** 2, NUM_TEMPLATES])
        ##########################################################################################################

        # ##############Output#####################################################################################
        with tf.name_scope('output_and_tile'):
            self.output_r = tf.matmul(self.reshaped_softmax, self.r)
            self.output_r = tf.reshape(tf.transpose(tf.reshape(
                self.output_r, [batch_size, (IM_SHAPE//PATCH_SIZE), (IM_SHAPE//PATCH_SIZE), 8, 8]),
                perm=[0, 1, 3, 2, 4]), [batch_size, IM_SHAPE, IM_SHAPE, 1])

            self.output_g = tf.matmul(self.reshaped_softmax, self.g)
            self.output_g = tf.reshape(tf.transpose(tf.reshape(
                self.output_g, [batch_size, (IM_SHAPE//PATCH_SIZE), (IM_SHAPE//PATCH_SIZE), 8, 8]),
                perm=[0, 1, 3, 2, 4]), [batch_size, IM_SHAPE, IM_SHAPE, 1])

            self.output_b = tf.matmul(self.reshaped_softmax, self.b)
            self.output_b = tf.reshape(tf.transpose(tf.reshape(
                self.output_b, [batch_size, (IM_SHAPE//PATCH_SIZE), (IM_SHAPE//PATCH_SIZE), 8, 8]),
                perm=[0, 1, 3, 2, 4]), [batch_size, IM_SHAPE, IM_SHAPE, 1])


            self.output_gr = tf.matmul(self.reshaped_softmax, self.gr)
            self.output_gr = tf.reshape(tf.transpose(tf.reshape(
                self.output_gr, [batch_size, (IM_SHAPE//PATCH_SIZE), (IM_SHAPE//PATCH_SIZE), 8, 8]),
                perm=[0, 1, 3, 2, 4]), [batch_size, IM_SHAPE, IM_SHAPE, 1])


            self.view_output = tf.concat([self.output_r, self.output_g, self.output_b], axis=3)
            self.grayscale_output = tf.tile(self.output_gr, [1, 1, 1, 3])




        with tf.name_scope('blurred_out'):
            # self.blurred_out = self.blur_recombine(self.view_output, w, stride=1)
            self.blurred_out = self.blur_recombine(self.grayscale_output, w, stride=1)
            # self.blurred_out = self.grayscale_output
            # self.gray_in = 

        ##########################################################################################################

        # ##############Loss and Regularizers######################################################################
        with tf.name_scope('VGG16_loss'):
            self.vgg2 = VGG16(input=self.blurred_out, trainable=False)
            self.vgg3 = VGG16(input=self.gray_im, trainable=False)
            self.im1 = self.blur_recombine(input, w, stride=2)
            self.im2 = self.blur_recombine(self.blurred_out, w, stride=2)
            self.vgg4 = VGG16(input=self.im1, trainable=False)
            self.vgg5 = VGG16(input=self.im2, trainable=False)
            self.im3 = self.blur_recombine(self.im1, w, stride=2)
            self.im4 = self.blur_recombine(self.im2, w, stride=2)
            self.vgg6 = VGG16(input=self.im3, trainable=False)
            self.vgg7 = VGG16(input=self.im4, trainable=False)

        self.entropy = EntropyRegularizer(self.softmax)
        self.variance = VarianceRegularizer(self.softmax, num_temps=NUM_TEMPLATES)

        self.f_loss1 = 1e5 * tf.losses.mean_squared_error(self.encoder.conv1_1, self.vgg2.conv1_1) / (self.encoder.conv1_1.get_shape().as_list()[1] * self.encoder.conv1_1.get_shape().as_list()[2] * self.encoder.conv1_1.get_shape().as_list()[3])
        self.f_loss2 = 1e5 * tf.losses.mean_squared_error(self.encoder.conv2_1, self.vgg2.conv2_1) / (self.encoder.conv2_1.get_shape().as_list()[1] * self.encoder.conv2_1.get_shape().as_list()[2] * self.encoder.conv1_1.get_shape().as_list()[3])
        self.f_loss3 = 1e5 * tf.losses.mean_squared_error(self.encoder.conv3_1, self.vgg2.conv3_1) / (self.encoder.conv3_1.get_shape().as_list()[1] * self.encoder.conv3_1.get_shape().as_list()[2] * self.encoder.conv1_1.get_shape().as_list()[3])
        self.f_loss4 = 1e5 * tf.losses.mean_squared_error(self.encoder.conv4_1, self.vgg2.conv4_1) / (self.encoder.conv4_1.get_shape().as_list()[1] * self.encoder.conv4_1.get_shape().as_list()[2] * self.encoder.conv1_1.get_shape().as_list()[3])
        self.f_loss5 = 1e5 * tf.losses.mean_squared_error(self.encoder.conv5_1, self.vgg2.conv5_1) / (self.encoder.conv5_1.get_shape().as_list()[1] * self.encoder.conv5_1.get_shape().as_list()[2] * self.encoder.conv1_1.get_shape().as_list()[3])


        self.f_loss1_1 = tf.losses.mean_squared_error(self.encoder.conv1_1, self.vgg2.conv1_1)
        self.f_loss1_2 = tf.losses.mean_squared_error(self.encoder.conv1_2, self.vgg2.conv1_2)
        self.f_loss2_1 = tf.losses.mean_squared_error(self.encoder.conv2_1, self.vgg2.conv2_1)
        self.f_loss2_2 = tf.losses.mean_squared_error(self.encoder.conv2_2, self.vgg2.conv2_2)
        self.f_loss3_1 = tf.losses.mean_squared_error(self.encoder.conv3_1, self.vgg2.conv3_1)
        self.f_loss3_2 = tf.losses.mean_squared_error(self.encoder.conv3_2, self.vgg2.conv3_2)
        self.f_loss3_3 = tf.losses.mean_squared_error(self.encoder.conv3_3, self.vgg2.conv3_3)
        self.f_loss4_1 = tf.losses.mean_squared_error(self.encoder.conv4_1, self.vgg2.conv4_1)
        self.f_loss4_2 = tf.losses.mean_squared_error(self.encoder.conv4_2, self.vgg2.conv4_2)
        self.f_loss4_3 = tf.losses.mean_squared_error(self.encoder.conv4_3, self.vgg2.conv4_3)
        self.f_loss5_1 = tf.losses.mean_squared_error(self.encoder.conv5_1, self.vgg2.conv5_1)
        self.f_loss5_2 = tf.losses.mean_squared_error(self.encoder.conv5_2, self.vgg2.conv5_2)
        self.f_loss5_3 = tf.losses.mean_squared_error(self.encoder.conv5_3, self.vgg2.conv5_3)


        self.blur_loss = 1e5 * (tf.losses.mean_squared_error(self.vgg4.conv1_1, self.vgg5.conv1_1) / (self.vgg4.conv1_1.get_shape().as_list()[1] * self.vgg4.conv1_1.get_shape().as_list()[2] * self.encoder.conv1_1.get_shape().as_list()[3])) + \
            (tf.losses.mean_squared_error(self.vgg6.conv1_1, self.vgg7.conv1_1) / (self.vgg4.conv1_1.get_shape().as_list()[1] * self.vgg4.conv1_1.get_shape().as_list()[2] * self.encoder.conv1_1.get_shape().as_list()[3])) + \
            (tf.losses.mean_squared_error(self.vgg4.conv2_1, self.vgg5.conv2_1) / (self.vgg4.conv1_1.get_shape().as_list()[1] * self.vgg4.conv1_1.get_shape().as_list()[2] * self.encoder.conv1_1.get_shape().as_list()[3])) + \
            (tf.losses.mean_squared_error(self.vgg6.conv2_1, self.vgg7.conv2_1) / (self.vgg4.conv1_1.get_shape().as_list()[1] * self.vgg4.conv1_1.get_shape().as_list()[2] * self.encoder.conv1_1.get_shape().as_list()[3]))


        self.in_colour_map = self.get_avg_colour(self.input)
        self.out_colour_map = self.get_avg_colour(self.view_output)
        self.in_colour_map_lab = rgb_to_lab(self.get_avg_colour(self.input))
        self.out_colour_map_lab = rgb_to_lab(self.get_avg_colour(self.view_output))

        self.colour_loss = tf.losses.mean_squared_error(self.in_colour_map, self.out_colour_map)
        # self.colour_loss_lab = 1e-3 * tf.losses.mean_squared_error(self.in_colour_map_lab, self.out_colour_map_lab)
        self.colour_loss_lab = 1e-1 * (tf.losses.mean_squared_error(self.in_colour_map_lab[:, :, 1:], self.out_colour_map_lab[:, :, 1:]) + \
                                       (0 * tf.losses.mean_squared_error(self.in_colour_map_lab[:, :, :1], self.out_colour_map_lab[:, :, :1])))

        self.loss = self.f_loss1 + self.f_loss2 + self.f_loss3 + self.f_loss4 + self.f_loss5
        self.tLoss = self.loss + self.blur_loss + self.colour_loss_lab
        ##########################################################################################################

        self.build_summaries()


    def get_avg_colour(self, input):
        return tf.nn.avg_pool(input, ksize=[1, PATCH_SIZE, PATCH_SIZE, 1], strides=[1, PATCH_SIZE, PATCH_SIZE, 1], padding='VALID')




    def build_summaries(self):
        tf.summary.image('target', tf.cast(self.input, tf.uint8), max_outputs=1)
        tf.summary.image('output', tf.cast(self.view_output, tf.uint8), max_outputs=1)

        tf.summary.image('in_colour_map', self.in_colour_map, max_outputs=1)
        tf.summary.image('out_colour_map', tf.cast(self.out_colour_map, tf.uint8), max_outputs=1)
        # tf.summary.image('downsampled_in', self.im1)
        # tf.summary.image('downsampled_out', self.im2)
        # tf.summary.image('blurred_out', self.blurred_out)

        tf.summary.scalar('entropy', self.entropy)
        tf.summary.scalar('variance', self.variance)
        tf.summary.scalar('temperature', self.temp)
        tf.summary.scalar('multiscale_structure_loss', self.loss)
        tf.summary.scalar('total_loss', self.tLoss)
        tf.summary.scalar('colour_loss', self.colour_loss)
        tf.summary.scalar('colour_loss_lab', self.colour_loss_lab)
        tf.summary.scalar('f_loss1',self.f_loss1)
        tf.summary.scalar('f_loss2',self.f_loss2)
        tf.summary.scalar('f_loss3',self.f_loss3)
        tf.summary.scalar('f_loss4',self.f_loss4)
        tf.summary.scalar('f_loss5',self.f_loss5)


        # tf.summary.image('e_1', tf.reduce_mean(self.encoder.conv1_1, axis=-1, keep_dims=True))
        # tf.summary.image('e_2', tf.reduce_mean(self.encoder.conv2_1, axis=-1, keep_dims=True))
        # tf.summary.image('e_3', tf.reduce_mean(self.encoder.conv3_1, axis=-1, keep_dims=True))
        # tf.summary.image('e_4', tf.reduce_mean(self.encoder.conv4_1, axis=-1, keep_dims=True))
        # tf.summary.image('e_5', tf.reduce_mean(self.encoder.conv5_1, axis=-1, keep_dims=True))

        # tf.summary.image('v_1', tf.reduce_mean(self.vgg2.conv1_1, axis=-1, keep_dims=True))
        # tf.summary.image('v_2', tf.reduce_mean(self.vgg2.conv2_1, axis=-1, keep_dims=True))
        # tf.summary.image('v_3', tf.reduce_mean(self.vgg2.conv3_1, axis=-1, keep_dims=True))
        # tf.summary.image('v_4', tf.reduce_mean(self.vgg2.conv4_1, axis=-1, keep_dims=True))
        # tf.summary.image('v_5', tf.reduce_mean(self.vgg2.conv5_1, axis=-1, keep_dims=True))





        # tf.summary.image('downsample_target_conv_1', tf.reduce_mean(self.vgg4.conv1_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample_target_conv_2', tf.reduce_mean(self.vgg4.conv2_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample_target_conv_3', tf.reduce_mean(self.vgg4.conv3_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample_target_conv_4', tf.reduce_mean(self.vgg4.conv4_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample_target_conv_5', tf.reduce_mean(self.vgg4.conv5_1, axis=-1, keep_dims=True))

        # tf.summary.image('downsample_output_conv_1', tf.reduce_mean(self.vgg5.conv1_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample_output_conv_2', tf.reduce_mean(self.vgg5.conv2_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample_output_conv_3', tf.reduce_mean(self.vgg5.conv3_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample_output_conv_4', tf.reduce_mean(self.vgg5.conv4_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample_output_conv_5', tf.reduce_mean(self.vgg5.conv5_1, axis=-1, keep_dims=True))

        # tf.summary.image('downsample2_target', self.im3)
        # tf.summary.image('downsample2_target_conv_1', tf.reduce_mean(self.vgg6.conv1_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample2_target_conv_2', tf.reduce_mean(self.vgg6.conv2_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample2_target_conv_3', tf.reduce_mean(self.vgg6.conv3_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample2_target_conv_4', tf.reduce_mean(self.vgg6.conv4_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample2_target_conv_5', tf.reduce_mean(self.vgg6.conv5_1, axis=-1, keep_dims=True))

        # tf.summary.image('downsample2_output', self.im4)
        # tf.summary.image('downsample2_output_conv_1', tf.reduce_mean(self.vgg7.conv1_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample2_output_conv_2', tf.reduce_mean(self.vgg7.conv2_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample2_output_conv_3', tf.reduce_mean(self.vgg7.conv3_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample2_output_conv_4', tf.reduce_mean(self.vgg7.conv4_1, axis=-1, keep_dims=True))
        # tf.summary.image('downsample2_output_conv_5', tf.reduce_mean(self.vgg7.conv5_1, axis=-1, keep_dims=True))

        self.summaries = tf.summary.merge_all()

    def blur_recombine(self, input, w, stride=1):
        with tf.name_scope('input'):
            r, g, b = tf.split(input, 3, axis=3)
            r = tf.nn.conv2d(r, w, strides=[1, stride, stride, 1], padding='SAME')
            g = tf.nn.conv2d(g, w, strides=[1, stride, stride, 1], padding='SAME')
            b = tf.nn.conv2d(b, w, strides=[1, stride, stride, 1], padding='SAME')

            return tf.concat([r, g, b], axis=3)






    def print_architecture(self):
        print(self.conv1_1.get_shape())
        print(self.conv1_2.get_shape())
        # print(self.pool1.get_shape())
        print(self.conv2_1.get_shape())
        print(self.conv2_2.get_shape())
        # print(self.pool2.get_shape())
        print(self.conv3_1.get_shape())
        print(self.conv3_2.get_shape())
        print(self.conv3_3.get_shape())
        # print(self.pool3.get_shape())
        print(self.conv4_1.get_shape())
        print(self.conv4_2.get_shape())
        print(self.conv4_3.get_shape())
        # print(self.pool4.get_shape())
        print(self.conv5_1.get_shape())
        print(self.conv5_2.get_shape())
        print(self.conv5_3.get_shape())
        # print(self.pool5.get_shape())

        print('################################')
        # print(self.up6.get_shape())
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
        # print(self.conv9_3.get_shape())
        print(self.conv10_1.get_shape())
        print(self.conv10_2.get_shape())
        print(self.softmax.get_shape())
        # print(self.prediction.get_shape())
        print(self.flat_softmax.get_shape())
        print(self.r.get_shape())
        print(self.conv11.get_shape())
        # print(self.tloss.get_shape())
        print(tf.trainable_variables())


if __name__ == '__main__':
    m = ASCIINet()
