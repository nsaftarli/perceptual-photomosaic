import tensorflow as tf
import numpy as np
from src.layers import *
from src.utils import *
from src.VGG16 import *


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
        self.build_graph()
        print('Num Variables: ', np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.all_variables()]))

    def build_graph(self):
        trainable = self.my_config['train']

        # GRAPH INPUTS
        with tf.name_scope('Graph_Inputs'):
            self.temperature = tf.placeholder(tf.float32, shape=[])
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
            self.val_loss = tf.placeholder(tf.float32, shape=[])
            self.next_batch = self.dataset.iterator.get_next()
            self.input = self.next_batch[0]
            self.index = self.next_batch[1]
            self.dataset_size = self.next_batch[2]

        self.batch_size = tf.shape(self.input)[0]
        templates_shape = self.templates.get_shape().as_list()
        self.template_h, self.template_w, self.num_templates = \
            [templates_shape[1], templates_shape[2], templates_shape[4]]

        # ENCODER
        with tf.name_scope('Encoder'):
            self.encoder = VGG16(input=self.input)
            self.decoder_in = self.encoder.pool3

        # DECODER
        with tf.name_scope('Decoder'):
            self.conv6 = ConvLayer(self.decoder_in, 'conv6', 4096, 1, trainable=trainable)
            self.conv7 = ConvLayer(self.conv6, 'conv7', 1024, 1, trainable=trainable)
            self.conv8 = ConvLayer(self.conv7, 'conv8', 512, 1, trainable=trainable)
            self.conv9 = ConvLayer(self.conv8, 'conv9', 256, 1, trainable=trainable)
            self.conv10 = ConvLayer(self.conv9, 'conv10', 128, 1, trainable=trainable)
            self.conv11 = ConvLayer(self.conv10, 'conv11', 64, 1, trainable=trainable)
            self.conv12 = ConvLayer(self.conv11, 'conv12', self.num_templates, 1, activation=None, trainable=trainable)

        # Computing template coefficients
        with tf.name_scope('Coefficient_Calculation'):
            # (B, H/H_T, W/W_T, N_T)
            self.softmax = tf.nn.softmax(self.conv12 * self.temperature)
            softmax_shape = tf.shape(self.softmax)
            self.softmax_h, self.softmax_w = (softmax_shape[1], softmax_shape[2])
            # (B, (H/H_T) * (W/W_T), N_T)
            self.reshaped_softmax = \
                tf.reshape(self.softmax,
                           [-1, self.softmax_h * self.softmax_w, self.num_templates])

        # TEMPLATE MODIFICATIONS
        with tf.name_scope('Template_Modifications'):
            self.template_r, self.template_g, self.template_b = tf.unstack(self.templates, axis=3)

            # Unstacked templates are reshaped and tiled to (B, N_T, H_T * W_T)
            self.template_r = tf.transpose(tf.reshape(self.template_r, [1, -1, self.num_templates]), perm=[0, 2, 1])
            self.template_r = tf.tile(self.template_r, [self.batch_size, 1, 1])

            self.template_g = tf.transpose(tf.reshape(self.template_g, [1, -1, self.num_templates]), perm=[0, 2, 1])
            self.template_g = tf.tile(self.template_g, [self.batch_size, 1, 1])

            self.template_b = tf.transpose(tf.reshape(self.template_b, [1, -1, self.num_templates]), perm=[0, 2, 1])
            self.template_b = tf.tile(self.template_b, [self.batch_size, 1, 1])

        # Constructing argmax output, channel-wise
        with tf.name_scope('Hard_Output'):
            self.reshaped_argmax = tf.argmax(self.reshaped_softmax, axis=-1)
            self.reshaped_argmax = tf.one_hot(self.reshaped_argmax, self.num_templates)
            self.hard_output = self.construct_output('Channel-wise_Soft_Output',
                                                     self.reshaped_argmax)

        if self.my_config['train']:
            # Constructing softmax output, channel-wise
            with tf.name_scope('Soft_Output'):
                self.soft_output = self.construct_output('Channel-wise_Soft_Output',
                                                         self.reshaped_softmax)
            with tf.name_scope('Perceptual_Loss'):
                self.loss = self.perceptual_loss(self.input, self.soft_output)

            self.build_summaries()

    def construct_output(self, name, coefficients):
        with tf.name_scope(name):
            # (B, (H/H_T) * (W/W_T), H_T * W_T)
            self.output_r = tf.matmul(coefficients, self.template_r)
            # (B, H, W, 1)
            self.output_r = tf.reshape(tf.transpose(tf.reshape(
                self.output_r, [self.batch_size, self.softmax_h, self.softmax_w, self.template_h, self.template_w]),
                perm=[0, 1, 3, 2, 4]), [self.batch_size, self.softmax_h * self.template_h, self.softmax_w * self.template_w, 1])

            self.output_g = tf.matmul(coefficients, self.template_g)
            self.output_g = tf.reshape(tf.transpose(tf.reshape(
                self.output_g, [self.batch_size, self.softmax_h, self.softmax_w, self.template_h, self.template_w]),
                perm=[0, 1, 3, 2, 4]), [self.batch_size, self.softmax_h * self.template_h, self.softmax_w * self.template_w, 1])

            self.output_b = tf.matmul(coefficients, self.template_b)
            self.output_b = tf.reshape(tf.transpose(tf.reshape(
                self.output_b, [self.batch_size, self.softmax_h, self.softmax_w, self.template_h, self.template_w]),
                perm=[0, 1, 3, 2, 4]), [self.batch_size, self.softmax_h * self.template_h, self.softmax_w * self.template_w, 1])

            return tf.concat([self.output_r, self.output_g, self.output_b], axis=3)

    def perceptual_loss(self, target, predicted):
        with tf.name_scope('Blurred_Predicted'):
            self.blurred_predicted = GaussianBlurLayer(predicted, 'Blurred_Predicted',
                                                       self.template_h, self.template_w)

        # Gaussian Pyramid
        with tf.name_scope('Gaussian_Pyramid'):
            self.target_downsampled_x2 = GaussianBlurLayer(target, 'Target_Downsampled_x2',
                                                      self.template_h, self.template_w, 2)
            self.target_downsampled_x4 = GaussianBlurLayer(self.target_downsampled_x2, 'Target_Downsampled_x4',
                                                      self.template_h, self.template_w, 2)
            self.predicted_downsampled_x2 = GaussianBlurLayer(predicted, 'Predicted_Downsampled_x2',
                                                         self.template_h, self.template_w, 2)
            self.predicted_downsampled_x4 = GaussianBlurLayer(self.predicted_downsampled_x2, 'Predicted_Downsampled_x4',
                                                         self.template_h, self.template_w, 2)

        # Get image features
        target_feats = self.encoder
        target_downsampled_x2_feats = VGG16(input=self.target_downsampled_x2)
        target_downsampled_x4_feats = VGG16(input=self.target_downsampled_x4)
        blurred_predicted_feats = VGG16(input=self.blurred_predicted)
        predicted_downsampled_x2_feats = VGG16(input=self.predicted_downsampled_x2)
        predicted_downsampled_x4_feats = VGG16(input=self.predicted_downsampled_x4)

        # Feature scale factors
        target_conv1_1 = target_feats.conv1_1
        conv1_1_shape = tf.cast(tf.shape(target_conv1_1), tf.float32)
        conv1_1_scale = 1.0 / (conv1_1_shape[1] *
                               conv1_1_shape[2] *
                               conv1_1_shape[3])
        target_conv2_1 = target_feats.conv2_1
        conv2_1_shape = tf.cast(tf.shape(target_conv2_1), tf.float32)
        conv2_1_scale = 1.0 / (conv2_1_shape[1] *
                               conv2_1_shape[2] *
                               conv2_1_shape[3])
        target_conv3_1 = target_feats.conv3_1
        conv3_1_shape = tf.cast(tf.shape(target_conv3_1), tf.float32)
        conv3_1_scale = 1.0 / (conv3_1_shape[1] *
                               conv3_1_shape[2] *
                               conv3_1_shape[3])
        target_conv4_1 = target_feats.conv4_1
        conv4_1_shape = tf.cast(tf.shape(target_conv4_1), tf.float32)
        conv4_1_scale = 1.0 / (conv4_1_shape[1] *
                               conv4_1_shape[2] *
                               conv4_1_shape[3])
        target_conv5_1 = target_feats.conv5_1
        conv5_1_shape = tf.cast(tf.shape(target_conv5_1), tf.float32)
        conv5_1_scale = 1.0 / (conv5_1_shape[1] *
                               conv5_1_shape[2] *
                               conv5_1_shape[3])

        # FEATURE RECONSTRUCTION LOSS
        with tf.name_scope('Multi_Scale_MSE'):
            with tf.name_scope('Original_Scale_MSE'):
                self.conv1_1_loss = \
                    tf.losses.mean_squared_error(
                        target_feats.conv1_1,
                        blurred_predicted_feats.conv1_1) #* \
                    #conv1_1_scale
                self.conv2_1_loss = \
                    tf.losses.mean_squared_error(
                        target_feats.conv2_1,
                        blurred_predicted_feats.conv2_1) #* \
                    #conv2_1_scale
                self.conv3_1_loss = \
                    tf.losses.mean_squared_error(
                        target_feats.conv3_1,
                        blurred_predicted_feats.conv3_1) #* \
                    #conv3_1_scale
                self.conv4_1_loss = \
                    tf.losses.mean_squared_error(
                        target_feats.conv4_1,
                        blurred_predicted_feats.conv4_1) #* \
                    #conv4_1_scale
                self.conv5_1_loss = \
                    tf.losses.mean_squared_error(
                        target_feats.conv5_1,
                        blurred_predicted_feats.conv5_1) #* \
                    #conv5_1_scale
                original_scale_losses = [self.conv1_1_loss, self.conv2_1_loss,
                                         self.conv3_1_loss, self.conv4_1_loss,
                                         self.conv5_1_loss]

            with tf.name_scope('Downsampled_x2_MSE'):
                downsampled_x2_conv1_1_loss = \
                    tf.losses.mean_squared_error(
                        target_downsampled_x2_feats.conv1_1,
                        predicted_downsampled_x2_feats.conv1_1) #* \
                    #(conv1_1_scale / 2)
                downsampled_x2_conv2_1_loss = \
                    tf.losses.mean_squared_error(
                        target_downsampled_x2_feats.conv2_1,
                        predicted_downsampled_x2_feats.conv2_1) #* \
                    #(conv2_1_scale / 2)
                downsampled_x2_conv3_1_loss = \
                    tf.losses.mean_squared_error(
                        target_downsampled_x2_feats.conv3_1,
                        predicted_downsampled_x2_feats.conv3_1) #* \
                    #(conv3_1_scale / 2)
                downsampled_x2_conv4_1_loss = \
                    tf.losses.mean_squared_error(
                        target_downsampled_x2_feats.conv4_1,
                        predicted_downsampled_x2_feats.conv4_1) #* \
                    #(conv4_1_scale / 2)
                downsampled_x2_conv5_1_loss = \
                    tf.losses.mean_squared_error(
                        target_downsampled_x2_feats.conv5_1,
                        predicted_downsampled_x2_feats.conv5_1) #* \
                    #(conv5_1_scale / 2)
                downsampled_x2_losses = [downsampled_x2_conv1_1_loss,
                                         downsampled_x2_conv2_1_loss,
                                         downsampled_x2_conv3_1_loss,
                                         downsampled_x2_conv4_1_loss,
                                         downsampled_x2_conv5_1_loss]

            with tf.name_scope('Downsampled_x4_MSE'):
                downsampled_x4_conv1_1_loss = \
                    tf.losses.mean_squared_error(
                        target_downsampled_x4_feats.conv1_1,
                        predicted_downsampled_x4_feats.conv1_1) #* \
                    #(conv1_1_scale / 4)
                downsampled_x4_conv2_1_loss = \
                    tf.losses.mean_squared_error(
                        target_downsampled_x4_feats.conv2_1,
                        predicted_downsampled_x4_feats.conv2_1) #* \
                    #(conv2_1_scale / 4)
                downsampled_x4_conv3_1_loss = \
                    tf.losses.mean_squared_error(
                        target_downsampled_x4_feats.conv3_1,
                        predicted_downsampled_x4_feats.conv3_1) #* \
                    #(conv3_1_scale / 4)
                downsampled_x4_conv4_1_loss = \
                    tf.losses.mean_squared_error(
                        target_downsampled_x4_feats.conv4_1,
                        predicted_downsampled_x4_feats.conv4_1) #* \
                    #(conv4_1_scale / 4)
                downsampled_x4_conv5_1_loss = \
                    tf.losses.mean_squared_error(
                        target_downsampled_x4_feats.conv5_1,
                        predicted_downsampled_x4_feats.conv5_1) #* \
                    #(conv5_1_scale / 4)
                downsampled_x4_losses = [downsampled_x4_conv1_1_loss,
                                         downsampled_x4_conv2_1_loss,
                                         downsampled_x4_conv3_1_loss,
                                         downsampled_x4_conv4_1_loss,
                                         downsampled_x4_conv5_1_loss]

        return \
            (tf.add_n(original_scale_losses) +
            tf.add_n(downsampled_x2_losses) +
            tf.add_n(downsampled_x4_losses)) / tf.cast(self.batch_size, tf.float32)

    def build_summaries(self):
        with tf.name_scope('Summaries'):
            tf.summary.image('target', tf.cast(self.input, tf.uint8), max_outputs=6)
            tf.summary.image('target_downsampled_x2', tf.cast(self.target_downsampled_x2, tf.uint8), max_outputs=6)
            tf.summary.image('target_downsampled_x4', tf.cast(self.target_downsampled_x4, tf.uint8), max_outputs=6)
            tf.summary.image('soft_output', tf.cast(self.soft_output, tf.uint8), max_outputs=6)
            tf.summary.image('blurred_predicted', tf.cast(self.blurred_predicted, tf.uint8), max_outputs=6)
            tf.summary.image('predicted_downsampled_x2', tf.cast(self.predicted_downsampled_x2, tf.uint8), max_outputs=6)
            tf.summary.image('predicted_downsampled_x4', tf.cast(self.predicted_downsampled_x4, tf.uint8), max_outputs=6)
            tf.summary.image('hard_output', tf.cast(self.hard_output, tf.uint8), max_outputs=6)
            tf.summary.scalar('entropy', EntropyLayer(self.softmax))
            tf.summary.scalar('variance', VarianceLayer(self.softmax, num_bins=self.num_templates))
            tf.summary.scalar('temperature', self.temperature)
            tf.summary.scalar('train_loss', self.loss)
            tf.summary.scalar('conv1_1_loss', self.conv1_1_loss)
            tf.summary.scalar('conv2_1_loss', self.conv2_1_loss)
            tf.summary.scalar('conv3_1_loss', self.conv3_1_loss)
            tf.summary.scalar('conv4_1_loss', self.conv4_1_loss)
            tf.summary.scalar('conv5_1_loss', self.conv5_1_loss)
            self.val_loss_summary = tf.summary.scalar('validation_loss', self.val_loss, collections=['val'])
            self.summaries = tf.summary.merge_all()

    def train(self):
        opt = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
        train_step = opt.minimize(self.loss)

        saver = tf.train.Saver(max_to_keep=0, pad_step_number=16)
        saved_iterator = \
            tf.contrib.data.make_saveable_from_iterator(self.dataset.iterator)

        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saved_iterator)

        with tf.Session(config=self.tf_config) as sess:
            resume, iterations_so_far = check_snapshots(self.my_config['run_id'])
            writer = tf.summary.FileWriter('logs/' + self.my_config['run_id'],
                                           sess.graph)


            if resume:
                saver.restore(sess, resume)
            else:
                sess.run(tf.global_variables_initializer())

            train_handle = sess.run(self.dataset.get_training_handle())
            val_handle = sess.run(self.dataset.get_validation_handle())
            temperature = self.my_config['init_temperature']
            learning_rate = self.my_config['learning_rate']
            for i in range(iterations_so_far, self.my_config['iterations']):
                # Temperature Schedule
                if i > 1 and i % 1000 == 0:
                    if i < 8000:
                        temperature *= 2

                train_feed_dict = {self.learning_rate: learning_rate,
                                   self.temperature: temperature,
                                   self.dataset.handle: train_handle}
                results = sess.run([train_step, self.loss, self.summaries],
                                   feed_dict=train_feed_dict)
                loss = results[1]
                train_summary = results[2]

                # Saving/Logging
                if i % self.my_config['print_freq'] == 0:
                    print('(' + self.my_config['run_id'] + ') ' +
                          'Iteration ' + str(i) +
                          ', Loss: ' + str(loss))

                if i % self.my_config['val_freq'] == 0:
                    # Reset validation iterator to beginning
                    sess.run(self.dataset.val_iterator.initializer)

                    val_feed_dict = {self.temperature: temperature,
                                     self.dataset.handle: val_handle}
                    val_loss = self.validate(sess, val_feed_dict, i)
                    val_summary = sess.run(self.val_loss_summary,
                                           feed_dict={self.val_loss: val_loss})
                    writer.add_summary(val_summary, i)
                    writer.flush()

                if i % self.my_config['log_freq'] == 0:
                    writer.add_summary(train_summary, i)
                    writer.flush()

                if i % self.my_config['chkpt_freq'] == 0 and i != iterations_so_far:
                    print('Saving Snapshot...')
                    saver.save(sess, 'snapshots/' +
                               self.my_config['run_id'] + '/' +
                               'checkpoint_iter', global_step=i)

    def validate(self, sess, feed_dict, train_iter):
        loss_sum = 0.0
        avg_loss = 0
        while True:
            try:
                res = sess.run([self.loss, self.dataset_size, self.index, self.input, self.hard_output],
                               feed_dict=feed_dict)
                loss_sum += res[0]
                output = np.concatenate([res[3], res[4]], axis=2)
                write_directory = 'data/out/' + self.my_config['run_id'] + \
                                  '/' + str(train_iter)
                write_images(output, write_directory, res[2])
            except tf.errors.OutOfRangeError:
                dataset_size = res[1][-1]
                avg_loss = loss_sum / dataset_size
                break
        return avg_loss

    def predict(self, model_path):
        saver = tf.train.Saver()
        checkpoint_path = tf.train.latest_checkpoint(model_path)

        with tf.Session(config=self.tf_config) as sess:
            saver.restore(sess, checkpoint_path)

            pred_handle = sess.run(self.dataset.get_prediction_handle())
            sess.run(self.dataset.pred_iterator.initializer)
            feed_dict = {self.temperature: 1.0,
                         self.dataset.handle: pred_handle}

            while True:
                try:
                    res = sess.run([self.index, self.hard_output],
                                   feed_dict=feed_dict)
                    output = res[1]
                    write_directory = self.dataset.pred_path + '/out/'
                    write_images(output, write_directory, res[0])
                except tf.errors.OutOfRangeError:
                    break
