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
            self.conv6 = ConvLayer(self.decoder_in, 'conv6', 256, 1, trainable=trainable)
            self.conv7 = ConvLayer(self.conv6, 'conv7', self.num_templates, 3, activation=None, trainable=trainable)

        # Computing template coefficients
        with tf.name_scope('Coefficient_Calculation'):
            # (B, H/H_T, W/W_T, N_T)
            self.softmax = tf.nn.softmax(self.conv7 * self.temperature)
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
                # Gather user options for perceptual loss
                layers = self.my_config['layers']
                layer_scale_factors = self.my_config['layer_scale_factors']
                downscale_factors = self.my_config['downscale_factors']
                blur_factors = self.my_config['blur_factors']
                blur_windows = self.my_config['blur_windows']

                # Compute perceptual loss at chosen layers and scales
                self.loss = \
                    self.perceptual_loss(self.input, self.soft_output,
                                         layers, layer_scale_factors,
                                         downscale_factors, blur_factors,
                                         blur_windows)

                # Average loss over batch
                self.loss = self.loss / tf.to_float(self.batch_size)

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

    def perceptual_loss(self, target, predicted, layers,
                        layer_scale_factors, downscale_factors,
                        blur_factors, blur_windows):
        # Split user params
        # Format:
        #   layers = 'conv1_1,conv2_1,conv3_1,conv4_1,conv5_1'
        #   layer_scale_factors = '1,0.5,0.25,0.125,0.03125'
        #   downscale_factors = '0,1,2:0,1,2:0,1,2:0,1,2:0,1,2'
        #   blur_factors = '3,3,3'  for blurring predicted at each scale
        #   blur_windows = '8,8,8'  for blurring predicted at each scale
        layers = layers.split(',')
        layer_scale_factors = [float(s) for s in layer_scale_factors.split(',')]
        downscale_factors = [s.split(',') for s in downscale_factors.split(':')]
        downscale_factors = [[float(s) for s in arr]
                             for arr in downscale_factors]
        blur_factors = [float(s) for s in blur_factors.split(',')]
        blur_windows = [float(s) for s in blur_windows.split(',')]

        # Flatten downscale_factors and retrieve (ordered) unique values
        # This will be used to construct the Gaussian Pyramid
        scales = np.unique([item for sublist in downscale_factors
                            for item in sublist])

        # Construct Gaussian Pyramid
        with tf.name_scope('Gaussian_Pyramid'):
            pyramid_predicted = []
            pyramid_target = []
            for s in scales:
                # Don't downsample if downscale factor is 0 (i.e., 2^0)
                if s == 0:
                    pyramid_predicted.append(predicted)
                    pyramid_target.append(target)
                    continue

                downscale_factor = int(2**s)

                predicted_name = 'Predicted_Downsampled_x' + \
                    str(downscale_factor)
                target_name = 'Target_Downsampled_x' + str(downscale_factor)

                # sigma = sqrt(n / 4) where n is level in Pascal's triangle
                sigma = (downscale_factor / 4.0)**0.5
                k_h = int(downscale_factor + 1)
                k_w = int(k_h)

                predicted_at_curr_scale = \
                    GaussianBlurLayer(predicted, predicted_name,
                                      k_h=k_h, k_w=k_w,
                                      stride=downscale_factor, sigma=sigma)
                target_at_curr_scale = \
                    GaussianBlurLayer(target, target_name,
                                      k_h=k_h, k_w=k_w,
                                      stride=downscale_factor, sigma=sigma)

                # Append newly computed images at current scale to pyramid
                pyramid_predicted.append(predicted_at_curr_scale)
                pyramid_target.append(target_at_curr_scale)

        # Get image features of predicted and target at each scale
        pyramid_predicted_feats = []
        pyramid_target_feats = []
        for i in range(len(scales)):
            # Blur each scale for predicted based on what user wants
            try:
                blur_factor = blur_factors[i]
                blur_window = int(blur_windows[i])
                blurred_predicted_name = 'Blurred_Predicted_Downsampled_x' + \
                    str(int(2**scales[i]))
                blurred_predicted = \
                    GaussianBlurLayer(pyramid_predicted[i],
                                      blurred_predicted_name,
                                      k_h=blur_window, k_w=blur_window,
                                      stride=1, sigma=blur_factor)
                pyramid_predicted[i] = blurred_predicted
            except IndexError:
                pass

            predicted_feats = VGG16(input=pyramid_predicted[i])
            target_feats = VGG16(input=pyramid_target[i])

            # Append feats at this scale to pyramid of feats
            pyramid_predicted_feats.append(predicted_feats)
            pyramid_target_feats.append(target_feats)

        # Compute feature reconstruction losses at each layer and pyramid scale
        with tf.name_scope('Feature_Reconstruction_Loss'):
            losses = []
            i = 0
            for scales_at_layer in downscale_factors:
                layer = layers[i]
                losses_at_layer = []
                for scale in scales_at_layer:
                    # Get index
                    index = np.where(scales == scale)[0][0]

                    predicted_layer = getattr(pyramid_predicted_feats[index],
                                              layer)
                    target_layer = getattr(pyramid_target_feats[index], layer)

                    # Append feature reconstruction loss
                    loss_layer = \
                        tf.losses.mean_squared_error(target_layer,
                                                     predicted_layer)
                    losses_at_layer.append(loss_layer)

                # Append to list of all losses at all scales and layers
                losses.append(losses_at_layer)

                i += 1

            # Add together losses at chosen scales for each layer
            loss_at_layer = []
            i = 0
            for losses_at_layer in losses:
                loss = tf.add_n(losses_at_layer)

                # Weight the total loss at this layer and average across scales
                loss *= (layer_scale_factors[i] / len(losses_at_layer))

                # Append to list of scale-averaged and user-weighted losses at
                # each layer
                loss_at_layer.append(loss)

                i += 1

            # Add together losses at all layers
            final_loss = tf.add_n(loss_at_layer)

            # Average across number of layers used
            final_loss /= len(layers)

        # For usage in summary
        self.layers = layers
        self.scales = scales
        self.losses = losses
        self.loss_at_layer = loss_at_layer
        self.pyramid_predicted = pyramid_predicted
        self.pyramid_target = pyramid_target

        return final_loss

    def build_summaries(self):
        with tf.name_scope('Summaries'):
            # Summary of gaussian pyramid of target and predicted
            for s in self.scales:
                if s == 0:
                    tf.summary.image('Target', tf.cast(self.input, tf.uint8),
                                     max_outputs=6)
                    tf.summary.image('Blurred_Predicted',
                                     tf.cast(self.pyramid_predicted[0],
                                             tf.uint8),
                                     max_outputs=6)
                    continue

                index = np.where(self.scales == s)[0][0]
                images_target = tf.cast(self.pyramid_target[index], tf.uint8)
                images_pred = tf.cast(self.pyramid_predicted[index], tf.uint8)
                downscale_factor = int(2**s)
                name_target = 'Target_Downsampled_x' + str(downscale_factor)
                name_pred = 'Blurred_Predicted_Downsampled_x' + \
                    str(downscale_factor)
                tf.summary.image(name_target, images_target, max_outputs=6)
                tf.summary.image(name_pred, images_pred, max_outputs=6)

            # Unblurred predicted output and argmax'd version of it
            tf.summary.image('Soft_Output', tf.cast(self.soft_output,
                                                    tf.uint8), max_outputs=6)
            tf.summary.image('Hard_Output', tf.cast(self.hard_output,
                                                    tf.uint8), max_outputs=6)

            # Misc
            tf.summary.scalar('Entropy', EntropyLayer(self.softmax))
            tf.summary.scalar('Variance', VarianceLayer(self.softmax, num_bins=self.num_templates))
            tf.summary.scalar('Temperature', self.temperature)

            # Losses
            tf.summary.scalar('Train_Loss', self.loss)
            self.val_loss_summary = \
                tf.summary.scalar('Validation_Loss', self.val_loss,
                                  collections=['val'])
            for i in range(len(self.loss_at_layer)):
                loss = self.loss_at_layer[i]
                loss_name = self.layers[i] + '_Loss'
                tf.summary.scalar(loss_name, loss)

            # Merge all summaries
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

            temp_schedule, total_num_temp_updates = \
                temperature_schedule(temperature, 15, 10,
                                     self.dataset.train_dataset_size,
                                     self.my_config['train_batch_size'])

            num_temp_updates = 0
            for i in range(iterations_so_far, self.my_config['iterations']):
                # Temperature Schedule (Linear)
                if i % 10 == 0 and num_temp_updates <= total_num_temp_updates:
                    temperature += temp_schedule
                    num_temp_updates += 1

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
