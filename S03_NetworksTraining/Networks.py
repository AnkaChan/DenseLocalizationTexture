import cv2, glob, itertools, json
import numpy as np
from os.path import join
import os, tqdm
from pathlib import Path

import tensorflow as tf
from feature_extractor import MobileNet, Resnet, Vgg16
from modules import atrous_spatial_pyramid_pooling
from datetime import datetime

class Config:
    def __init__(self):
        self.lrDecayStep = 1000
        self.lrDecayRate = 0.7
        self.momentum = 0.5

class UVExtractor(object):
    def __init__(self, base_architecture, training=True, ignore_label=0, batch_norm_momentum=0.9997,
                 pre_trained_model=None, log_dir='./', inputShape = None, labelShape=None, cfg=Config()):
        tf.reset_default_graph()

        self.cfg=cfg
        self.outputStride=4

        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.ignore_label = ignore_label
        self.inputs_shape = inputShape if inputShape is not None else [None, 60, 60, 3]
        self.labels_shape = labelShape if inputShape is not None else [None, 60, 60, 2]

        self.imgWidth = inputShape[2]
        self.imgHeight = inputShape[1]

        self.training = training
        self.inputs = tf.placeholder(tf.float32, shape=self.inputs_shape, name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=self.labels_shape, name='labels')

        self.target_height = tf.placeholder(tf.int32, None, name='target_image_height')
        self.target_width = tf.placeholder(tf.int32, None, name='target_image_width')

        self.weight_decay = tf.placeholder(tf.float32, None, name='weight_decay')
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
        self.batch_norm_momentum = batch_norm_momentum

        self.feature_map = self.backbone_initializer(base_architecture)
        if pre_trained_model:
            self.initialize_backbone_from_pretrained_weights(pre_trained_model)
        self.outputs, self.outputs_resized = self.model_initializer()

        self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
        self.loss = self.loss_initializer()

        self.optStep = tf.Variable(0, trainable=False)
        self.lrDecayRate_ph = tf.placeholder(tf.float32, None, name='lrDecayRate_ph')
        self.lrDecayStep_ph = tf.placeholder(tf.int32, None, name='lrDecayStep_ph')

        self.learning_rate_decayed = tf.train.exponential_decay(self.learning_rate, self.optStep, self.lrDecayStep_ph,
                                                                self.lrDecayRate_ph)
        
        self.optimizer = self.optimizer_initializer()

        # Initialize tensorflow session
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        # if self.training:
        #     self.train_step = 0
        #     now = datetime.now()
        #     self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
        #     self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
        #     self.train_summaries, self.valid_summaries = self.summary()

    def backbone_initializer(self, base_architecture):

        with tf.variable_scope('backbone'):
            if base_architecture == 'vgg16':
                features = Vgg16(self.inputs, self.weight_decay, self.batch_norm_momentum)
            elif base_architecture.startswith('resnet'):
                n_layers = int(base_architecture.split('_')[-1])
                features = Resnet(n_layers, self.inputs, self.weight_decay, self.batch_norm_momentum, self.is_training, output_stride=self.outputStride)
            elif base_architecture.startswith('mobilenet'):
                depth_multiplier = float(base_architecture.split('_')[-1])
                features = MobileNet(depth_multiplier, self.inputs, self.weight_decay, self.batch_norm_momentum,
                                     self.is_training)
            else:
                raise ValueError('Unknown backbone architecture!')

        return features

    def model_initializer(self):

        pools = atrous_spatial_pyramid_pooling(inputs=self.feature_map, filters=256, regularizer=self.regularizer)
        logits = tf.layers.conv2d(inputs=pools, filters=2, kernel_size=(1, 1), name='logits')
        #         outputs = tf.image.resize_bilinear(images=logits, size=(self.target_height, self.target_width), name='resized_outputs')
        outputs = logits
        outputs_resized = tf.image.resize_bilinear(images=logits, size=(self.target_height, self.target_width),
                                                   name='resized_outputs')
        return outputs, outputs_resized

    def loss_initializer(self):

        #         labels_linear = tf.reshape(tensor=self.labels, shape=[-1])
        #         not_ignore_mask = tf.to_float(tf.not_equal(labels_linear, self.ignore_label))
        # The locations represented by indices in indices take value on_value, while all other locations take value off_value.
        # For example, ignore label 255 in VOC2012 dataset will be set to zero vector in onehot encoding (looks like the not ignore mask is not required)
        # onehot_labels = tf.one_hot(indices=labels_linear, depth=, on_value=1.0, off_value=0.0)

        # loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=tf.reshape(self.outputs, shape=[-1, self.num_classes]), weights=not_ignore_mask)

        loss = self.Smooth_l1_loss(self.labels, self.outputs_resized)

        return loss

    def optimizer_initializer(self):

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.optStep)

        return optimizer

    def initialize_backbone_from_pretrained_weights(self, path_to_pretrained_weights):

        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=['global_step'])
        valid_prefix = 'backbone/'
        tf.train.init_from_checkpoint(path_to_pretrained_weights,
                                      {v.name[len(valid_prefix):].split(':')[0]: v for v in variables_to_restore if
                                       v.name.startswith(valid_prefix)})

    def Smooth_l1_loss(self, labels, predictions, scope=tf.GraphKeys.LOSSES):
        with tf.variable_scope(scope):
            diff = tf.abs(labels - predictions)
            less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)  # Bool to float32
            smooth_l1_loss = (less_than_one * 0.5 * diff ** 2) + (1.0 - less_than_one) * (diff - 0.5)  # 同上图公式
            return tf.reduce_mean(smooth_l1_loss)

    def close(self):

        if self.training:
            self.writer.close()
        self.sess.close()

    def summary(self):

        with tf.name_scope('loss'):
            train_loss_summary = tf.summary.scalar('train', self.loss)
            valid_loss_summary = tf.summary.scalar('valid', self.loss)

        return train_loss_summary, valid_loss_summary

    def predict(self, imgs, batchSize = 20):
        numBatch = int(np.ceil(imgs.shape[0] / batchSize))
        predictions = []
        for iBatch in range(numBatch):
            fdVal = {self.inputs: imgs[iBatch*batchSize:(iBatch+1)*batchSize, ...],
                     self.is_training: False,
                     self.target_width: self.imgWidth,
                     self.target_height: self.imgHeight,
                     }

            output = self.sess.run(self.outputs_resized, feed_dict=fdVal)
            predictions.append(output)

        return np.concatenate(predictions, axis=0)
