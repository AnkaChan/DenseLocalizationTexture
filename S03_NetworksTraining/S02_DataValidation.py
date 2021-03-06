import cv2, glob, itertools, json
import numpy as np
from os.path import join
import os, tqdm
from pathlib import Path

import tensorflow as tf
from feature_extractor import MobileNet, Resnet, Vgg16
from modules import atrous_spatial_pyramid_pooling
from datetime import datetime

class UVExtractor(object):
    def __init__(self, base_architecture, training=True, ignore_label=0,batch_norm_momentum=0.9997, pre_trained_model=None, log_dir='./' ):

        self.outputStride = 4

        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.ignore_label = ignore_label
        self.inputs_shape = [None, 60, 60, 1]
        self.labels_shape = [None, 60, 60, 2]
        self.training = training
        self.inputs = tf.placeholder(tf.float32, shape=self.inputs_shape, name='inputs')
        self.labels = tf.placeholder(tf.uint8, shape=self.labels_shape, name='labels')

        self.target_height = tf.placeholder(tf.int32, None, name='target_image_height')
        self.target_width = tf.placeholder(tf.int32, None, name='target_image_width')

        self.weight_decay = tf.placeholder(tf.float32, None, name='weight_decay')
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
        self.batch_norm_momentum = batch_norm_momentum

        self.feature_map = self.backbone_initializer(base_architecture)
        if pre_trained_model:
            self.initialize_backbone_from_pretrained_weights(pre_trained_model)
        self.outputs = self.model_initializer()

        self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
        # self.loss = self.loss_initializer()
        # self.optimizer = self.optimizer_initializer()

        # Initialize tensorflow session
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.training:
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
            self.train_summaries, self.valid_summaries = self.summary()

    def backbone_initializer(self, base_architecture):

        with tf.variable_scope('backbone'):
            if base_architecture == 'vgg16':
                features = Vgg16(self.inputs, self.weight_decay, self.batch_norm_momentum)
            elif base_architecture.startswith('resnet'):
                n_layers = int(base_architecture.split('_')[-1])
                features = Resnet(n_layers, self.inputs, self.weight_decay, self.batch_norm_momentum, self.is_training, outputStride = self.outputStride)
            elif base_architecture.startswith('mobilenet'):
                depth_multiplier = float(base_architecture.split('_')[-1])
                features = MobileNet(depth_multiplier, self.inputs, self.weight_decay, self.batch_norm_momentum, self.is_training)
            else:
                raise ValueError('Unknown backbone architecture!')

        return features

    def model_initializer(self):

        pools = atrous_spatial_pyramid_pooling(inputs=self.feature_map, filters=256, regularizer=self.regularizer)
        logits = tf.layers.conv2d(inputs=pools, filters=2, kernel_size=(1, 1), name='logits')
        outputs = tf.image.resize_bilinear(images=logits, size=(self.target_height, self.target_width), name='resized_outputs')

        return outputs

    # def loss_initializer(self):
    #
    #     labels_linear = tf.reshape(tensor=self.labels, shape=[-1])
    #     not_ignore_mask = tf.to_float(tf.not_equal(labels_linear, self.ignore_label))
    #     # The locations represented by indices in indices take value on_value, while all other locations take value off_value.
    #     # For example, ignore label 255 in VOC2012 dataset will be set to zero vector in onehot encoding (looks like the not ignore mask is not required)
    #     # onehot_labels = tf.one_hot(indices=labels_linear, depth=, on_value=1.0, off_value=0.0)
    #
    #     # loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=tf.reshape(self.outputs, shape=[-1, self.num_classes]), weights=not_ignore_mask)
    #
    #     return loss

    def initialize_backbone_from_pretrained_weights(self, path_to_pretrained_weights):

        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=['global_step'])
        valid_prefix = 'backbone/'
        tf.train.init_from_checkpoint(path_to_pretrained_weights, {v.name[len(valid_prefix):].split(':')[0]: v for v in variables_to_restore if v.name.startswith(valid_prefix)})

    def close(self):

        if self.training:
            self.writer.close()
        self.sess.close()


if __name__ == '__main__':
    dataFolder = r'F:\WorkingCopy2\2021_RandomlyDeformedGridMesh\Data'
    numMarkers = 50

    uvExtractor = UVExtractor('resnet_50', training=False,)
    print(uvExtractor.outputs.shape)

    for iM in range(numMarkers):
        outImgDataFile = join(dataFolder, 'ImgMarker_' + str(iM).zfill(3) + '.npy')
        outUVDataFile = join(dataFolder, 'UVMarker_' + str(iM).zfill(3) + '.npy')

        img = np.load(outImgDataFile)
        uv = np.load(outUVDataFile)

        print(img.shape)
        print(uv.shape)

        for i in range(50):
            iImg = np.random.randint(0, img.shape[0])
            cv2.imshow('crop', img[iImg, :, :])
            cv2.imshow('u', uv[iImg, :,:,0])
            cv2.imshow('v', uv[iImg, :,:,1])
            cv2.waitKey()
