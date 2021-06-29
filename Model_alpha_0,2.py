#!/usr/bin/env python
# coding: utf-8

# Authors: Stefan Jokic, Andrea Schwaller, Filipe Barata, David Cleres

"""
    The following article provided a basis for the implementation:
    "Lossless Triplet loss" (February 2018)
    - Link to article: https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
    - Link to python file on GitHub: https://gist.github.com/marcolivierarsenault/3be90c1977e53224811ae0faa5476da5#file-defaultsiamese-py
"""

# Libraries
import tensorflow as tf

tf.compat.v1.set_random_seed(0)

from keras.models import Sequential
from keras.layers import Conv2D, Activation, Input, concatenate
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense
import tensorflow.keras.backend as K
from keras.models import Model


# Define Sensitivity & Specificity Metrics
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


# Initialize Weights, Bias, Model
def triplet_loss(y_true, y_pred, N=4096, alpha=0.2):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """

    anchor = tf.convert_to_tensor(y_pred[:, 0:N])
    positive = tf.convert_to_tensor(y_pred[:, N : N * 2])
    negative = tf.convert_to_tensor(y_pred[:, N * 2 : N * 3])

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    margin_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(margin_loss, 0.0)
    return loss


def get_embedding_cnn(in_dims, out_dims=4096):
    """
    Base CNN to be shared.
    Model architecture based on 'Towards Device-Agnostic Mobile Cough Detection with Convolutional Neural Networks' (Barata et al., 2019)
    Implementation by David Cleres & Adjustments to architecture by Stefan Jokic.
    """
    model = Sequential()

    # Layer 1
    model.add(Conv2D(filters=16, kernel_size=(1, 10), input_shape=(in_dims)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))

    # Layer 2
    model.add(Conv2D(64, kernel_size=(1, 7)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))

    # Layer 3
    model.add(Conv2D(64, kernel_size=(1, 4)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))

    # Layer 4
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))

    # Layer 5
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))

    # Layer 6
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))

    # Layer 7
    model.add(Flatten())
    model.add(Dense(out_dims, activation="sigmoid", kernel_regularizer=l2(5e-4)))

    return model


def get_triplet_network(in_dims, out_dims=4096):
    # Network definition

    # Create the 3 inputs
    anchor_in = Input(shape=in_dims)
    pos_in = Input(shape=in_dims)
    neg_in = Input(shape=in_dims)

    # Share base network with the 3 inputs
    base_network = get_embedding_cnn(in_dims, out_dims)
    anchor_out = base_network(anchor_in)
    pos_out = base_network(pos_in)
    neg_out = base_network(neg_in)
    merged_vector = concatenate([anchor_out, pos_out, neg_out], axis=-1)

    # Define the trainable model
    model = Model(inputs=[anchor_in, pos_in, neg_in], outputs=merged_vector)

    return model
