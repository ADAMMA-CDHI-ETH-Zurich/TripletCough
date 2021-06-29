#!/usr/bin/env python
# -*- coding: utf-8 -*-

# N-Way K-Shot Testing
# Authors: Stefan Jokic, Andrea Schwaller, Filipe Barata, David Cleres

# DESCRIPTION
# This script loads mel-spectrogram data (test) from a pickle file,
# and tests a triplet network model (i.e., computes one-shot test accuracies)

# Libraries & Seed
import numpy as np

np.random.seed(0)

import os
import pickle
import time
import numpy.random as rng
import sys
import tensorflow as tf
import tensorflow.keras.backend as K

# Parameters
# Mel-scaled spectrogram parameters
img_width = 237
img_height = 80

# Testing
N_way = int(sys.argv[3])  # Number of classes used for few-shot tasks, set manually if not ran as python script
k = int(sys.argv[4])  # Number of samples per class, set manually if not ran as python script
n_tasks = 125

# Model & Training

# Directories
weights = sys.argv[1]
data = sys.argv[2]

data_split_name = data.split("/")
data_name = data_split_name[0] + "_" + data_split_name[1]
data_path = "%s%s/" % ("./data/", data)
weights_path = "%s%s/" % ("./weights_testing/", weights)
testing_path = "./testing/"

# Load Testing Data from Pickle File
with open(os.path.join(data_path, "test.pickle"), "rb") as f:
    (X_test, n_samples_test) = pickle.load(f)

# Load Model
from Model import get_triplet_network, get_embedding_cnn

model = get_triplet_network((img_height, img_width, 1))
embedding_cnn = get_embedding_cnn((img_height, img_width, 1))


# Generate Few-shot Tasks
# Creates pairs consisting of test image and support set for generating N-way-k-shot tasks
def make_kshot_task(N, k, X, n_samples):

    n_participants, n_examples, h, w = X.shape

    # Randomly choose N participants
    N_participants = rng.choice(n_participants, size=(N,), replace=False)

    # First select k+1 samples u.a.r. from P1 (first of N randomly chosen participants)
    P1_samples = rng.choice(range(n_samples[N_participants[0]]), size=(k + 1,), replace=False)
    # Select first sample from P1 as test sample
    P1_test_sample = P1_samples[0]
    # Select the other k samples (different from the test sample) from P1 as anchor samples
    P1_anchor_samples = P1_samples[1:]

    # Retrieve images from test and anchor samples of P1
    test_img = X[N_participants[0], P1_test_sample, :, :].reshape(1, h, w, 1)
    P1_anchor_imgs = X[N_participants[0], P1_anchor_samples, :, :].reshape(k, h, w, 1)

    # Select k samples u.a.r. from each of the remaining N-1 participants as anchor samples
    # and retrieve the corresponding images
    rest_anchor_imgs = []
    for participant in N_participants[1:]:
        sample_set = rng.choice(range(n_samples[participant]), size=(k,), replace=False)
        rest_anchor_imgs.append(X[participant, sample_set, :, :].reshape(k, h, w, 1))

    # Return test image, set of anchor images of P1 and list containing set of images of each of the remaining
    # N-1 participants
    task = [test_img, P1_anchor_imgs, rest_anchor_imgs]
    return task


# Tests average N-way-k-shot learning accuracy of a triplet network over n k-shot tasks
def test_kshot(model, n, N, k):
    n_correct = 0
    print("\n ------------- ")
    print("Evaluating model on {} random {} way {}-shot learning tasks ...".format(n, N, k))

    for i in range(n):
        inputs = make_kshot_task(N, k, X_test, n_samples_test)
        embedding_test_img = embedding_cnn.predict(inputs[0])
        embedding_P1_anchor_imgs = embedding_cnn.predict(inputs[1])

        # Compute mean distance between test sample and anchor samples of first participant P1
        distances_P1_anchor_imgs = []
        for emb in embedding_P1_anchor_imgs:
            distances_P1_anchor_imgs.append(K.sum(K.square(embedding_test_img - emb), axis=1))

        mean_tensor = tf.keras.metrics.MeanTensor()

        for j in range(len(distances_P1_anchor_imgs)):
            mean_tensor.update_state(distances_P1_anchor_imgs[j])

        mean_distance_P1 = mean_tensor.result()

        # Compute mean distance between test sample and anchor samples of each of the N-1 other participants
        embedding_rest_anchor = []

        for img_set in inputs[2]:
            embedding_rest_anchor.append(embedding_cnn.predict(img_set))

        mean_distances_rest = []
        for emb_set in embedding_rest_anchor:
            distances = []
            for emb in emb_set:
                distances.append(K.sum(K.square(embedding_test_img - emb), axis=1))

            m = tf.keras.metrics.MeanTensor()

            for j in range(len(distances)):
                m.update_state(distances[j])

            mean_distances_rest.append(m.result())

        correct = True
        for participant_mean_distance in mean_distances_rest:
            if participant_mean_distance < mean_distance_P1:
                correct = False
                break

        if correct:
            n_correct += 1
            print("Task {} correctly predicted.".format(i))

    percent_correct = 100.0 * n_correct / n
    print("Got an average of {}% {} way {}-shot validation accuracy \n".format(percent_correct, N, k))
    return percent_correct


# Test Model
print("\n-------------------------------------")
print("Start testing process!")
print("-------------------------------------\n")

# Print parameter configuration
print("Parameter Configuration:\n")
print("Script Filename:", sys.argv[0])
print("Data Path:", sys.argv[1])
print("Weights Path:", sys.argv[2])
print("\n ------------- \n")

# Loop over all weight files
timestamp = time.strftime("%Y%m%d_%H%M%S__")
os.makedirs(os.path.join(testing_path, timestamp), exist_ok=True)
for weight_file in os.listdir(weights_path):
    np.random.seed(0)

    # Load weights
    model.load_weights(os.path.join(weights_path, weight_file))
    embedding_cnn.set_weights(model.get_weights())

    # Compute N-way one-shot test accuracy for 10'000 random trials
    acc_model = test_kshot(model, n_tasks, N_way, k)

    # Print Output
    print("Model Weights: ", weight_file)
    print("Model Accuracy: ", acc_model)

    file = open(
        "%s%s%s%s%s%s%s%s%s%s%s%s%s"
        % (
            testing_path,
            timestamp,
            "train_",
            weights,
            "_test_",
            data_name,
            "_",
            str(n_tasks),
            "_N",
            str(N_way),
            "_k",
            str(k),
            "__testing.txt",
        ),
        "w",
    )
    file.write("Model Weights: {}\n".format(weight_file))
    file.write("Testing on {} {}-way-{}-shot tasks\n".format(n_tasks, N_way, k))
    file.write("Model Accuracy: {}".format(acc_model))
    file.close()
