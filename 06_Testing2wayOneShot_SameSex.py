#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 2-Way One-Shot Testing using only pairs of participants of same sex
# Authors: Stefan Jokic, Andrea Schwaller, Filipe Barata, David Cleres

# This script loads mel-spectrogram data (test) from a pickle file,
# and tests a triplet network model (i.e., computes k-shot test accuracies)

# Libraries & Seed

import numpy as np
np.random.seed(0)
import numpy.random as rng

import os
import pickle
from itertools import permutations
import sys
import statistics
import csv
import time
import tensorflow as tf
import tensorflow.keras.backend as K

# Parameters
# Mel-scaled spectrogram parameters
img_width = 237
img_height = 80

# Model & Training
# Directories
weights = sys.argv[1]
data = sys.argv[2]

data_split_name = data.split("/")
data_name = data_split_name[0] + "_" + data_split_name[1]
data_path = "%s%s/" % ("./data/", data)
weights_path = "%s%s/" % ("./weights_testing/", weights)
testing_path = "./testing/"

# Testing
k = 1  # Only one-shot tasks were considered
n_tasks = 125

# Load Model
from Model import get_triplet_network, get_embedding_cnn

model = get_triplet_network((img_height, img_width, 1))
embedding_cnn = get_embedding_cnn((img_height, img_width, 1))

# ## Load Testing Data from Pickle File

with open(os.path.join(data_path, "test.pickle"), "rb") as f:
    (X, n_samples) = pickle.load(f)

# Assign sex to participants in test set
# Voluntary cough data
gender_test = np.array(["f", "f", "f", "m", "f", "m", "f", "f", "f", "f"])

n_participants, n_examples, h, w = X.shape
male_participants = np.array(range(n_participants))[gender_test == "m"].tolist()
female_participants = np.array(range(n_participants))[gender_test == "f"].tolist()

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

# Set up csv output file
timestamp = time.strftime("%Y%m%d_%H%M%S__")
os.makedirs(os.path.join(testing_path, timestamp), exist_ok=True)
with open(
    "%s%s%s%s%s%s%s%s%s"
    % (testing_path, timestamp, "train_", weights, "_test_SameSex_", data_name, "_", n_tasks, "__testing.csv"),
    "w",
    newline="",
) as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)

    # Write header line to csv file
    permulation_list = list(permutations(male_participants, 2)) + list(permutations(female_participants, 2))
    permulation_list.insert(0, "Participant Combinations")
    writer.writerow(permulation_list)

    # Loop over all weight files
    for weight_file in os.listdir(weights_path):
        np.random.seed(0)

        # Load weights
        model.load_weights(os.path.join(weights_path, weight_file))
        embedding_cnn.set_weights(model.get_weights())

        acc_model = []
        count = 0
        # Loop over all participant combinations
        for participant_combo in list(permutations(male_participants, 2)) + list(permutations(female_participants, 2)):
            n_test = 0
            n_correct = 0

            # For each pair of participants, generate n_tasks (default: 100) random 2-way-k-shot tasks and
            # evaluate accuracy over these tasks
            for i in range(0, n_tasks):

                # First select k+1 samples u.a.r. from P1
                P1_samples = rng.choice(range(n_samples[participant_combo[0]]), size=(k + 1,), replace=False)
                # Select first sample from P1 as test sample
                P1_test_sample = P1_samples[0]
                # Select the other k samples (different from the test sample) from P1 as anchor samples
                P1_anchor_samples = P1_samples[1:]

                # Select k samples u.a.r. from P2 as anchor samples
                P2_anchor_samples = rng.choice(range(n_samples[participant_combo[1]]), size=(k,), replace=False)

                # Retrieve images from test and anchor samples
                test_img = X[participant_combo[0], P1_test_sample, :, :].reshape(1, img_height, img_width, 1)

                P1_anchor_imgs = X[participant_combo[0], P1_anchor_samples, :, :].reshape(k, img_height, img_width, 1)
                P2_anchor_imgs = X[participant_combo[1], P2_anchor_samples, :, :].reshape(k, img_height, img_width, 1)

                # Create support set composed of k P1 and k P2 anchor images
                support_set = np.concatenate((P1_anchor_imgs, P2_anchor_imgs), axis=0)

                # Test model
                # Compute embeddings for test sample and for each anchor sample in support set

                embedding_test_img = embedding_cnn.predict(test_img)
                embedding_support_set = embedding_cnn.predict(support_set)

                distances = []

                # Compute distances between embeddings of test sample and each anchor sample in support set
                for emb in embedding_support_set:
                    distances.append(K.sum(K.square(embedding_test_img - emb), axis=1))

                # Compute mean distance between test sample and anchor samples of P1 / test sample and anchor samples of P2
                m = tf.keras.metrics.MeanTensor()

                for i in range(0, k):
                    m.update_state(distances[i])

                P1_mean_distance = m.result()

                m.reset_states()

                for i in range(k, len(distances)):
                    m.update_state(distances[i])

                P2_mean_distance = m.result()

                if P1_mean_distance < P2_mean_distance:
                    n_correct += 1

            # Compute k-shot test accuracy for this participant combination
            acc = 100.0 * n_correct / n_tasks
            print("ACC: ", acc)
            acc_model.append(acc)
            count += n_tasks

        # Store and print output metrics
        metrics = []
        metrics.append(("Model Weights: ", weight_file))
        metrics.append(("Number of {}-Shot Tests: ".format(k), count))
        metrics.append(("Mean Accuracy: ", statistics.mean(acc_model)))
        metrics.append(("Standard Deviation Accuracy: ", statistics.stdev(acc_model)))
        metrics.append(("Min Accuracy: ", min(acc_model)))
        metrics.append(("Max Accuracy: ", max(acc_model)))

        print("Model accuracy: ", acc_model)

        for i in metrics:
            print(i)

        # Write to csv file
        acc_model.insert(0, weight_file)
        writer.writerow(acc_model)
        writer.writerow(metrics)
