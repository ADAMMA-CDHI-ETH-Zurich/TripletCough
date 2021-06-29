#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Verification Tasks

# Authors: Stefan Jokic, Andrea Schwaller, Filipe Barata, David Cleres

# Libraries
import numpy as np
import numpy.random as rng

import os
import pickle
import sys
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

n_enrollment = 10
n_test = 5

THRESHOLD = float(sys.argv[3])

# Load Model
from Model import get_triplet_network, get_embedding_cnn

model = get_triplet_network((img_height, img_width, 1))
embedding_cnn = get_embedding_cnn((img_height, img_width, 1))

# Load Testing Data from Pickle File
with open(os.path.join(data_path, "test.pickle"), "rb") as f:
    (X, n_samples) = pickle.load(f)

# Test Model via verification tasks
print("\n-------------------------------------")
print("Start testing process!")
print("-------------------------------------\n")

# Print parameter configuration
print("Parameter Configuration:\n")
print("Script Filename:", sys.argv[0])
print("Data Path:", sys.argv[1])
print("Weights Path:", sys.argv[2])
print("Selected threshold:", sys.argv[3])
print("\n ------------- \n")

# Set up csv output file
timestamp = time.strftime("%Y%m%d_%H%M%S__")
os.makedirs(os.path.join(testing_path, timestamp), exist_ok=True)
with open(
    "%s%s%s%s%s%s%s%s%s"
    % (
        testing_path,
        timestamp,
        "train_",
        weights,
        "_test_",
        data_name,
        "_threshold_",
        THRESHOLD,
        "_verification__testing.csv",
    ),
    "w",
    newline="",
) as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)

    # Write header line to csv file
    participant_list = list(range(len(n_samples)))
    participant_list.insert(0, "Participants")
    writer.writerow(participant_list)
    participant_list.pop(0)

    # Loop over all weight files
    for weight_file in os.listdir(weights_path):

        np.random.seed(0)

        # Load weights
        model.load_weights(os.path.join(weights_path, weight_file))
        embedding_cnn.set_weights(model.get_weights())

        tp, fp, tn, fn = 4 * [0]

        # Loop over all participants
        for participant in participant_list:

            print("Current participant to be verified: ", participant)

            P1_num_enrollment = min(n_enrollment, n_samples[participant] - n_test)

            # First select (P1_num_enrollment + n_test) samples u.a.r. from P1
            P1_samples = rng.choice(range(n_samples[participant]), size=(P1_num_enrollment + n_test,), replace=False)
            # Select first P1_num_enrollment samples from P1 as enrollment samples
            P1_enrollment_samples = P1_samples[0:P1_num_enrollment]
            # Select the remaining n_test samples (different from the enrollment samples) from P1 as test samples
            P1_test_samples = P1_samples[P1_num_enrollment:]

            print(
                "For the participant (ID {userID}) to be verified, the number of enrollment samples is: ".format(
                    userID=participant
                ),
                P1_num_enrollment,
            )
            print(
                "For the participant (ID {userID}) to be verified, the following enrollment samples have been selected: ".format(
                    userID=participant
                ),
                P1_enrollment_samples,
            )
            print(
                "For the participant (ID {userID}) to be verified, the following test samples have been selected: ".format(
                    userID=participant
                ),
                P1_test_samples,
            )

            writer.writerow(
                [
                    "For the participant (ID {userID}) to be verified, the number of enrollment samples is: ".format(
                        userID=participant
                    ),
                    P1_num_enrollment,
                ]
            )
            writer.writerow(
                [
                    "For the participant (ID {userID}) to be verified, the following enrollment samples have been selected: ".format(
                        userID=participant
                    ),
                    P1_enrollment_samples,
                ]
            )
            writer.writerow(
                [
                    "For the participant (ID {userID}) to be verified, the following test samples have been selected: ".format(
                        userID=participant
                    ),
                    P1_test_samples,
                ]
            )

            # Select n_test participants u.a.r. that are different from P1 (same participant can be selected multiple times)
            participants = []
            for i in range(n_test):
                participants.append((participant + rng.randint(1, len(n_samples))) % len(n_samples))

            # Count how many times each participant has been selected
            participants_count = {i: participants.count(i) for i in np.unique(participants)}

            print(
                "The following participants and number of samples per participant have been selected for the verification test: "
            )
            print(participants_count)

            writer.writerow(
                [
                    "The following participants and number of samples per participant have been selected for the verification test: ",
                    participants_count,
                ]
            )

            test_samples = []
            for p in participants_count.keys():
                num_samples = participants_count[p]
                samples = rng.choice(range(n_samples[p]), size=(num_samples,), replace=False)
                print(
                    "For participant {userID}, the following test samples have been selected: ".format(userID=p),
                    samples,
                )
                writer.writerow(
                    [
                        "For participant {userID}, the following test samples have been selected: ".format(userID=p),
                        samples,
                    ]
                )
                imgs = X[p, samples, :, :].reshape(num_samples, img_height, img_width, 1)

                test_samples.append(imgs)

            test_samples = np.concatenate(test_samples)

            # Retrieve images from P1
            P1_enrollment = X[participant, P1_enrollment_samples, :, :].reshape(
                P1_num_enrollment, img_height, img_width, 1
            )

            P1_test = X[participant, P1_test_samples, :, :].reshape(n_test, img_height, img_width, 1)

            # Compute embeddings
            emb_P1_enrollment = embedding_cnn.predict(P1_enrollment)
            emb_P1_test = embedding_cnn.predict(P1_test)
            emb_others = embedding_cnn.predict(test_samples)

            m = tf.keras.metrics.MeanTensor()

            print()

            for test_sample in emb_P1_test:
                for enrollment_sample in emb_P1_enrollment:
                    distance = K.sum(K.square(test_sample - enrollment_sample), axis=0)
                    m.update_state(distance)
                mean_distance = m.result()
                print(
                    "Mean distance between enrollment samples of participant {userID} and sample from same participant: ".format(
                        userID=participant
                    ),
                    mean_distance,
                )
                if mean_distance < THRESHOLD:
                    tp += 1
                else:
                    fn += 1
                m.reset_states()

            for test_sample in emb_others:
                for enrollment_sample in emb_P1_enrollment:
                    distance = K.sum(K.square(test_sample - enrollment_sample), axis=0)
                    m.update_state(distance)
                mean_distance = m.result()
                print(
                    "Mean distance between enrollment samples of participant {userID} and sample from other participant: ".format(
                        userID=participant
                    ),
                    mean_distance,
                )
                if mean_distance < THRESHOLD:
                    fp += 1
                else:
                    tn += 1
                m.reset_states()

        print()
        print("----------")
        print()

        # Store and print output metrics
        metrics = []
        metrics.append(("Model Weights: ", weight_file))
        writer.writerow(metrics[0])

        # For each threshold: Compute accuracy, FAR and FRR over all participants
        acc = (tp + tn) / (tp + tn + fp + fn)
        far = fp / (fp + tn)
        frr = fn / (fn + tp)

        metrics.append(("Threshold: ", THRESHOLD, "Accuracy: ", acc, "FAR: ", far, "FRR: ", frr))
        writer.writerow(metrics[1])
        print(metrics[1])
