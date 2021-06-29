#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Triplet Networking Training & Triplet Mining
# Authors: Stefan Jokic, Andrea Schwaller, Filipe Barata, David Cleres

# This script loads mel-spectrogram data (training and validation) from pickle files,
# trains and validates a triplet network,
# logs training/validation loss to observe training progress via tensorboard,
# and saves model weights based on optimal validation loss
# -

# Libraries & Seed
import numpy as np

np.random.seed(0)

import os
import pickle
import time
import numpy.random as rng
import sys
import importlib

from keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K

# Parameters
# import Parameters_10 as param # Comment if executed as python script
param = importlib.import_module(sys.argv[1], package=None)  # Uncomment if executed as python script

# Load Training and Validation Data from Pickle Files
with open(os.path.join(param.train_folder, "train.pickle"), "rb") as f:
    (X_train, n_samples_train) = pickle.load(f)

with open(os.path.join(param.val_folder, "val.pickle"), "rb") as f:
    (X_val, n_samples_val) = pickle.load(f)

# Initialize Weights, Bias, Model
from Model import get_triplet_network, triplet_loss, get_embedding_cnn

model = get_triplet_network((param.img_height, param.img_width, 1))
optimizer = Adam(lr=param.learning_rate)
model.compile(loss=triplet_loss, optimizer=optimizer)

embedding_cnn = get_embedding_cnn((param.img_height, param.img_width, 1))


# Random batch generation (Mini-batch)
# Creates a batch of n triplets
def get_batch_random(batch_size, X, n_samples):

    n_participants, n_examples, h, w = X.shape

    # Initialize 3 empty arrays for the input image batch
    triplets = [np.zeros((batch_size, h, w, 1)) for i in range(3)]

    # Randomly sample several participants to use in the batch
    selected_participants = rng.choice(n_participants, size=(batch_size,), replace=True)

    for i in range(batch_size):
        # Select first participant
        participant_1 = selected_participants[i]

        # Select second participant : Add a random number to the category modulo n classes to ensure 2nd image has a different category
        participant_2 = (participant_1 + rng.randint(1, n_participants)) % n_participants

        # Select anchor sample from first participant u.a.r.
        sample_anchor = rng.randint(0, n_samples[participant_1])

        # Select positive sample from same participant u.a.r. and make sure that the sample is different from anchor
        sample_positive = (sample_anchor + rng.randint(1, n_samples[participant_1])) % n_samples[participant_1]

        # Select negative sample from second participant u.a.r.
        sample_negative = rng.randint(0, n_samples[participant_2])

        triplets[0][i, :, :, :] = X[participant_1, sample_anchor].reshape(h, w, 1)
        triplets[1][i, :, :, :] = X[participant_1, sample_positive].reshape(h, w, 1)
        triplets[2][i, :, :, :] = X[participant_2, sample_negative].reshape(h, w, 1)

    return triplets


# Batch generation with triplet mining
"""
The following article provided a basis for the implementation:
- Link to Article: https://medium.com/@crimy/one-shot-learning-siamese-networks-and-triplet-loss-with-keras-2885ed022352
- Link to python file on GitHub: https://gist.github.com/CrimyTheBold/9fd074e1bd089b4629a3e15e5e321291#file-tripletloss-hardbatch-py
"""


def get_batch_tripletmining(model, batch_size, X, n_samples):

    if batch_size < 32:
        draw_batch_size = 50
    else:
        draw_batch_size = 2 * batch_size

    # Initialize a vector for targets (These won't actually be used, but has to be defined in Keras)
    targets = np.zeros((batch_size,))

    # Pick a random batch of triplets for which we will evaluate the loss
    randombatch = get_batch_random(draw_batch_size, X, n_samples)

    randombatchloss = np.zeros(draw_batch_size)

    # Compute embeddings for anchors, positive and negative
    A = model.predict(randombatch[0])
    P = model.predict(randombatch[1])
    N = model.predict(randombatch[2])

    # Compute the loss with current network : d(A,P)-d(A,N). Alpha can be omitted here.
    randombatchloss = np.sum(np.square(A - P), axis=1) - np.sum(np.square(A - N), axis=1)

    # Sort by loss (largest loss first) and take the first batch_size/2 triplets
    selection = np.argsort(randombatchloss)[::-1][: batch_size // 2]

    # The rest of the samples in the batch will be selected u.a.r. from the previously generated random batch
    selection2 = np.random.choice(np.delete(np.arange(draw_batch_size), selection), batch_size // 2, replace=False)

    selection = np.append(selection, selection2)

    triplets = [
        randombatch[0][selection, :, :, :],
        randombatch[1][selection, :, :, :],
        randombatch[2][selection, :, :, :],
    ]

    return triplets, targets


# 2-way-k-shot Task
# Creates pairs of test image, support set for testing 2-way-k-shot learning.
def make_kshot_task(N, X, n_samples, k):

    n_participants, n_examples, h, w = X.shape

    # Randomly choose 2 participants
    N_participants = rng.choice(n_participants, size=(2,), replace=False)

    # First select k+1 samples u.a.r. from P1
    P1_samples = rng.choice(range(n_samples[N_participants[0]]), size=(k + 1,), replace=False)
    # Select first sample from P1 as test sample
    P1_test_sample = P1_samples[0]
    # Select the other k samples (different from the test sample) from P1 as anchor samples
    P1_anchor_samples = P1_samples[1:]

    # Select k samples u.a.r. from P2 as anchor samples
    P2_anchor_samples = rng.choice(range(n_samples[N_participants[0]]), size=(k,), replace=False)

    # Retrieve images from test and anchor samples
    test_img = X[N_participants[0], P1_test_sample, :, :].reshape(1, h, w, 1)

    P1_anchor_imgs = X[N_participants[0], P1_anchor_samples, :, :].reshape(k, h, w, 1)
    P2_anchor_imgs = X[N_participants[1], P2_anchor_samples, :, :].reshape(k, h, w, 1)

    # Create support set composed of k P1 and k P2 anchor images
    support_set = np.concatenate((P1_anchor_imgs, P2_anchor_imgs), axis=0)

    # Create input pair consisting of test sample and support set
    pairs = [test_img, support_set]

    return pairs


# Tests average 2-way-k-shot learning accuracy of a triplet network over n k-shot tasks
def test_kshot(model, N, n, k):

    n_correct = 0

    print("\n ------------- ")
    print("Evaluating model on {} random {} way {}-shot learning tasks ...".format(n, N, k))

    # Create input pairs of test image & support set and validate model
    for i in range(n):
        inputs = make_kshot_task(N, X_val, n_samples_val, k)
        embedding_test_img = model.predict(inputs[0])
        embedding_support_set = model.predict(inputs[1])

        distances = []

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

    percent_correct = 100.0 * n_correct / n

    print("Got an average of {}% {} way {}-shot validation accuracy \n".format(percent_correct, N, k))

    return percent_correct


if __name__ == "__main__":

    # Train & Validate Model
    print("\n-------------------------------------")
    print("Starting training process!")
    print("-------------------------------------\n")

    # Print parameter configuration
    print("Parameter Configuration:\n")
    print("Script Filename:", sys.argv[0])
    print("Parameter Filename:", sys.argv[1], "\n")
    print("window: ", param.window)
    print("bands: ", param.bands)
    print("hop_length: ", param.hop_length)
    print("img_width: ", param.img_width)
    print("img_height: ", param.img_height)
    print("n_samples_max: ", param.n_samples_max, "\n")
    print("learning_rate: ", param.learning_rate)
    print("batch_size: ", param.batch_size)
    print("n_iter: ", param.n_iter)
    print("evaluate_every: ", param.evaluate_every)
    print("N_way: ", param.N_way)
    print("n_val: ", param.n_val)

    # Create summary directory for this training run
    timestamp = time.strftime("%Y%m%d_%H%M%S__")
    summary_path_run = "%s/%s%s" % (param.summary_path, timestamp, sys.argv[1])
    os.makedirs(summary_path_run, exist_ok=True)
    writer = tf.summary.create_file_writer(summary_path_run)

    # Initialize
    t_start = time.time()
    best = sys.maxsize
    best_iteration = 0

    # Start training process
    with writer.as_default():
        for i in range(1, param.n_iter + 1):

            # Create training batch and train model on batch
            (inputs, targets) = get_batch_tripletmining(embedding_cnn, param.batch_size, X_train, n_samples_train)
            loss = model.train_on_batch(inputs, targets)

            tf.summary.scalar("loss", loss, step=i)
            writer.flush()

            # Validate model from time to time
            if i % param.evaluate_every == 0:

                # Create validation batch and validate model on batch
                inputs = get_batch_random(param.n_val, X_val, n_samples_val)
                targets = np.zeros((param.n_val,))
                val_loss = model.test_on_batch(inputs, targets)

                # Save summary to observe validation progress on tensorboard
                tf.summary.scalar("validation loss", val_loss, step=i)
                writer.flush()

                embedding_cnn.set_weights(model.get_weights())
                # Validate model on k-shot tasks and save summary to observe progress of k-shot validation accuracy
                val_acc = test_kshot(embedding_cnn, param.N_way, param.n_val, param.k)
                tf.summary.scalar("validation accuracy", val_acc, step=i)
                writer.flush()

                print("Time for {0} iterations: {1} mins".format(i, (time.time() - t_start) / 60.0))
                print("Train Loss: {0}".format(loss))
                print("Validation loss: {0}".format(val_loss))
                if val_loss < best:
                    print(
                        "Current best validation loss: {0}, Previous best validation loss: {1}".format(val_loss, best)
                    )
                    best = val_loss
                    best_iteration = i
                    # Early stopping: Only save weights associated with lowest validation loss
                    print("Saving current best model weights ...")
                    model.save_weights(
                        os.path.join(param.weights_path, "%s%s%s" % (timestamp, sys.argv[1], "__weights.h5"))
                    )

    print("\n-------------------------------------")
    print("Training process completed!")
    print("Achieved optimal validation loss of {} after {} iterations.".format(best, best_iteration))
    print("-------------------------------------\n")
