#!/usr/bin/env python
# coding: utf-8

import os

# # Parameters

# Directories
data_path = "./data/Rode/close/"
train_folder = os.path.join(data_path, "train")
val_folder = os.path.join(data_path, "val")
test_folder = os.path.join(data_path, "test")

weights_path = "./weights_testing/"
summary_path = "./summaries/"


# Audio processing
## Mel-scaled spectrogram parameters
window = 1.2
bands = 80
hop_length = 112

## Spectrogram dimensions
img_width = 237
img_height = bands

n_samples_max = 25  # max number of samples of a participant (close/distant: 25, both: 50)

# Model & Training
learning_rate = 1e-2
batch_size = 8

n_iter = 5000  # number of training iterations
evaluate_every = 100  # interval for evaluating on one-shot tasks
N_way = 2  # how many classes for testing one-shot tasks
k = 1  # how many samples per class (k-shot tasks for validation)
n_val = 50  # how many few-shot tasks to validate on
