#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Stefan Jokic, Andrea Schwaller, Filipe Barata, David Cleres
"""
The following articles provided a basis for the implementation:
"One Shot Learning with Siamese Networks using Keras" (September 2019)
- Link to Article: https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d
- Link to GitHub Repository: https://sorenbouma.github.io/blog/oneshot/

"One Shot Learning and Siamese Networks in Keras" (September 2019)
- Link to Article: https://sorenbouma.github.io/blog/oneshot/
- Link to GitHub Repository: https://github.com/sorenbouma/keras-oneshot

"Building a Speaker Identification System from Scratch with Deep Learning" (September 2019)
- Link to Article: https://medium.com/analytics-vidhya/building-a-speaker-identification-system-from-scratch-with-deep-learning-f4c4aa558a56
- Link to GitHub Repository: https://github.com/oscarknagg/voicemap
"""

# This script loads cough data from audio files,
# processes and converts them to mel-spectrograms,
# and stores them into pickle files.
# --> training, validation, test set

# Libraries
import os
import pickle
import librosa
import numpy as np

np.random.seed(0)
import numpy.random as rng

# Parameters
import Parameters as param  # "Parameters" must be the name of .py script containing parameters, c.f. params directory


# Load & Process Training, Validation & Test Data
# Extracts a window around the maximum of the signal
def extract_Signal_Of_Importance(signal, window, sample_rate):
    window_size = int(window * sample_rate)
    start = max(0, np.argmax(np.abs(signal)) - (window_size // 2))
    end = min(np.size(signal), start + window_size)
    signal = signal[start:end]
    length = np.size(signal)
    assert length <= window_size, "extracted signal is longer than the allowed window size"
    if length < window_size:
        signal = np.concatenate((signal, np.zeros(window_size - length)))  # Pad zeros to the signal if too short

    return signal


def standardize(signal):
    maxValue = np.max(signal)
    minValue = np.min(signal)
    signal = (signal - minValue) / (maxValue - minValue)
    return signal


# Converts audio to mel-spectrogram
def preprocess(sound_file, window, bands, hop_length):
    time_signal, sample_rate = librosa.load(sound_file, mono=True, res_type="kaiser_fast")
    time_signal = extract_Signal_Of_Importance(time_signal, window, sample_rate)
    time_signal = standardize(time_signal)
    mel_spec = librosa.feature.melspectrogram(
        y=time_signal, sr=sample_rate, power=1, n_mels=bands, hop_length=hop_length
    )
    return mel_spec


# Loads audio files of all participants from a directory into a tensor
#  Returns a tensor X (#participants, #samples, width, heigth) and a vector containing the number of samples per participant
def loadimgs(path, window, bands, hop_length, img_height, img_width, n_samples_max):
    X = []
    n_samples = []

    # Load every participant seperately
    for participant in os.listdir(path):
        print("Loading participant: ", participant)
        if participant != ".DS_Store":
            participant_path = os.path.join(path, participant)
            mel_spec_images = []

            # Read all/50 random audio files of the current participant and convert them to mel-spectrograms
            n_samples_participant = len(os.listdir(participant_path))
            if n_samples_participant > n_samples_max:
                filelist = rng.choice(os.listdir(participant_path), size=(n_samples_max,), replace=False)
                n_samples_participant = 500
            else:
                filelist = os.listdir(participant_path)

            for filename in filelist:
                if filename == ".DS_Store":
                    print("Filename: ", filename)
                audio_path = os.path.join(participant_path, filename)
                mel_spec = preprocess(audio_path, window=window, bands=bands, hop_length=hop_length)
                mel_spec_images.append(mel_spec)

            # Fill remaining empty image slots with NaN to have same amount of samples for each participant
            print(np.asarray(mel_spec_images).shape)
            print(np.asarray(np.full((n_samples_max - n_samples_participant, img_height, img_width), np.nan).shape))
            mel_spec_images = np.concatenate(
                (mel_spec_images, np.full((n_samples_max - n_samples_participant, img_height, img_width), np.nan))
            )

            X.append(mel_spec_images)
            n_samples.append(n_samples_participant)

    return np.asarray(X), n_samples


if __name__ == "__main__":
    # Load training data
    X_train, n_samples_train = loadimgs(
        param.train_folder,
        param.window,
        param.bands,
        param.hop_length,
        param.img_height,
        param.img_width,
        param.n_samples_max,
    )

    # Load validation data
    X_val, n_samples_val = loadimgs(
        param.val_folder,
        param.window,
        param.bands,
        param.hop_length,
        param.img_height,
        param.img_width,
        param.n_samples_max,
    )

    # Load test data
    X_test, n_samples_test = loadimgs(
        param.test_folder,
        param.window,
        param.bands,
        param.hop_length,
        param.img_height,
        param.img_width,
        param.n_samples_max,
    )

    # # Store Training, Validation & Test Data into Pickle Files

    with open(os.path.join(param.data_path, "train.pickle"), "wb") as f:
        pickle.dump((X_train, n_samples_train), f)

    with open(os.path.join(param.data_path, "val.pickle"), "wb") as f:
        pickle.dump((X_val, n_samples_val), f)

    with open(os.path.join(param.data_path, "test.pickle"), "wb") as f:
        pickle.dump((X_test, n_samples_test), f)

    with open(os.path.join(param.data_path, "train.pickle"), "rb") as f:
        (X_train, n_samples_train) = pickle.load(f)
    n_participants, n_examples, h, w = X_train.shape

    with open(os.path.join(param.data_path, "val.pickle"), "rb") as f:
        (X_val, n_samples_train) = pickle.load(f)
    n_participants, n_examples, h, w = X_val.shape
