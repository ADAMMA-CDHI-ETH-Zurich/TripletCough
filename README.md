<!-- #region -->
# TripletCough: Cougher Identification and Verification from Contact-Free Smartphone-Based Audio Recordings Using Metric Learning

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![DOI: XXX](https://zenodo.org/badge/doi/XXX/XXX.svg)](https://doi.org/XXX)   [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)


This repository contains the code to reproduce the experiments in the paper *TripletCough: Cougher Identification and Verification from Contact-Free Smartphone-Based Audio Recordings Using Metric Learning* by ([Stefan Jokic](https://github.com/sjokic), [Dr. sc. Filipe Barata](https://github.com/pipo3000))*, [David Cleres](https://github.com/dcleres), Frank Rassouli, Claudia Steurer-Stey, Milo A. Puhan, Martin Brutsche, and Elgar Fleisch.

<div style="text-align:center" width="100%">
    <img width="50%" src="https://i.ibb.co/6HVH4yv/Triplet-Network-1.png">
</div>

## 1 - How to use the pre-trained models
We have already trained our TripletCough network on a data set of voluntary coughs recorded using the RØDE NT1000 studio microphone, as described in the paper. We provide you with the pre-trained model file in `weights_testing/rode_close/20201108_065654__Parameters_rode_close_weights.h5`. The code used to evaluate this pre-trained model can be found in the  section "Python Scripts" which explains how to run the various python scripts available for performing the identification and verification tests that are reported in the paper.

### 1.1 - Verification Task
- To use the model to for **Verification**, please follow the following procedure:
  ```python
  # Libraries
    import os
    import numpy as np
    import tensorflow as tf
    import tensorflow.keras.backend as K

    # Define the relevant variables (number of enrollment and test samples, etc.)
    nb_participants = 20
    nb_enrollment_samples = 10
    nb_test_samples = 5

    # Define the expected size of the spectrogram
    spect_height = 80
    spect_width = 237

    # Load Model
    from Model import get_triplet_network, get_embedding_cnn
    model = get_triplet_network((spect_height, spect_width, 1))
    embedding_cnn = get_embedding_cnn((spect_height, spect_width, 1))

    # 20 participants with 10 enrollment samples each
    X_enrollment = np.random.rand(nb_participants, nb_enrollment_samples, spect_height, spect_width)
    n_samples_enrollment = [nb_enrollment_samples] * nb_participants

    # 20 participants with 5 test samples each
    X_test = np.random.rand(nb_participants, nb_test_samples, spect_height, spect_width)
    n_samples_test = [nb_test_samples] * nb_participants

    weights_file = os.path.join("weights_testing", "rode_close", "20201108_065654__Parameters_rode_close_weights.h5")

    # Load weights
    model.load_weights(weights_file)
    embedding_cnn.set_weights(model.get_weights())

    tp, fp, tn, fn = 4 * [0]

    for participant in range(nb_participants):
        # Select the remaining n_test samples (different from the enrollment samples) from P1 as test samples
        remaining_participant_list = list(range(nb_participants))
        remaining_participant_list.remove(participant)

        # Retrieve images from P1
        P1_enrollment = X_enrollment[participant, :, :, :].reshape(
        nb_enrollment_samples, spect_height, spect_width, 1
        )
        P1_test = X_test[participant, :, :, :].reshape(nb_test_samples, spect_height, spect_width, 1)

        test_samples = X_test[remaining_participant_list, :, :, :].reshape(len(remaining_participant_list * nb_test_samples), spect_height, spect_width, 1)

        # Compute the embeddings
        emb_P1_enrollment = embedding_cnn.predict(P1_enrollment)
        emb_P1_test = embedding_cnn.predict(P1_test)
        emb_others = embedding_cnn.predict(test_samples)

        m = tf.keras.metrics.MeanTensor()

        for test_sample in emb_P1_test:
            for enrollment_sample in emb_P1_enrollment:
                distance = K.sum(K.square(test_sample - enrollment_sample), axis=0)
                m.update_state(distance)
            mean_distance = m.result()
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
            if mean_distance < THRESHOLD:
                fp += 1
            else:
                tn += 1
            m.reset_states()

    acc = (tp + tn) / (tp + tn + fp + fn)
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
  ```
  - More details on how to compute the distance metrics can be find in the file `07_VerificationTasks.py`. 

  -  **!!DISCLAIMER!!**: the models was trained on reflex and voluntary coughs from the data sets described in the paper. It is possible that by using coughs from a different disease,  recording devices, participants or environmental conditions, the performance from the model can be different to the ones reported in the paper.

### 1.2 - Identification Task
- To use the model for **Identification**, please follow the following procedure to use the model for a 2-way 4-shot evaluation:
  ```python

  from Model import get_triplet_network, get_embedding_cnn
  from itertools import permutations
  import os
  import numpy as np
  import tensorflow as tf
  import tensorflow.keras.backend as K
  import numpy.random as rng

  nb_participants = 20
  nb_enrollment_samples = 10
  nb_test_samples = 5
  spect_height = 80
  spect_width = 237

  # Identification task
  n_tasks = 125
  k = 4 # 4-shot evaluation

  # Load Model
  model = get_triplet_network((spect_height, spect_width, 1))
  embedding_cnn = get_embedding_cnn((spect_height, spect_width, 1))

  # Generate the Testing Data from Pickle File
  # 20 participants with 10 enrollment samples each
  X = np.random.rand(nb_participants, nb_enrollment_samples, spect_height, spect_width)
  n_samples = [nb_enrollment_samples] * nb_participants

  # Write header line to csv file
  permutation_list = list(permutations(range(X.shape[0]), 2))
  permutation_list.insert(0, "Participant Combinations")

  # Loaded the pre-trained weight file
  weights_path = os.path.join("./weights_testing/rode_close/20201108_065654__Parameters_rode_close_weights.h5")
  np.random.seed(0)

  # Load weights
  model.load_weights(weights_path)
  embedding_cnn.set_weights(model.get_weights())

  acc_model = []
  count = 0
  # Loop over all participant combinations
  for participant_combo in permutations(range(X.shape[0]), 2):
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
          test_img = X[participant_combo[0], P1_test_sample, :, :].reshape(1, spect_height, spect_width, 1)

          P1_anchor_imgs = X[participant_combo[0], P1_anchor_samples, :, :].reshape(k, spect_height, spect_width, 1)
          P2_anchor_imgs = X[participant_combo[1], P2_anchor_samples, :, :].reshape(k, spect_height, spect_width, 1)

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
      
  mean_acc = sum(acc_model)/len(acc_model)
  ```

  - A more detailed implementation can be found in the python script: `03_2wayFewShotTesting.py`.

  - **!!DISCLAIMER!!**: the models was trained on reflex and voluntary coughs from the data sets described in the paper. It is possible that by using coughs from a different disease,  recording devices, participants or environmental conditions, the performance from the model can be different to the ones reported in the paper.

## 2 - Repository structure

- `params` : This directory contains all parameter files required for training the model using different hyperparameters (batch size, learning rate) and audio recordings from different devices

- `summaries`: This directory will contain the summary files storing the training progress, which can be viewed in a `tensorboard` instance. Each time the model is trained, a summary file will be stored in this directory. If this directory does not exist, it must be created.

- `testing`: This directory will contain the output .csv files containing the result of testing the model via N-way K-shot classification tasks. Each time the model is tested, a .csv will be generated and stored in this directory. This directory already contains the results of various testing scenarios.

- `weights`: When training the model, the weights associated with the optimal validation loss are stored in this directory.

- `weights_testing`: When testing the model, the weights stored in this directory will be used. It already contains the weight files of the model trained in different scenarios (different hyperparameters and learning rates, different recording devices)

- `dataFiles`: Directory should contain the raw .wav files of coughs. The name of the directory can be changed and adjusted accordingly in the parameters file. **NOTE**: The data in this folder are not publicly available to protect the privacy of the study's participants.

- `data`: Directory contains the processed .pickle files containing the mel-scaled spectrograms of the coughs from `dataFiles`, which are then split into training, validation, and testing data. The name of the directory can be changed and adjusted accordingly in the parameters file. **NOTE**: The entire data set of the paper should be in this folder. **However**, due to privacy constraints we could not publicly share the data as we have to protect the privacy of the study participants. The demo data we included is in the pickle format and was generated with the following code:

  ```python
  import pickle
  import os 

  nb_participants = 10
  nb_enrollment_samples = 10
  nb_test_samples = 5
  spect_height = 80
  spect_width = 237

  X_train_tutorial = np.random.rand(nb_participants, nb_enrollment_samples + nb_test_samples, spect_height, spect_width)
  n_samples_train_tutorial = [nb_enrollment_samples + nb_test_samples] * nb_participants
  X_val_tutorial = np.random.rand(nb_participants, nb_enrollment_samples + nb_test_samples, spect_height, spect_width)
  n_samples_val_tutorial = [nb_enrollment_samples + nb_test_samples] * nb_participants
  X_test_tutorial = np.random.rand(nb_participants, nb_enrollment_samples + nb_test_samples, spect_height, spect_width)
  n_samples_test_tutorial = [nb_enrollment_samples + nb_test_samples] * nb_participants

  # Save the data to pickle
  # Store Training, Validation & Test Data into Pickle Files
  with open(os.path.join(data_path, "train", "train.pickle"), "wb") as f:
      pickle.dump((X_train_tutorial, n_samples_train_tutorial), f)

  with open(os.path.join(data_path, "val", "val.pickle"), "wb") as f:
      pickle.dump((X_val_tutorial, n_samples_val_tutorial), f)

  with open(os.path.join(data_path, "test", "test.pickle"), "wb") as f:
      pickle.dump((X_test_tutorial, n_samples_test_tutorial), f)
      
  ```

  You can read the file with the following command in the root directory:<br>
  
  ```python
  import pickle
  import os 

  # Load Training Data from Pickle Files
  with open(os.path.join("./data/Rode/close/train", "train.pickle"), "rb") as f:
      (X_train, n_samples_train) = pickle.load(f)
  ```

  In this case `X_train` as a shape of `(3, 4, 80, 237)` where the dimension 0 represent the participant (here 3 participants), dimension 1 is the number of samples from this given participant. Finally, dimension 2 and 3 represent the dimension of the mel-scaled spectrogram with 80 bands. The `n_samples_train_tutorial` variable contains the number of samples from each participant. Here in this case, we included 3 samples per participants however, in the real data set the `n_samples_train_tutorial` looked like: `[10, 21, 19, 23, 23, 17, 14, 15, 19, 15, 24, 17, 16, 16, 18, 11, 25, 19]` as not all the participants had the same number of samples.
  The test and validation datasets contain also 3 participants each with four samples per patient. The test sample is used to determine whether a cough was emitted from a female or male participant. In the array contained on the pickle the first two participants are female and the third one was a male participant. 
    
    For example, the `data` directory should have the following format:

    ```bash
    ├── README.md
    └── Rode
        └── close
            ├── test
            │   └── test.pickle
            ├── train
            │   └── train.pickle
            └── val
                └── val.pickle
    ```

## 3 - Parameters File

When running any of the scripts, the path to a parameters file must be specified. These are located in the `params` directory.

The structure of the parameters file is as follows:
- Lines 7-9 (train_folder, val_folder, test_folder) contain the paths to the directories containing the raw .wav cough audio files. By default, these should be located in `dataFiles`. In particular, within `dataFiles` there must be a directory for each partition (train, validation, and test) and within each such directory, there must be a directory for each participant containing his/her associated cough .wav files (e.g. `./dataFiles/coughs_train/participant_01/`). Note that these paths are only relevant for the preprocessing pipeline and are disregarded during training.
- Line 11 contains the path where the mel-scaled spectrograms obtained by processing the raw .wav files from dataFiles will be stored. By default, this should be a directory within `data`. The preprocessing pipeline will generate three pickle files containing the mel-scaled spectrograms of each of the .wav files from the training, validation and testing partition, respectively.
- Line 12 contains the path to where the weights file should be stored after training.
- Lines 17-22 contain parameters for the preprocessing pipeline (In particular, the mel-scaled spectrogram parameters)
- The remaining lines contain relevant parameters for training, such as learning rate and batch size. You may ignore the parameters regarding testing, i.e., N and K for generating N-way K-shot classification tasks. These are passed as arguments to the command line when running the script.

## 4 - Model Architecture File

When running `02_ModelTrainingValidation_TripletMining.py` for training the network, the path to a file that contains the network architecture must be specified. The files containing the triplet network architecture (along with the associated CNN architecture) are: `Model.py`, `Model_alpha_0,1.py`, `Model_alpha_0,2.py`, `Model_alpha_0,5.py`.

In `Model.py` the gap parameter g of the triplet loss is set to 1.0.
In `Model_alpha_0,x.py` the gap parameter g of the triplet loss is set to x.

These files also contain a function that implements the hinge loss for triplets, also known as the triplet loss.

## 5 - Python Scripts

### 5.1. - 01a_DataPreprocessing_VoluntaryCoughs
- Script used for preprocessing the raw .wav files of voluntary coughs into mel spectrograms which are finally stored in .pickle files. As described in the `Parameters File` section, the .wav files should be contained within data and the output .pickle files will be stored in the `data` directory (by default). There will be a .pickle file for each of the training, testing and validation partitions, respectively.

### 5.2. - 01b_DataPreprocessing_ReflexCoughs
- Similar to `01a_DataPreprocessing_VoluntaryCoughs`, this script is used to preprocess the .wav files of reflex coughs into mel spectrograms. The sole difference between this script and the former is that the maximum number of used cough samples per participant is set to 500.

### 05.3. - 2_ModelTrainingValidation_TripletMining
- Run this script to train the triplet network. When doing so, you must specify a parameters file, e.g. run `python 02_ModelTrainingValidation_TripletMining.py Parameters_64_0,001`. By default, the parameters file must be located in the same directory as the python script. A simple triplet mining heuristic is employed to select a better set of triplets to train on than sampling them randomly. A model checkpoint is set such that only the weights associated with the lowest validation loss are saved. Weight files are stored in the `weights` directory by default.
  
  <br>

  To run this script, please do the following in the root directory of the repository: 
  ``` bash
  python ./02_ModelTrainingValidation_TripletMining.py parameter_package_import_name

  # Example:
  python ./02_ModelTrainingValidation_TripletMining.py params.Parameters_8_1e-2
  ```
  This assumes that you have a `params` file with all the relevant information inside. Templates of such files can be found in the `params/` directory.

### 5.4. - 03_2wayFewShotTesting
- Run this script to test the trained model via 2-way K-shot classification tasks. When doing so, you must specify a directory within `weights_testing` which contains the weight files to be used for the model, a directory within `data` containing the .pickle files associated with the mel spectrograms of the cough audio data, and K, the number of samples used for each of the 2 participants in the 2-way K-shot classification tasks. For instance, run: `python 03_TestingFewShot.py rode_close Rode/close/test 4`, where `rode_close` is a directory within `weights_testing` and contains .h5 weights files and `./data/Rode/close/test` must contain the `test.pickle`. The results of the testing will be printed and subsequently stored in a .csv file within the `testing` directory (The directory is created if it does not already exist).

  <br>

  In summary, to run the code, please do the following, please do the following in the root directory of the repository: 
  ``` bash
  python ./03_2wayFewShotTesting.py trained_weights_path data_folder_path K
  
  # Example:
  python ./03_2wayFewShotTesting.py rode_close Rode/close/test 4
  ```

### 5.5. - 04_NwayFewShotTesting
- Run this script to test the trained model via N-way K-shot classification tasks. When doing so, you must specify a directory within `weights_testing` which contains the weight files to be used for the model, a directory within `data` containing the .pickle files associated with the mel spectrograms of the cough audio data, the number of classes (coughers) N, and the number of samples K used for each of the N participants in the N-way K-shot classification tasks. For instance, run: `python 04_NwayFewShotTesting.py rode_close Rode/close/test 3 4`, where `rode_close` is be a directory within `weights_testing` and contains .h5 weights files and `./data/Rode/close/test` must contain `test.pickle`. The results of the testing will be printed and subsequently stored in a .csv file within `testing`. <br>
    In summary, to run the code, please do the following in the root directory of the repository: 
  ``` bash
  python ./04_NwayFewShotTesting.py trained_weights_path data_folder_path n_classes K
  
  # Example:
  python ./04_NwayFewShotTesting.py rode_close Rode/close/test 3 4
  ```

### 5.6. - 05_Testing2wayOneShot_DiffSex
- Run this script to test the trained model via 2-way one-shot classification tasks that only consider pairs of participants of opposite sex. When doing so, you must specify a directory within `weights_testing` which contains the weight files to be used for the model and a directory within `data` containing the .pickle files associated with the mel spectrograms of the cough audio data. For instance, run: `python 05_TestingFewShot_DiffSex.py rode_close Rode/close/test`, where `rode_close` is a directory within `weights_testing` and contains .h5 weights files and `./data/Rode/close/test` must contain `test.pickle`. The results of the testing will be printed and subsequently stored in a .csv file within `testing`.
  <br>

    In summary, to run the code, please do the following, please do the following in the root directory of the repository: 
  ``` bash
  python ./05_Testing2wayOneShot_DiffSex.py trained_weights_path data_folder_path
  
  # Example:
  python ./05_Testing2wayOneShot_DiffSex.py rode_close Rode/close/test
  ```

### 5.7. - 06_Testing2wayOneShot_SameSex
- Run this script to test the trained model via 2-way one-shot classification tasks that only consider pairs of participants of the same sex. When doing so, you must specify a directory within `weights_testing` which contains the weight files to be used for the model and a directory within `data` containing the .pickle files associated with the mel spectrograms of the cough audio data. For instance, run: `python 06_TestingFewShot_SameSex.py rode_close Rode/close/test`, where `rode_close` is a directory within `weights_testing` and contains .h5 weights files and `./data/Rode/close/test` must contain `test.pickle`. The results of the testing will be printed and subsequently stored in a .csv file within `testing`.
  <br>

    In summary, to run the code, please do the following in the root directory of the repository: 
  ``` bash
  python ./06_Testing2wayOneShot_SameSex.py trained_weights_path data_folder_path
  
  # Example:
  python ./06_Testing2wayOneShot_SameSex.py rode_close Rode/close/test
    ```


### 5.8. - 07_VerificationTasks
- Run this script to test the trained model regarding the verification task. When doing so, you must specify a directory within `weights_testing` which contains the weight files to be used for the model, a directory within `data` containing the .pickle files associated with the mel spectrograms of the cough audio data and, finally, the third argument is the verification threshold that should be applied. For instance, run: `python 07_VerificationTasks.py rode_close Rode/close/test`, where `rode_close` is a directory within `weights_testing` and contains .h5 weights files and `./data/Rode/close/test` must contain `test.pickle`. The results of the testing will be printed and subsequently stored in a .csv file within `testing`.<br>

  In summary, to run the code, please do the following in the root directory of the repository: 
  ```bash
  python ./07_VerificationTasks.py trained_weights_path data_folder_path threshold

  # Example:
  python ./07_VerificationTasks.py rode_close Rode/close/test 0.22
  ```

### 5.9. - 07b_VerificationTasks_GridSearch_Threshold
- Run this script to perform a grid search on the threshold used for the verification tasks. When doing so, you must specify a directory within `weights_testing` which contains the weight files to be used for the model and a directory within `data` containing the .pickle files associated with the mel spectrograms of the cough audio data. This time there is no third argument, as the script uses a grid search approach on the validation data to determine the best threshold. For instance, run: `python 07_VerificationTasks.py rode_close Rode/close/val`, where `rode_close` is a directory within `weights_testing` and contains .h5 weights files and `./data/Rode/close/val` must contain `val.pickle`. The result of the grid search, i.e. the best choice for the threshold, will be printed and subsequently stored in a .csv file within `testing`.<br>

  In summary, to run the code, please do the following in the root directory of the repository: 
  ```bash
  python ./07b_VerificationTasks_GridSearch_Threshold.py trained_weights_path val_data_folder_path

  # Example:
  python ./07b_VerificationTasks_GridSearch_Threshold.py rode_close  Rode/close/val
  ```

### 5.10. -  08_Compute_EER
- Run this script to compute the equal error rate (EER) of the verification tasks performed on the test set. This script runs verification tests with a set of different values for the threshold, in order to the determine the threshold that results in an equal false acceptance rate (FAR) and false rejection rate (FRR), i.e. the EER. 

  In summary, to run the code, please do the following in the root directory of the repository: 
  ```bash
  python ./08_Compute_EER.py trained_weights_path data_folder_path

  # Example:
  python ./08_Compute_EER.py rode_close  Rode/close/test
  ```


## 6 - Bash Scripts

`voluntary_split.sh` and `reflex_split.sh` are bash scripts that were used to generate the training, validation and test data splits for the voluntary and reflex cough data set respectively, by moving the corresponding raw .wav cough audio recordings into new directories. Within each of the directories `training`, `validation` and `testing`, a directory is created for each of the participants (coughers) containing their associated cough recordings (.wav).

## 7 - News
- July 2021: Paper submitted for review.

## 8 - Requirements
The following libraries are used:

- TensorFlow: `2.5.0`
- TensorBoard: `2.5.0`
- keras: `2.4.3`

We provide a requirements.txt file that can be used to create a conda environment to run the code in this repository. You can install the python packages listed in the requirements.txt using `pip` by running: 

```
pip install -r requirements.txt -U --no-cache-dir 
```

## 9 - Cite this Work
For now, please cite our Arxiv version:

Jokic, Stefan, and Barata, Filipe et al. "TripletCough: Cougher Identification and Verification from Contact-Free Smartphone-Based Audio Recordings Using Metric Learning" (2021).

```
@article{jokicbarata2021tripletCough,
  title={TripletCough: Cougher Identification and Verification from Contact-Free Smartphone-Based Audio Recordings Using Metric Learning},
  author={Jokic, Stefan and Barata, Filipe and David, Cleres and Rassouli, Frank and Steurer-Stey, Claudia and Puhan, Milo and Brutsche, Martin and Fleisch, Elgar},
  journal={XXX},
  year={2021}
}
```

## 10 - Core Team

[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/)


Chair of Information Management at D-​MTEC at ETH Zürich:
- [Dr. sc. Filipe Barata](https://github.com/pipo3000)
- [David Cleres](https://github.com/dcleres)
- [Stefan Jokic](https://github.com/sjokic)


*: Contributed equally.
