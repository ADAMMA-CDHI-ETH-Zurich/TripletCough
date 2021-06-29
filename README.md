<!-- #region -->
# TripletCough: Cougher Identification and Verification from Contact-Free Smartphone-Based Audio Recordings Using Metric Learning

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![DOI: XXX](https://zenodo.org/badge/doi/XXX/XXX.svg)](https://doi.org/XXX)   [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)






This repository contains the code to reproduce the experiments in the paper TripletCough: Cougher Identification and Verification from Contact-Free Smartphone-Based Audio Recordings Using Metric Learning by [Stefan Jokic](https://gitlab.ethz.ch/jokics), [Dr. sc. Filipe Barata](https://github.com/pipo3000), [David Cleres](https://github.com/dcleres), Frank Rassouli, Claudia Steurer-Stey, Milo A. Puhan, Martin Brutsche, and Elgar Fleisch.

<div style="text-align:center" width="100%">
    <img width="50%" src="https://i.ibb.co/6HVH4yv/Triplet-Network-1.png">
</div>

## Repository structure

- `params` : This directory contains all parameter files required for training the model using different hyperparameters (batch size, learning rate) and audio recordings from different devices

- `summaries`: This directory will contain the summary files storing the training progress, which can be viewed in a `tensorboard` instance. Each time the model is trained, a summary file will be stored in this directory. If this directory does not exist, it must be created.

- `testing`: This directory will contain the output .csv files containing the result of testing the model via N-way K-shot classification tasks. Each time the model is tested, a .csv will be generated and stored in this directory. This directory already contains the results of various testing scenarios.

- `weights`: When training the model, the weights associated with the optimal validation loss are stored in this directory.

- `weights_testing`: When testing the model, the weights stored in this directory will be used. It already contains the weight files of the model trained in different scenarios (different hyperparameters and learning rates, different recording devices)

- `dataFiles`: Directory should contain the raw .wav files of coughs. The name of the directory can be changed and adjusted accordingly in the parameters file. **NOTE**: The data in this folder are not publicly available to protect the privacy of the study's participants.

- `data`: Directory contains the processed .pickle files containing the mel-scaled spectrograms of the coughs from `dataFiles`, which are then split into training, validation, and testing data. The name of the directory can be changed and adjusted accordingly in the parameters file. **NOTE**: The data in this folder are not publicly available to protect the privacy of the study's participants. 
  
  For example, the `data` directory should have the following format:

  ```
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

## Parameters File

When running any of the scripts, the path to a parameters file must be specified. These are located in the `params` directory.

The structure of the parameters file is as follows:
- Lines 7-9 (train_folder, val_folder, test_folder) contain the paths to the directories containing the raw .wav cough audio files. By default, these should be located in `dataFiles`. In particular, within `dataFiles` there must be a directory for each partition (train, validation, and test) and within each such directory, there must be a directory for each participant containing his/her associated cough .wav files (e.g. `./dataFiles/coughs_train/participant_01/`). Note that these paths are only relevant for the preprocessing pipeline and are disregarded during training.
- Line 11 contains the path where the mel-scaled spectrograms obtained by processing the raw .wav files from dataFiles will be stored. By default, this should be a directory within `data`. The preprocessing pipeline will generate three pickle files containing the mel-scaled spectrograms of each of the .wav files from the training, validation and testing partition, respectively.
- Line 12 contains the path to where the weights file should be stored after training.
- Lines 17-22 contain parameters for the preprocessing pipeline (In particular, the mel-scaled spectrogram parameters)
- The remaining lines contain relevant parameters for training, such as learning rate and batch size. You may ignore the parameters regarding testing, i.e., N and K for generating N-way K-shot classification tasks. These are passed as arguments to the command line when running the script.

## Model File

When running `02_ModelTrainingValidation_TripletMining.py` for training the network, the path to a Model file must be specified. The Model files are: `Model.py`, `Model_alpha_0,1.py`, `Model_alpha_0,2.py`, `Model_alpha_0,5.py`.

In `Model.py` the gap parameter g of the triplet loss is set to 1.0.
In `Model_alpha_0,x.py` the gap parameter g of the triplet loss is set to x.

The Model file contains the triplet network architecture along with the CNN architecture that it employs. It also contains a function that implements the hinge loss for triplets, also known as the triplet loss.

## TripletCough Model 
The `TripletCough` model presented in the paper can be found in the following location: `weights_testing/rode_close/20201108_065654__Parameters_rode_close_weights.h5`. The code used to train and evaluate this model can be found in the next section which explain how to run the various python scripts available for testing the N way K shot tasks as well as the verification tasks that are reported in the paper.


## Python Scripts

### 01a_DataPreprocessing_VoluntaryCoughs
- Script used for preprocessing the raw .wav files of voluntary coughs into mel spectrograms which are finally stored in .pickle files. As described in the `Parameters File` section, the .wav files should be contained within data and the output .pickle files will be stored in the `data` directory (by default). There will be a .pickle file for each of the training, testing and validation partitions, respectively.

### 01b_DataPreprocessing_ReflexCoughs
- Similar to `01a_DataPreprocessing_VoluntaryCoughs`, this script is used to preprocess the .wav files of reflex coughs into mel spectrograms. The sole difference between this script and the former is that the maximum number of used cough samples per participant is set to 500.

### 02_ModelTrainingValidation_TripletMining
- Run this script to train the triplet network. When doing so, you must specify a parameters file, e.g. run `python 02_ModelTrainingValidation_TripletMining.py Parameters_64_0,001`. By default, the parameters file must be located in the same directory as the python script. A simple triplet mining heuristic is employed to select a better set of triplets to train on than sampling them randomly. A model checkpoint is set such that only the weights associated with the lowest validation loss are saved. Weight files are stored in the `weights` directory by default.
  
  <br>

  To run this script, please do the following in the root directory of the repository: 
  ``` bash
  python ./02_ModelTrainingValidation_TripletMining.py parameter_package_import_name

  # Example:
  python ./02_ModelTrainingValidation_TripletMining.py params.Parameters_8_1e-2
  ```
  This assumes that you have a `params` file with all the relevant information inside. Templates of such files can be found in the `params/` directory.

### 03_2wayFewShotTesting
- Run this script to test the trained model via 2-way K-shot classification tasks. When doing so, you must specify a directory within `weights_testing` which contains the weight files to be used for the model, a directory within `data` containing the .pickle files associated with the mel spectrograms of the cough audio data, and K, the number of samples used for each of the 2 participants in the 2-way K-shot classification tasks. For instance, run: `python 03_TestingFewShot.py rode_close Rode/close/test 4`, where `rode_close` is a directory within `weights_testing` and contains .h5 weights files and `./data/Rode/close/test` must contain the `test.pickle`. The results of the testing will be printed and subsequently stored in a .csv file within the `testing` directory (The directory is created if it does not already exist).

  <br>

  In summary, to run the code, please do the following, please do the following in the root directory of the repository: 
  ``` bash
  python ./03_2wayFewShotTesting.py trained_weights_path data_folder_path K
  
  # Example:
  python ./03_2wayFewShotTesting.py rode_close Rode/close/test 4
  ```

### 04_NwayFewShotTesting
- Run this script to test the trained model via N-way K-shot classification tasks. When doing so, you must specify a directory within `weights_testing` which contains the weight files to be used for the model, a directory within `data` containing the .pickle files associated with the mel spectrograms of the cough audio data, the number of classes (coughers) N, and the number of samples K used for each of the N participants in the N-way K-shot classification tasks. For instance, run: `python 04_NwayFewShotTesting.py rode_close Rode/close/test 3 7`, where `rode_close` is be a directory within `weights_testing` and contains .h5 weights files and `./data/Rode/close/test` must contain `test.pickle`. The results of the testing will be printed and subsequently stored in a .csv file within `testing`. <br>
    In summary, to run the code, please do the following in the root directory of the repository: 
  ``` bash
  python ./04_NwayFewShotTesting.py trained_weights_path data_folder_path n_classes K
  
  # Example:
  python ./04_NwayFewShotTesting.py rode_close Rode/close/test 3 7
  ```

### 05_Testing2wayOneShot_DiffSex
- Run this script to test the trained model via 2-way one-shot classification tasks that only consider pairs of participants of opposite sex. When doing so, you must specify a directory within `weights_testing` which contains the weight files to be used for the model and a directory within `data` containing the .pickle files associated with the mel spectrograms of the cough audio data. For instance, run: `python 05_TestingFewShot_DiffSex.py rode_close Rode/close/test`, where `rode_close` is a directory within `weights_testing` and contains .h5 weights files and `./data/Rode/close/test` must contain `test.pickle`. The results of the testing will be printed and subsequently stored in a .csv file within `testing`.
  <br>

    In summary, to run the code, please do the following, please do the following in the root directory of the repository: 
  ``` bash
  python ./05_Testing2wayOneShot_DiffSex.py trained_weights_path data_folder_path
  
  # Example:
  python ./05_Testing2wayOneShot_DiffSex.py rode_close Rode/close/test
  ```

### 06_Testing2wayOneShot_SameSex
- Run this script to test the trained model via 2-way one-shot classification tasks that only consider pairs of participants of the same sex. When doing so, you must specify a directory within `weights_testing` which contains the weight files to be used for the model and a directory within `data` containing the .pickle files associated with the mel spectrograms of the cough audio data. For instance, run: `python 06_TestingFewShot_SameSex.py rode_close Rode/close/test`, where `rode_close` is a directory within `weights_testing` and contains .h5 weights files and `./data/Rode/close/test` must contain `test.pickle`. The results of the testing will be printed and subsequently stored in a .csv file within `testing`.
  <br>

    In summary, to run the code, please do the following in the root directory of the repository: 
  ``` bash
  python ./06_Testing2wayOneShot_SameSex.py trained_weights_path data_folder_path
  
  # Example:
  python ./06_Testing2wayOneShot_SameSex.py rode_close Rode/close/test
    ```


### 07_VerificationTasks
- Run this script to test the trained model regarding the verification task. When doing so, you must specify a directory within `weights_testing` which contains the weight files to be used for the model, a directory within `data` containing the .pickle files associated with the mel spectrograms of the cough audio data and, finally, the third argument is the verification threshold that should be applied. For instance, run: `python 07_VerificationTasks.py rode_close Rode/close/test`, where `rode_close` is a directory within `weights_testing` and contains .h5 weights files and `./data/Rode/close/test` must contain `test.pickle`. The results of the testing will be printed and subsequently stored in a .csv file within `testing`.<br>

  In summary, to run the code, please do the following in the root directory of the repository: 
  ```bash
  python ./07_VerificationTasks.py trained_weights_path data_folder_path threshold

  # Example:
  python /Users/dcleres/cough-assignment/07_VerificationTasks.py rode_close Rode/close/test 0.22
  ```


### 07b_VerificationTasks_GridSearch_Threshold
- Run this script to test the trained model regarding the verification task. When doing so, you must specify a directory within `weights_testing` which contains the weight files to be used for the model and a directory within `data` containing the .pickle files associated with the mel spectrograms of the cough audio data. This time there is no third argument, as the script uses a grid search approach on the validation data to determine the best threshold. For instance, run: `python 07_VerificationTasks.py rode_close Rode/close/val`, where `rode_close` is a directory within `weights_testing` and contains .h5 weights files and `./data/Rode/close/val` must contain `val.pickle`. The results of the testing will be printed and subsequently stored in a .csv file within `testing`.<br>

  In summary, to run the code, please do the following in the root directory of the repository: 
  ```bash
  python ./07b_VerificationTasks_GridSearch_Threshold.py trained_weights_path val_data_folder_path

  # Example:
  python ./07b_VerificationTasks_GridSearch_Threshold.py rode_close  Rode/close/val
  ```

### 08_Compute_EER
  In summary, to run the code, please do the following in the root directory of the repository: 
  ```bash
  python ./08_Compute_EER.py trained_weights_path data_folder_path

  # Example:
  python ./08_Compute_EER.py rode_close  Rode/close/test
  ```


## Bash Scripts

`voluntary_split.sh` and `reflex_split.sh` are bash scripts that were used to generate the training, validation and test data splits for the voluntary and reflex cough data set respectively, by moving the corresponding raw .wav cough audio recordings into new directories. Within each of the directories `training`, `validation` and `testing`, a directory is created for each of the participants (coughers) containing their associated cough recordings (.wav).

## News
- July 2021: Paper submitted for review.

## Requirements
The following libraries are used:

- TensorFlow: `2.5.0`
- TensorBoard: `2.5.0`
- keras: `2.4.3`

We provide a requirements.txt file that can be used to create a conda environment to run the code in this repository. You can install the python packages listed in the requirements.txt using `pip` by running: 

```
pip install -r requirements.txt -U --no-cache-dir 
```

## Cite this Work
For now, please cite our Arxiv version:

Jokic, Stefan, Barata, Filipe et al. "TripletCough: Cougher Identification and Verification from Contact-Free Smartphone-Based Audio Recordings Using Metric Learning" (2021).

```
@article{jokicbarata2021tripletCough,
  title={TripletCough: Cougher Identification and Verification from Contact-Free Smartphone-Based Audio Recordings Using Metric Learning},
  author={Jokic, Stefan and Barata, Filipe and David, Cleres and Rassouli, Frank and Steurer-Stey, Claudia and Puhan, Milo and Brutsche, Martin and Fleisch, Elgar},
  journal={XXX},
  year={2021}
}
```

## Core Team

[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/)


Chair of Information Management at D-​MTEC at ETH Zürich:
- [Dr. sc. Filipe Barata](https://github.com/pipo3000)
- [David Cleres](https://github.com/dcleres)
- [Stefan Jokic](https://gitlab.ethz.ch/jokics)
