# RandomSamplingGAN

## Setup/Install
``` pip install -r requirements.txt ```

Code is known to work with python version 3.9.4

## Changing parameters

Almost all parameters are located in ***config.yaml***

The important variables to note are:
- EPOCHS: changes training duration. for fast attribution debugging can set to 1. default: 200
- DEVICE: the string input to "torch.device(DEVICE)". default: "cuda:0"
- BIAS_LIMIT: how many survey points are sampled, greatly impacts attribution time taken. default: 2500, tested up to 10,000
- SAVE_DICT: a dictionary of booleans dictating what should be saved on each run
  - SAVE_IG: save integrated gradients flag
  - SAVE_WEIGHTS: save weights flag
  - SAVE_DATASET: save data set flag

## Running Code

``` python experiments.py ```

Will first train a RWGAN system for EPOCHS number of steps, and then run attribution (Integrated gradients) using trained system

## Playing around and debugging Integrated gradients

All Integrated gradients code is found within ***attribution.py***

in particular, the first 3 functions:
- def randomly_select_valid_points
- def compute_equispaced_inputs_IG
- def compute_IG_weights

To adjust, debug, or see the output of attribution you have two options:
- add print statements to attribution.py
- load the saved integrated gradients

## Saved files after each run

Running experiments.py will generate the following save directories:
- Run: Tensorboard files
- save_week(config['week'])
  - trial:trial_id directory
    - data_0.npz: raw survey data
    - one_hot_data.npz: one hot encoded survey data (use this one)
    - embedding_dict_trialID_.pikl: how to encode from raw data to one hot encoding
    - grad_history_0.npz: integrated gradients
