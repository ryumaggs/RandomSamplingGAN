import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import csv
from scipy.special import softmax
import random
import os
from datetime import datetime
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='train_log.log', level=logging.INFO)
import pandas as pd
from sklearn import preprocessing
import pytorch_warmup as warmup
import pickle

#local imports
from main.Dataset import Axios_ipsosdataset, HouseholdPulse_dataset, D4P_dataset, HouseholdPulse_synthetic, Ari_dataset
from main.GAN import GAN, WGAN_GP, WGAN_GP_fict
from main.Discriminator import DataDiscriminator
from main.util import set_seed
from DataProcessing.GeneralDataProcessing import *

#optimize over w directly, dont use any theta shenannigans. 



#NEW TRAINING VARIABLE SET UP
# For learning
#runs_consistency_SameData_SameSeen_diffNetwork,Seed:359556
DEBUG_MODE = False
BATCH_SIZE = 2
SUBSET_SIZE = 32
GENERATOR_TRAINING_FACTOR = 1
DISCRIMINATOR_TRAINING_FACTOR = 8
GT_LIMIT = 50000
BIAS_LIMIT = 10000 #10000
GT_LIMIT_SYNTHETIC = 5000
BIAS_LIMIT_SYNTHETIC = 2500
SAVE_DICT = {}
SAVE_DICT['SAVE_DATASET'] = True
SAVE_DICT['SAVE_WEIGHTS'] = False
SAVE_DICT['SAVE_IG'] = False
SAVE_DICT['SAVE_GENERATOR'] = False

KLIEP_DOWNSAMPLE = BIAS_LIMIT
LAMBDAGP = 10 #hold this at 10. higher = less discriminator expressability
LAMBDAW = 0.003 #.00075 for 10k
LAMBDAD = 13.5 #13.5
LAMBDA_FIRST_LAYER = 0
GENERATOR_LEARNING_RATE = 1e-5 #2e-3 or 1e-5
DISCRIMINATOR_LEARNING_RATE = 5e-5 #1e-3 or 5e-5
BATCHS_IN_EPOCH = 1
EPOCHS = 50
NUM_TRIALS = 1
GEN_HISTORY_LENGTH = 0
WARMUP_EPOCHS = 0
GEN_LAYERS = [256, 256] #[1024, 1024, 1024] #[256 for _ in range(5)]
DISC_LAYERS = [256, 256]
GEN_DROPOUT = 0.2
DISC_DROPOUT = 0.2
TEMPERATURE_START = 1
TEMPERATURE_END = 0.3
TEMPERATURE = 0.1

DEBUG_MODE = False

if DEBUG_MODE:
    BATCH_SIZE=2
    SUBSET_SIZE=2
    DISCRIMINATOR_TRAINING_FACTOR = 2
    BIAS_LIMIT = 5
    EPOCHS = 30
    NUM_TRIALS = 1
    WARMUP_EPOCHS = 10
    SAVE_DATASET = False
    SAVE_WEIGHTS = False

SAVE_EVERY = None
SAME_DATA_GT = False
SAME_DATA_BIAS = False
SAME_DATA_SEEN = False
SAME_NETWORK_INIT = True

#measures of consistency

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# For dataset
zero_prob = 0

generator_types = ['dataGen'] #['dataGen','weightsGen','onesGen']

regularizer_lambdas = [1]

warmup_durations = [1]
results = []
'''
Experiments to be done:
1. Testing layer size
2. Testing Learning Rate
3. Testing Warmup_duration
4. Testing Dropout
5. testing uneven training
'''

def train(census_dataset_path,
          survey_dataset_path,
          save_path,
          save_dict,
          num_trials,
          train_device):
    '''
    census_dataset_path- str, path to census data csv
    survey_dataset_path- str, path to survey data csv
    save_path - str- directory path to folder to save
    '''
    print("STARTING TRAINING OF", num_trials, " TOTAL RUNS")
    #create save path if doesnt eixst
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    fixed_seed = [np.random.randint(0,1e6)]
    all_rngs = []
    print('SETTING SEED: ', fixed_seed)
    if type(fixed_seed) == list:
        for _ in range(num_trials):
            t_rng = []
            for seed in fixed_seed:
                t_rng.append(set_seed(seed,
                                        train_device,
                                    data_init=[SAME_DATA_GT, SAME_DATA_BIAS],
                                    data_gen =SAME_DATA_SEEN,
                                    network_init=SAME_NETWORK_INIT,))
            all_rngs.append(t_rng)
    else:
        for _ in range(num_trials):
            all_rngs.append(set_seed(fixed_seed,
                                    train_device,
                                data_init=[SAME_DATA_GT, SAME_DATA_BIAS],
                                data_gen =SAME_DATA_SEEN,
                                network_init=SAME_NETWORK_INIT,))
    for tid in range(num_trials):
        trial_save_path = os.path.join(save_path,"trial_"+str(tid)+"/")
        if not os.path.isdir(trial_save_path):
            os.mkdir(trial_save_path)

        print("-----------------------------------")
        print("RUN:", tid+1, "/", num_trials)
        print("")
        rngs = all_rngs[tid][0]
        hparams = {
            "glearningrate": GENERATOR_LEARNING_RATE,
            "dlearningrate": DISCRIMINATOR_LEARNING_RATE,
            "gen_layers": GEN_LAYERS,
            "disc_layers": DISC_LAYERS,
            "wudur": warmup_durations[0],
            "generator_dropout": GEN_DROPOUT,
            "discriminator_dropout": DISC_DROPOUT,
            "gtrainingfactor": GENERATOR_TRAINING_FACTOR,
            "dtrainingfactor": DISCRIMINATOR_TRAINING_FACTOR,
            "subset_size": SUBSET_SIZE,
            "batch_size": BATCH_SIZE,
            "lambdagp": LAMBDAGP,
            "lambdaw": LAMBDAW,
            "lambdad": LAMBDAD,
            "tau": TEMPERATURE,
            "gen_history_length": GEN_HISTORY_LENGTH,
            "epochs": EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "lambda_first_layer": LAMBDA_FIRST_LAYER,
            "KLIEP_downsample": KLIEP_DOWNSAMPLE,
        }
        
        d = HouseholdPulse_dataset(ground_truth_path=census_dataset_path,
                        bias_path = survey_dataset_path,
                        rngs=rngs,
                        label_information = {'RECVDVACC':0},
                        device=train_device,
                        columns_to_keep = None,
                        gt_limit = GT_LIMIT,
                        bias_limit = BIAS_LIMIT, 
                        )
        #devices have been checked. should work with device input
        

        gan = WGAN_GP(
            rngs=rngs,
            dataset=d,
            generator_type='deepSet',
            discriminator_type='deepSet',
            gen_learning_rate=hparams["glearningrate"],
            disc_learning_rate=hparams["dlearningrate"],
            batch_size=hparams["batch_size"],
            truth_sample_size=hparams["subset_size"],
            gen_layers=hparams["gen_layers"],
            disc_layers=hparams["disc_layers"],
            bias_sample_size=hparams["subset_size"],
            lambda_gp=hparams["lambdagp"],
            lambda_weights=hparams["lambdaw"],
            lambda_demo=hparams["lambdad"],
            gen_history_length=hparams["gen_history_length"],
            temperature=hparams["tau"],
            warmup_length=hparams["wudur"],
            lambda_regularizer=0,
            lambda_first_layer=hparams["lambda_first_layer"],
            generator_dropout=hparams["generator_dropout"],
            discriminator_dropout=hparams["discriminator_dropout"],
            KLIEP_downsample=hparams['KLIEP_downsample'],
            device=train_device,
            save_dict= save_dict,
            save_dir = trial_save_path,
        )

        comment = "runs"
        base_logdir = save_path

        log_dir = os.path.join(
            base_logdir,
            comment,
            datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        writer = SummaryWriter(log_dir=log_dir)
        
        for key, item in hparams.items():
            if isinstance(item, list):
                hparams[key] = str(hparams[key])
        
        writer.add_hparams(hparams,{})


        weights, bias_labels, prob_diffs,  test_probs, test_prob_diffs, generator_losses, discriminator_losses  = gan.train(BATCHS_IN_EPOCH,
                                                                                                                                hparams['epochs'],
                                                                                                                                hparams['warmup_epochs'],
                                                                                                                                TEMPERATURE_START,
                                                                                                                                TEMPERATURE_END,
                                                                                                                                hparams['gtrainingfactor'],
                                                                                                                                hparams['dtrainingfactor'],
                                                                                                                                SAVE_EVERY,
                                                                                                                                writer,
                                                                                                                                tid,
                                                                                                                                synthetic=False,
                                                                                                                                synthetic_col_info=[''])
        
        #results.append((exp_var,cvar,predicted_target))

        #with open('./results.pkl', 'wb') as file:
        #    pickle.dump(results,file)
        writer.close()

def train_synthetic(census_dataset_path,
          survey_dataset_path,
          save_path,
          save_dict,
          num_trials,
          train_device):
    '''
    census_dataset_path- str, path to census data csv
    survey_dataset_path- str, dummy argument doesnt use
    save_path - str- directory path to folder to save
    '''
    print("STARTING TRAINING OF", num_trials, " TOTAL RUNS")
    #create save path if doesnt eixst
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    fixed_seed = [np.random.randint(0,1e6)]
    all_rngs = []
    print('SETTING SEED: ', fixed_seed)
    if type(fixed_seed) == list:
        for _ in range(num_trials):
            t_rng = []
            for seed in fixed_seed:
                t_rng.append(set_seed(seed,
                                        train_device,
                                    data_init=[SAME_DATA_GT, SAME_DATA_BIAS],
                                    data_gen =SAME_DATA_SEEN,
                                    network_init=SAME_NETWORK_INIT,))
            all_rngs.append(t_rng)
    else:
        for _ in range(num_trials):
            all_rngs.append(set_seed(fixed_seed,
                                    train_device,
                                data_init=[SAME_DATA_GT, SAME_DATA_BIAS],
                                data_gen =SAME_DATA_SEEN,
                                network_init=SAME_NETWORK_INIT,))
    for tid in range(num_trials):
        trial_save_path = os.path.join(save_path,"trial_"+str(tid)+"/")
        if not os.path.isdir(trial_save_path):
            os.mkdir(trial_save_path)

        print("-----------------------------------")
        print("RUN:", tid+1, "/", num_trials)
        print("")
        rngs = all_rngs[tid][0]
        hparams = {
            "glearningrate": GENERATOR_LEARNING_RATE,
            "dlearningrate": DISCRIMINATOR_LEARNING_RATE,
            "gen_layers": GEN_LAYERS,
            "disc_layers": DISC_LAYERS,
            "wudur": warmup_durations[0],
            "generator_dropout": GEN_DROPOUT,
            "discriminator_dropout": DISC_DROPOUT,
            "gtrainingfactor": GENERATOR_TRAINING_FACTOR,
            "dtrainingfactor": DISCRIMINATOR_TRAINING_FACTOR,
            "subset_size": SUBSET_SIZE,
            "batch_size": BATCH_SIZE,
            "lambdagp": LAMBDAGP,
            "lambdaw": LAMBDAW,
            "lambdad": LAMBDAD,
            "tau": TEMPERATURE,
            "gen_history_length": GEN_HISTORY_LENGTH,
            "epochs": EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "lambda_first_layer": LAMBDA_FIRST_LAYER,
            "KLIEP_downsample": KLIEP_DOWNSAMPLE,
        }
        
        d = HouseholdPulse_synthetic(ground_truth_path=census_dataset_path,
                        bias_path = survey_dataset_path,
                        rngs=rngs,
                        label_information = {'RECVDVACC':0},
                        device=train_device,
                        columns_to_keep = None,
                        gt_limit = GT_LIMIT_SYNTHETIC,
                        bias_limit = BIAS_LIMIT_SYNTHETIC, 
                        max_num_ran_var=2,
                        )
        #devices have been checked. should work with device input
        

        gan = WGAN_GP(
            rngs=rngs,
            dataset=d,
            generator_type='deepSet',
            discriminator_type='deepSet',
            gen_learning_rate=hparams["glearningrate"],
            disc_learning_rate=hparams["dlearningrate"],
            batch_size=hparams["batch_size"],
            truth_sample_size=hparams["subset_size"],
            gen_layers=hparams["gen_layers"],
            disc_layers=hparams["disc_layers"],
            bias_sample_size=hparams["subset_size"],
            lambda_gp=hparams["lambdagp"],
            lambda_weights=hparams["lambdaw"],
            lambda_demo=hparams["lambdad"],
            gen_history_length=hparams["gen_history_length"],
            temperature=hparams["tau"],
            warmup_length=hparams["wudur"],
            lambda_regularizer=0,
            lambda_first_layer=hparams["lambda_first_layer"],
            generator_dropout=hparams["generator_dropout"],
            discriminator_dropout=hparams["discriminator_dropout"],
            KLIEP_downsample=hparams['KLIEP_downsample'],
            device=train_device,
            save_dict= save_dict,
            save_dir = trial_save_path,
        )

        comment = "runs"
        base_logdir = save_path

        log_dir = os.path.join(
            base_logdir,
            comment,
            datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        writer = SummaryWriter(log_dir=log_dir)
        
        for key, item in hparams.items():
            if isinstance(item, list):
                hparams[key] = str(hparams[key])
        
        writer.add_hparams(hparams,{})


        weights, bias_labels, prob_diffs,  test_probs, test_prob_diffs, generator_losses, discriminator_losses  = gan.train(BATCHS_IN_EPOCH,
                                                                                                                                hparams['epochs'],
                                                                                                                                hparams['warmup_epochs'],
                                                                                                                                TEMPERATURE_START,
                                                                                                                                TEMPERATURE_END,
                                                                                                                                hparams['gtrainingfactor'],
                                                                                                                                hparams['dtrainingfactor'],
                                                                                                                                SAVE_EVERY,
                                                                                                                                writer,
                                                                                                                                tid,
                                                                                                                                synthetic=True,
                                                                                                                                synthetic_col_info=d.joint_table)
        
        #results.append((exp_var,cvar,predicted_target))

        #with open('./results.pkl', 'wb') as file:
        #    pickle.dump(results,file)
        writer.close()