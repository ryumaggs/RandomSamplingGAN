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
device = torch.device('cuda:0')
import random
import os
from datetime import datetime
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='train_log.log', level=logging.INFO)

from main.Dataset import Axios_ipsosdataset, HouseholdPulse_dataset, D4P_dataset, HouseholdPulse_synthetic, Ari_dataset
from main.GAN import GAN, WGAN_GP, WGAN_GP_fict
from main.Discriminator import DataDiscriminator
from main.util import set_seed
import itertools
import pandas as pd
#optimize over w directly, dont use any theta shenannigans. 
from sklearn import preprocessing
import pytorch_warmup as warmup
from DataProcessing import *
import pickle
from itertools import combinations

#NEW TRAINING VARIABLE SET UP
# For learning
#runs_consistency_SameData_SameSeen_diffNetwork,Seed:359556
BATCH_SIZE = 2
SUBSET_SIZE = 32
GENERATOR_TRAINING_FACTOR = 1
DISCRIMINATOR_TRAINING_FACTOR = 200
GT_LIMIT = 25000
BIAS_LIMIT = 2500
LAMBDAGP = 10
LAMBDAW = 1 #40
LAMBDAD = 1
GENERATOR_LEARNING_RATE = 1e-5
DISCRIMINATOR_LEARNING_RATE = 1e-4
BATCHS_IN_EPOCH = 1
EPOCHS = 100 # the stream is infinite so one epoch will be defined as BATCHS_IN_EPOCH * BATCH_SIZE
NUM_TRIALS = 1
GEN_HISTORY_LENGTH = 0
WARMUP_EPOCHS = 600

SAME_DATA_GT = False
SAME_DATA_BIAS = False
SAME_DATA_SEEN = False
SAME_NETWORK_INIT = False

DEBUG_MODE = True

if DEBUG_MODE:
    BATCH_SIZE=2
    SUBSET_SIZE=2
    DISCRIMINATOR_TRAINING_FACTOR = 5
    EPOCHS = 10
    NUM_TRIALS = 1
    WARMUP_EPOCHS = 20

SAVE_EVERY = None

GEN_DROPOUT = 0.2
DISC_DROPOUT = 0.2

#measures of consistency

TEMPERATURE_START = 1
TEMPERATURE_END = 0.3
TEMPERATURE = 0.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# For dataset
zero_prob = 0
    
GEN_LAYERS = [1024, 1024, 1024] #[256 for _ in range(5)]
DISC_LAYERS = [1024, 1024, 1024]

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
          save_dataset,
          save_weights,
          num_trials):
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
                                        device,
                                    data_init=[SAME_DATA_GT, SAME_DATA_BIAS],
                                    data_gen =SAME_DATA_SEEN,
                                    network_init=SAME_NETWORK_INIT,))
            all_rngs.append(t_rng)
    else:
        for _ in range(num_trials):
            all_rngs.append(set_seed(fixed_seed,
                                    device,
                                data_init=[SAME_DATA_GT, SAME_DATA_BIAS],
                                data_gen =SAME_DATA_SEEN,
                                network_init=SAME_NETWORK_INIT,))
    for tid in range(num_trials):
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
    }
        
        d = HouseholdPulse_dataset(ground_truth_path=census_dataset_path,
                        bias_path = survey_dataset_path,
                        rngs=rngs,
                        device=device,
                        columns_to_keep = None,
                        gt_limit = GT_LIMIT,
                        bias_limit = BIAS_LIMIT, 
                        )
        
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
                generator_dropout=hparams["generator_dropout"],
                discriminator_dropout=hparams["discriminator_dropout"],
                save_dataset=save_dataset,
                save_weights=save_weights,
                save_dir = save_path,
            )

        comment = "runs"
        base_logdir = "./saves"

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
                                                                                                                                    tid)
        predicted_target = (weights @ bias_labels).item()
        
        #results.append((exp_var,cvar,predicted_target))

        #with open('./results.pkl', 'wb') as file:
        #    pickle.dump(results,file)
        writer.close()

    #rename the runs folder into its appropriate name
    #if NUM_TRIALS > 1:
    #    os.rename("./runs", "./runs_week"+str(w))