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

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='train_log.log', level=logging.INFO)

from Dataset import Axios_ipsosdataset, HouseholdPulse_dataset, D4P_dataset, HouseholdPulse_synthetic, Ari_dataset
from GAN import GAN, WGAN_GP, WGAN_GP_fict
from Discriminator import DataDiscriminator
from util import set_seed
import itertools
import pandas as pd
from sklearn import preprocessing
import pytorch_warmup as warmup
from DataProcessing import *
import pickle
from itertools import combinations
from attribution import *
import yaml
#NEW TRAINING VARIABLE SET UP
# For learning
#runs_consistency_SameData_SameSeen_diffNetwork,Seed:359556
DEBUG_MODE = False


'''
what i need this seed to do is to be able to recreate an experiment while changing some small
aspect of the randomness.

e.g. keep everyting the same but use a different GT subset of the GT data. 

Therefore fixed_seed should be a single seed and should be appended
to all rngs NUM_TRIALS times. 


'''
if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    #setup randomness
    fixed_seed = np.random.randint(1e6)
    all_rngs = []
    print('SETTING SEED: ', fixed_seed)
    for _ in range(config['NUM_TRIALS']):
        all_rngs.append(set_seed(fixed_seed,
                                device,
                            data_init=[config['SAME_DATA_GT'], config['SAME_DATA_BIAS']],
                            data_gen =config['SAME_DATA_SEEN'],
                            network_init=config['SAME_NETWORK_INIT'],))

    device = torch.device(config['DEVICE'])

    #make a new runs and save folder
    w = str(config['week'][0])
    main_dir = ""
    if any(config['SAVE_DICT'].values()):
        main_dir = "./saves_week"+str(w)
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
            print(f"Directory '{main_dir}' created.")
        else:
            print(f"Directory '{main_dir}' already exists.")

    for tid in range(config['NUM_TRIALS']):
        rngs = all_rngs[tid]
        trial_dir = ""
        if any(config['SAVE_DICT'].values()):
            trial_dir = os.path.join(main_dir,"trial:"+str(tid))
            if not os.path.exists(trial_dir):
                os.makedirs(trial_dir)
                print(f"Directory '{trial_dir}' created.")
            else:
                print(f"Directory '{trial_dir}' already exists.")

        hparams = {
            "glearningrate": config['GENERATOR_LEARNING_RATE'],
            "dlearningrate": config['DISCRIMINATOR_LEARNING_RATE'],
            "gen_layers": config['GEN_LAYERS'],
            "disc_layers": config['DISC_LAYERS'],
            "wudur": 1,
            "generator_dropout": config['GEN_DROPOUT'],
            "discriminator_dropout": config['DISC_DROPOUT'],
            "gtrainingfactor": config['GENERATOR_TRAINING_FACTOR'],
            "dtrainingfactor": config['DISCRIMINATOR_TRAINING_FACTOR'],
            "subset_size": config['SUBSET_SIZE'],
            "batch_size": config['BATCH_SIZE'],
            "lambdagp": config['LAMBDAGP'],
            "lambdaw": config['LAMBDAW'],
            "lambdad": config['LAMBDAD'],
            "tau": config['TEMPERATURE'],
            "gen_history_length": config['GEN_HISTORY_LENGTH'],
            "epochs": config['EPOCHS'],
            "warmup_epochs": config['WARMUP_EPOCHS'],
            "lambda_first_layer": config['LAMBDA_FIRST_LAYER'],
            "KLIEP_downsample": config['KLIEP_DOWNSAMPLE'],
        }
        
        d = HouseholdPulse_dataset(ground_truth_path='./data/censusHouseholdPulse_data/cleaned/ipums_cleaned_combined.csv',
                        bias_path = './data/censusHouseholdPulse_data/cleaned/pulse_week'+w+'_cleaned.csv',
                        rngs=rngs,
                        label_information={'RECVDVACC':0}, #{'VAC':0, 'HLTHINS1':1, 'RECVDVACC},
                        device=device,
                        columns_to_keep = None,
                        gt_limit = config['GT_LIMIT'],
                        bias_limit = config['BIAS_LIMIT'],
        )
        
        #print("idealy 1k to 5k generator updates...")
        #print("N Generator updates: ", (EPOCHS * GT_LIMIT) / (BATCH_SIZE * hparams['dtrainingfactor']))
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
                    save_dict=config['SAVE_DICT'],
                    save_dir = trial_dir,
                )

        synthetic = (w == 'SYN')
        if synthetic: #synthetic data experiments
            writer = SummaryWriter(comment='Variables=' + str(d.column_names) + '||Cell=' + str(d.upscaled_cell)+'||seed:'+str(rngs['seed_bias']))
        else: #non synthetic data experiments
            writer = SummaryWriter(comment='iter: ' + str(tid) + '||Week='+w+
                                '||seed:'+str(rngs['seed_bias']))
        
        for key, item in hparams.items():
            if isinstance(item, list):
                hparams[key] = str(hparams[key])
        
        writer.add_hparams(hparams,{})
        c_names = ""
        if w == "Syn":
            c_names = d.column_names
        
        #print(d.column_names)
        weights, bias_labels, prob_diffs,  test_probs, test_prob_diffs, generator_losses, discriminator_losses  = gan.train(config['BATCHS_IN_EPOCH'],
                                                                                                                    config['EPOCHS'],
                                                                                                                    config['WARMUP_EPOCHS'],
                                                                                                                    config['TEMPERATURE_START'],
                                                                                                                    config['TEMPERATURE_END'],
                                                                                                                    config['GENERATOR_TRAINING_FACTOR'],
                                                                                                                    config['DISCRIMINATOR_TRAINING_FACTOR'],
                                                                                                                    config['SAVE_DICT']['SAVE_EVERY'],
                                                                                                                    writer,
                                                                                                                    tid,
                                                                                                                    synthetic=synthetic,
                                                                                                                    synthetic_col_names=c_names)
        #predicted_target = (weights @ bias_labels).item()
        
        #results.append((exp_var,cvar,predicted_target))
        writer.close()

        '''
        d = Axios_ipsosdataset(ground_truth_path='./data/axios_ipsos_data/cleaned/ipums_cleaned.csv',
                        bias_path = './data/axios_ipsos_data/cleaned/week'+week+'_cleaned.csv',
                        rngs=rngs,
                        label_information={'RECVDVACC':0}, #{'VAC':0, 'HLTHINS1':1},
                        device=device,
                        columns_to_keep = None,
                        gt_limit = GT_LIMIT,
                        bias_limit = BIAS_LIMIT, 
                        )

        d = D4P_dataset(ground_truth_path='./data/progress_data/cleaned/ipums_cleaned_combined.csv',
                        bias_path = './data/progress_data/cleaned/d4p_week'+week+'_cleaned.csv',
                        rngs=rngs,
                        label_information={'RECVDVACC':0}, #{'VAC':0, 'HLTHINS1':1},
                        device=device,
                        columns_to_keep = None,
                        gt_limit = GT_LIMIT,
                        bias_limit = BIAS_LIMIT, 
                        )
        d = HouseholdPulse_dataset(ground_truth_path='./data/HouseholdPulse_data/cleaned/ipums_cleaned_combined.csv',
                        bias_path = './data/HouseholdPulse_data/cleaned/pulse_week'+week+'_cleaned.csv',
                        rngs=rngs,
                        label_information={'RECVDVACC':0}, #{'VAC':0, 'HLTHINS1':1, 'RECVDVACC},
                        device=device,
                        columns_to_keep = None,
                        gt_limit = GT_LIMIT,
                        bias_limit = BIAS_LIMIT, 
                        )

        d = HouseholdPulse_synthetic(ground_truth_path='./data/censusHouseholdPulse_data/cleaned/ipums_cleaned.csv',
                        bias_path = './data/censusHouseholdPulse_data/cleaned/pulse_week'+week+'_cleaned.csv',
                        rngs=rngs,
                        device=device,
                        gt_limit = GT_LIMIT*len(rngs),
                        bias_limit = BIAS_LIMIT, 
                        )

        data_creation_counter = 0
        try:
            while True:
                
                if not d.missing_values:
                    break
                else:
                    data_creation_counter += 1
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Exiting gracefully.")
        '''