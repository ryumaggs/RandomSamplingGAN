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
from main.Dataset import build_dataset

from hyperparam_tuning.run_experiment import load_config, run, build_rngs


SAVE_EVERY = None
SAME_DATA_GT = False
SAME_DATA_BIAS = False
SAME_DATA_SEEN = False
SAME_NETWORK_INIT = True

'''
Experiments to be done:
1. Testing layer size
2. Testing Learning Rate
3. Testing Warmup_duration
4. Testing Dropout
5. testing uneven training
'''

def train(save_path,
          train_device,
          cfg,
          data_cfg):
    '''
    census_dataset_path- str, path to census data csv
    survey_dataset_path- str, path to survey data csv
    save_path - str- directory path to folder to save
    '''
    num_trials = cfg['training']['NUM_TRIALS']
    print("STARTING TRAINING OF", cfg['training']['NUM_TRIALS'], " TOTAL RUNS")
    #create save path if doesnt eixst
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    all_rngs = [build_rngs(cfg["rng"], train_device) for _ in range(num_trials)]
    for tid in range(num_trials):
        rngs = all_rngs[tid]
        print(rngs)
        trial_save_path = os.path.join(save_path,"trial_"+str(tid)+"/")
        if not os.path.isdir(trial_save_path):
            os.mkdir(trial_save_path)

        print("-----------------------------------")
        print("RUN:", tid+1, "/", num_trials)
        print("")
        
        d = build_dataset(cfg, 
                    data_cfg,
                    rngs,
                    cfg['data']['weeks'][0], 
                    train_device,)  
      
        hparams = cfg['hparams']
        print(hparams["gen_layers"])
        gan = WGAN_GP(
            rngs=rngs,
            dataset=d,
            generator_type=hparams['generator_type'],
            discriminator_type=hparams['discriminator_type'],
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
            lambda_JSD = hparams['lambdaJSD'],
            gen_history_length=0,
            temperature=hparams["tau"],
            warmup_length=0,
            lambda_regularizer=0,
            lambda_first_layer=0,
            generator_dropout=hparams["generator_dropout"],
            discriminator_dropout=hparams["discriminator_dropout"],
            KLIEP_downsample=-1,
            device=train_device,
            save_dict= cfg['SAVE_DICT'],
            save_dir = trial_save_path,
        )

        comment = "runs"
        base_logdir = save_path

        log_dir = os.path.join(
            base_logdir,
            comment,
            datetime.now().strftime("%Y%m%d-%H%M%S")+"||Week="+str(cfg['data']['weeks'][0])+"||trial_id="+str(tid),
            
        )

        writer = SummaryWriter(log_dir=log_dir)
        
        hparams_copy = copy.deepcopy(hparams)
        for key, item in hparams_copy.items():
            if isinstance(item, list):
                hparams_copy[key] = str(hparams_copy[key])
        
        writer.add_hparams(hparams_copy,{})


        _  = gan.train(
                        cfg['training']['epochs'],
                        cfg['training']['gtrainingfactor'],
                        cfg['training']['dtrainingfactor'],
                        writer,
                        trial_id=tid,
                        synthetic=False,
                        synthetic_col_names = "")
        
        writer.close()

def train_synthetic(census_dataset_path,
          survey_dataset_path,
          save_path,
          save_dict,
          num_trials,
          train_device,
          cfg,
          data_cfg):
    '''
    census_dataset_path- str, path to census data csv
    survey_dataset_path- str, dummy argument doesnt use
    save_path - str- directory path to folder to save
    '''
    print("STARTING TRAINING OF", num_trials, " TOTAL RUNS")
    #create save path if doesnt eixst
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    for tid in range(num_trials):
        trial_save_path = os.path.join(save_path,"trial_"+str(tid)+"/")
        if not os.path.isdir(trial_save_path):
            os.mkdir(trial_save_path)

        print("-----------------------------------")
        print("RUN:", tid+1, "/", num_trials)
        print("")

        all_rngs = [build_rngs(cfg["rng"], train_device) for _ in range(num_trials)]

        rngs = all_rngs[tid][0]
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