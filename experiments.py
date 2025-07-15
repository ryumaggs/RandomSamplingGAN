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
#optimize over w directly, dont use any theta shenannigans. 
from sklearn import preprocessing
import pytorch_warmup as warmup
from DataProcessing import *
import pickle
from itertools import combinations

#NEW TRAINING VARIABLE SET UP
# For learning
#runs_consistency_SameData_SameSeen_diffNetwork,Seed:359556
DEBUG_MODE = False

BATCH_SIZE = 16
SUBSET_SIZE = 128
GENERATOR_TRAINING_FACTOR = 1
DISCRIMINATOR_TRAINING_FACTOR = 8
GT_LIMIT = 25000
BIAS_LIMIT = 2000
LAMBDAGP = 0
LAMBDAW = 1 #40
LAMBDAD = 1
GENERATOR_LEARNING_RATE = 1e-4 #5e-6
DISCRIMINATOR_LEARNING_RATE = 1e-5 #1e-6
BATCHS_IN_EPOCH = 1
EPOCHS = 500 # the stream is infinite so one epoch will be defined as BATCHS_IN_EPOCH * BATCH_SIZE
NUM_TRIALS = 300
GEN_HISTORY_LENGTH = 1

SAME_DATA_GT = True
SAME_DATA_BIAS = True
SAME_DATA_SEEN = True
SAME_NETWORK_INIT = True
#
fixed_seed = [np.random.randint(1e6)]
all_rngs = []
print('SETTING SEED: ', fixed_seed)
if type(fixed_seed) == list:
    for _ in range(NUM_TRIALS):
        t_rng = []
        for seed in fixed_seed:
            t_rng.append(set_seed(seed,
                                    device,
                                data_init=[SAME_DATA_GT, SAME_DATA_BIAS],
                                data_gen =SAME_DATA_SEEN,
                                network_init=SAME_NETWORK_INIT,))
        all_rngs.append(t_rng)
else:
    for _ in range(NUM_TRIALS):
        all_rngs.append(set_seed(fixed_seed,
                                device,
                            data_init=[SAME_DATA_GT, SAME_DATA_BIAS],
                            data_gen =SAME_DATA_SEEN,
                            network_init=SAME_NETWORK_INIT,))

DEBUG_MODE = False

if DEBUG_MODE:
    BATCH_SIZE=2
    SUBSET_SIZE=2
    DISCRIMINATOR_TRAINING_FACTOR = 5
    EPOCHS = 10
    NUM_TRIALS = 30

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


    
GEN_LAYERS = [1024] #[256 for _ in range(5)]
DISC_LAYERS = [1024]

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

def check_identical_networks(gan1, gan2):
    if gan1 is None or gan2 is None:
        return
    print("Generator rhos: ", all(torch.equal(p1, p2) for p1, p2 in zip(gan1.generator.rho.parameters(), gan2.generator.rho.parameters())))
    print("Generator phis: ", all(torch.equal(p1, p2) for p1, p2 in zip(gan1.generator.phi.parameters(), gan2.generator.phi.parameters())))
    print("Discriminator rhos: ", all(torch.equal(p1, p2) for p1, p2 in zip(gan1.discriminator.rho.parameters(), gan2.discriminator.rho.parameters())))
    print("Discriminator phis: ", all(torch.equal(p1, p2) for p1, p2 in zip(gan1.discriminator.phi.parameters(), gan2.discriminator.phi.parameters())))

if __name__ == "__main__":
    for w in range(29,30):
        #make a new runs folder
        week = str(w)
        exp_vars = {}
        exp_vars['disc_layers'] = [[1024,1024,1024]]
        #exp_vars['gen_layers'] = [[1024]]
        #exp_vars['learning_rates'] = [5e-5, 1e-5, 5e-6, 1e-6]
        #exp_vars['warmup_duration'] = [1000, 2000, 3000, 4000]
        #exp_vars['discriminator_dropout'] = [0.1, 0.2, 0.3]
        #exp_vars['disc_training_factor'] = [1,2]
        #exp_vars['lambda_gp'] = [10]
        #exp_vars['disc_learning_rate'] = [3e-6]
        #exp_vars['gen_learning_rate'] = [1e-6]
        #exp_vars['gt_limit'] = [GT_LIMIT]
        #exp_vars['bias_limit'] = [BIAS_LIMIT]
        #exp_vars['subset_size'] = [512]
        #exp_vars['gen_training_factor'] = [1,3,5]
        #exp_vars['batch_size'] = [1,4,8,16,32,64]
        #exp_vars['lambdaw'] = [500]
        #exp_vars['tau'] = [0.05, 0.25, 0.55, 0.75, 0.95]
        #exp_vars['columns_to_keep'] = [('EDUC', 'INCTOT', 'AGE')] #['REGION', 'EDUC', 'INCTOT', 'SEX', 'MARST', 'FAMSIZE', 'RACE','AGE', 'BIDENPERC']
        #exp_vars['lambdad'] = [0,10,20,30,40]
        #exp_vars['gen_history_length'] = [1]
        old_g = None

        if 'columns_to_keep' in exp_vars and len(exp_vars['columns_to_keep']) == 0:
            # Generate power set up to size 2
            power_set = []
            for r in [1,2,3,4,5,6,7,8]:
                power_set.extend(combinations(exp_vars['columns_to_keep'], r))
            
            exp_vars['columns_to_keep'] = power_set

        for exp_var,val in exp_vars.items():
            for tid in range(NUM_TRIALS):
                rngs = all_rngs[tid]

                if False:
                    d = ImportanceDataset(ground_truth_path=None,
                                    num_biased_data_points=num_biased_data_points,
                                    device=device)
                    d = RealImportanceDataset(ground_truth_path='./data/normalizedCleaned100.csv',
                                            num_totaldata_ground_truth=1000,
                                            device=device)
                    #save_new_XBOX_csvs()
                    d = RealPredictionDataset(ground_truth_path='./data/XBOX_GT.csv',
                                            bias_path = './data/XBOX_bias.csv',
                                            device=device)
                    
                    d = HouseholdPulse_dataset(ground_truth_path='./data/censusHouseholdPulse_data/ipums_cleaned.csv',
                                        bias_path = './data/censusHouseholdPulse_data/pulse_week22_cleaned.csv',
                                        device=device)
                    
                    d = Axios_ipsosdataset(ground_truth_path='./data/axios_processed_data/ipums_processed.csv',
                                            bias_path = './data/axios_processed_data/wave'+str(wave_id)+'_processed.csv',
                                            device=device,
                                            gt_limit=gt_limit)

                for cvar in val:

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
                    }

                    if exp_var == 'gt_limit':
                        pass
                    elif exp_var == 'bias_limit':
                        pass
                    elif exp_var in hparams:
                        hparams[exp_var] = cvar
                    else:
                        raise NotImplementedError
                    
                    '''
                    d = D4P_dataset(ground_truth_path='./data/censusHouseholdPulse_data/ipums_cleaned.csv',
                                    bias_path = './data/progress_data/week'+week+'_cleaned.csv',
                                    rngs=rngs,
                                    device=device,
                                    gt_limit = GT_LIMIT,
                                    )
                    d = HouseholdPulse_dataset(ground_truth_path='./data/censusHouseholdPulse_data/ipums_cleaned.csv',
                                    bias_path = './data/censusHouseholdPulse_data/pulse_week'+week+'_cleaned.csv',
                                    rngs=rngs,
                                    device=device,
                                    columns_to_keep = None,
                                    gt_limit = GT_LIMIT*len(rngs),
                                    bias_limit = BIAS_LIMIT, 
                                    )
                    d = Ari_dataset(ground_truth_path='./data/ari_survey/ipums_cleaned.csv',
                                    bias_path = './data/ari_survey/aricleaned.csv',
                                    rngs=rngs,
                                    device=device,
                                    columns_to_keep = None,
                                    gt_limit = GT_LIMIT*len(rngs),
                                    bias_limit = BIAS_LIMIT,
                                    )
                    '''
                    data_creation_counter = 0
                    try:
                        while True:
                            d = HouseholdPulse_synthetic(ground_truth_path='./data/censusHouseholdPulse_data/ipums_cleaned.csv',
                                    bias_path = './data/censusHouseholdPulse_data/pulse_week'+week+'_cleaned.csv',
                                    rngs=rngs,
                                    device=device,
                                    gt_limit = GT_LIMIT*len(rngs),
                                    bias_limit = BIAS_LIMIT, 
                                    )
                            if not d.missing_values:
                                break
                            else:
                                data_creation_counter += 1
                    except KeyboardInterrupt:
                        print("\nKeyboard interrupt received. Exiting gracefully.")

                    if type(rngs) == list:
                        rngs = rngs[0]

                    print("idealy 1k to 5k generaor updates...")
                    print("N Generator updates: ", (EPOCHS * GT_LIMIT*len(rngs)) / (BATCH_SIZE * hparams['dtrainingfactor']))

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
                                temperature=hparams["tau"],
                                warmup_length=hparams["wudur"],
                                lambda_regularizer=0,
                                generator_dropout=hparams["generator_dropout"],
                                discriminator_dropout=hparams["discriminator_dropout"],
                            )

                    synthetic = hasattr(d, 'original_distribution')
                    if synthetic: #synthetic data experiments
                        writer = SummaryWriter(comment='Variables=' + str(d.column_names) + '||Cell=' + str(d.upscaled_cell)+'||seed:'+str(rngs['seed_bias']))
                    else: #non synthetic data experiments
                        writer = SummaryWriter(comment='iter: ' + str(tid) + '||Week='+week+'||VAR = '+str(exp_var)+":"+str(cvar)+
                                            '||seed:'+str(rngs['seed_bias']))
                    
                    for key, item in hparams.items():
                        if isinstance(item, list):
                            hparams[key] = str(hparams[key])
                    
                    writer.add_hparams(hparams,{})

                    weights, bias_labels, prob_diffs,  test_probs, test_prob_diffs, generator_losses, discriminator_losses  = gan.train(BATCHS_IN_EPOCH,
                                                                                                                                EPOCHS,
                                                                                                                                TEMPERATURE_START,
                                                                                                                                TEMPERATURE_END,
                                                                                                                                hparams['gtrainingfactor'],
                                                                                                                                hparams['dtrainingfactor'],
                                                                                                                                SAVE_EVERY,
                                                                                                                                writer)
                    predicted_target = (weights @ bias_labels).item()
                    
                    #results.append((exp_var,cvar,predicted_target))

                    #with open('./results.pkl', 'wb') as file:
                    #    pickle.dump(results,file)
                    logger.info("RESULT: " + str((exp_var,cvar,predicted_target)))
                    
                    writer.close()
            
        #rename the runs folder into its appropriate name
        #if NUM_TRIALS > 1:
        #    os.rename("./runs", "./runs_week"+str(w))