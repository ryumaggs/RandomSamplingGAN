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
from attribution import *
#NEW TRAINING VARIABLE SET UP
# For learning
#runs_consistency_SameData_SameSeen_diffNetwork,Seed:359556
DEBUG_MODE = False

BATCH_SIZE = 2
SUBSET_SIZE = 32
GENERATOR_TRAINING_FACTOR = 1
DISCRIMINATOR_TRAINING_FACTOR = 8
GT_LIMIT = 50000
BIAS_LIMIT = 2500
SAVE_DICT = {}
SAVE_DICT['SAVE_IG'] = False
SAVE_DICT['SAVE_DATASET'] = False
SAVE_DICT['SAVE_WEIGHTS'] = False
SAVE_DICT['SAVE_GENERATOR'] = False
SAVE_DICT['SAVE_CRITIC'] = False
SAVE_DICT['SAVE_EVERY'] = False #-1 means only save the last one

KLIEP_DOWNSAMPLE = BIAS_LIMIT
LAMBDAGP = 10 #hold this at 10. higher = less discriminator expressability
LAMBDAW = 0 #.003 #0.00075 #.00075 for 10k .006 for 2.5k
LAMBDAD = 15 #13.5
LAMBDA_FIRST_LAYER = 0
GENERATOR_LEARNING_RATE = 1e-5 #2e-3 or 1e-5
DISCRIMINATOR_LEARNING_RATE = 5e-5 #1e-3 or 5e-5
BATCHS_IN_EPOCH = 1
EPOCHS = 300
NUM_TRIALS = 1
GEN_HISTORY_LENGTH = 0
WARMUP_EPOCHS = 0
GEN_LAYERS = [1024, 1024, 1024] #[256 for _ in range(5)]
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


SAME_DATA_GT = False
SAME_DATA_BIAS = False
SAME_DATA_SEEN = False
SAME_NETWORK_INIT = True
'''
what i need this seed to do is to be able to recreate an experiment while changing some small
aspect of the randomness.

e.g. keep everyting the same but use a different GT subset of the GT data. 

Therefore fixed_seed should be a single seed and should be appended
to all rngs NUM_TRIALS times. 


'''
fixed_seed = np.random.randint(1e6)
all_rngs = []
print('SETTING SEED: ', fixed_seed)
for _ in range(NUM_TRIALS):
    all_rngs.append(set_seed(fixed_seed,
                            device,
                        data_init=[SAME_DATA_GT, SAME_DATA_BIAS],
                        data_gen =SAME_DATA_SEEN,
                        network_init=SAME_NETWORK_INIT,))

SAVE_EVERY = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# For dataset    

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

#weeks = list(range(8,9))
#weeks = ['Syn']
#weeks = ['ALL']
#weeks = ['lucid']
weeks = list(range(25,26)) #for testing on HHP
#weeks = [35, 37, 39, 40, 42, 43, 44, 45] #for axios ipsos
#weeks = [20,21,22,23,24,25] #for d4p
#weeks = [45]
if __name__ == "__main__":
    for w in weeks:
        #make a new runs folder
        week = str(w)
        exp_vars = {}
        #exp_vars['disc_layers'] = [[256], [256, 256], [256, 256, 256], [512], [512,512], [512,512,512],[1024], [1024,1024], [1024,1024,1024]]
        #exp_vars['gen_layers'] = [[1024]]
        #exp_vars['gen_layers'] = [[256]]
        #exp_vars['learning_rates'] = [5e-5, 1e-5, 5e-6, 1e-6]
        #exp_vars['warmup_duration'] = [1000, 2000, 3000, 4000]
        #exp_vars['discriminator_dropout'] = [0.1, 0.2, 0.3]
        #exp_vars['dtrainingfactor'] = [8]
        #exp_vars['gtrainingfactor'] = [1]
        #exp_vars['warmup_epochs'] = [0]
        #exp_vars['lambdagp'] = [10]
        #exp_vars['dlearningrate'] = [5e-6]
        #exp_vars['glearningrate'] = [5e-6]
        #exp_vars['gt_limit'] = [GT_LIMIT]
        #exp_vars['bias_limit'] = [2500]
        #exp_vars['subset_size'] = [32,64,128,256]
        #exp_vars['gen_training_factor'] = [2]
        #exp_vars['batch_size'] = [1,2]
        #exp_vars['lambdaw'] = [.0025, .005, .0075]
        #exp_vars['tau'] = [0.05, 0.25, 0.55, 0.75, 0.95]
        #exp_vars['columns_to_keep'] = [('EDUC', 'INCTOT', 'AGE')] #['REGION', 'EDUC', 'INCTOT', 'SEX', 'MARST', 'FAMSIZE', 'RACE','AGE', 'BIDENPERC']
        exp_vars['lambdad'] = [0,5,10,12.5,15,17.5,20]
        #exp_vars['KLIEP_downsample'] = [5000]
        #exp_vars['lambda_first_layer'] = [0]
        #exp_vars['epochs'] = [1000]
        old_g = None

        main_dir = ""
        if any(SAVE_DICT.values()):
            main_dir = "./saves_week"+str(w)
            if not os.path.exists(main_dir):
                os.makedirs(main_dir)
                print(f"Directory '{main_dir}' created.")
            else:
                print(f"Directory '{main_dir}' already exists.")

        if 'columns_to_keep' in exp_vars and len(exp_vars['columns_to_keep']) == 0:
            # Generate power set up to size 2
            power_set = []
            for r in [1,2,3,4,5,6,7,8]:
                power_set.extend(combinations(exp_vars['columns_to_keep'], r))
            
            exp_vars['columns_to_keep'] = power_set
        for exp_var,val in exp_vars.items():
            for tid in range(NUM_TRIALS):
                rngs = all_rngs[tid]

                trial_dir = ""
                if any(SAVE_DICT.values()):
                    trial_dir = os.path.join(main_dir,"trial:"+str(tid))
                    if not os.path.exists(trial_dir):
                        os.makedirs(trial_dir)
                        print(f"Directory '{trial_dir}' created.")
                    else:
                        print(f"Directory '{trial_dir}' already exists.")

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
                        "epochs": EPOCHS,
                        "warmup_epochs": WARMUP_EPOCHS,
                        "lambda_first_layer": LAMBDA_FIRST_LAYER,
                        "KLIEP_downsample": KLIEP_DOWNSAMPLE,
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
                    #week='ALL'
                    
                    

                    d = HouseholdPulse_dataset(ground_truth_path='./data/censusHouseholdPulse_data/cleaned/ipums_cleaned_combined.csv',
                                    bias_path = './data/censusHouseholdPulse_data/cleaned/pulse_week'+week+'_cleaned.csv',
                                    rngs=rngs,
                                    label_information={'RECVDVACC':0}, #{'VAC':0, 'HLTHINS1':1, 'RECVDVACC},
                                    device=device,
                                    columns_to_keep = None,
                                    gt_limit = GT_LIMIT,
                                    bias_limit = BIAS_LIMIT,
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
                                save_dict=SAVE_DICT,
                                save_dir = trial_dir,
                            )

                    synthetic = (weeks[0] == 'SYN')
                    if synthetic: #synthetic data experiments
                        writer = SummaryWriter(comment='Variables=' + str(d.column_names) + '||Cell=' + str(d.upscaled_cell)+'||seed:'+str(rngs['seed_bias']))
                    else: #non synthetic data experiments
                        writer = SummaryWriter(comment='iter: ' + str(tid) + '||Week='+week+'||VAR = '+str(exp_var)+":"+str(cvar)+
                                            '||seed:'+str(rngs['seed_bias']))
                    
                    for key, item in hparams.items():
                        if isinstance(item, list):
                            hparams[key] = str(hparams[key])
                    
                    writer.add_hparams(hparams,{})
                    c_names = ""
                    if week == "Syn":
                        c_names = d.column_names
                    
                    #print(d.column_names)
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
                                                                                                                                synthetic=(week=='Syn'),
                                                                                                                                synthetic_col_names=c_names)
                    #predicted_target = (weights @ bias_labels).item()
                    
                    #results.append((exp_var,cvar,predicted_target))
                    writer.close()
            
        #rename the runs folder into its appropriate name
        #if NUM_TRIALS > 1:
        #    os.rename("./runs", "./runs_week"+str(w))