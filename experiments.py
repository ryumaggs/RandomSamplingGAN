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
import sys
sys.path.insert(0, '/scratch/ryu1/RandomSamplingGAN')
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='train_log.log', level=logging.INFO)

import Dataset
from GAN import GAN, WGAN_GP, WGAN_GP_fict
from Discriminator import DataDiscriminator
from util import set_seed, build_rngs, load_config, cartesian_product_hyperparams
import itertools
import pandas as pd
#optimize over w directly, dont use any theta shenannigans. 
from sklearn import preprocessing
import pytorch_warmup as warmup
from data_processing.DataProcessing import *
import pickle
from itertools import combinations
from attribution.attribution import *
#NEW TRAINING VARIABLE SET UP
# For learning
#runs_consistency_SameData_SameSeen_diffNetwork,Seed:359556
'''
Best params: {'lambdad': 47.06726125301136, 'lambdaJSD': 32.03099184880099, 'lambdaw': 0.011258598476026149}
Best value:  0.032357343743603884


DEBUG_MODE = False
if DEBUG_MODE:
    DEFAULTS.update({
        "batch_size":    2,
        "subset_size":   2,
        "dtrainingfactor": 2,
        "epochs":        30,
        "warmup_epochs": 10,
    })
    BIAS_LIMIT = 5
    NUM_TRIALS = 1
'''
cfg = load_config('./configs/default_config.yaml')
data_cfg = load_config('./configs/all_datasets.yaml')
device = torch.device(cfg['device'])

if __name__ == "__main__":
    weeks = cfg['data']['weeks']
    SAVE_DICT = cfg['SAVE_DICT']
    NUM_TRIALS = cfg['training']['NUM_TRIALS']
    experimental_params = cfg['experimental_hparams']
    for w_idx, w in enumerate(weeks):
        week = str(w)
        main_dir = ""
        if any(SAVE_DICT.values()):
            main_dir = "./saves_week"+str(w)
            if not os.path.exists(main_dir):
                os.makedirs(main_dir)
                print(f"Directory '{main_dir}' created.")
            else:
                print(f"Directory '{main_dir}' already exists.")
        
        if cfg['experimental_hparams'] is not None:
            list_var_updates = cartesian_product_hyperparams(cfg['experimental_hparams'],
                                                    product=cfg['cartesian_product'])
        else:
            list_var_updates = [{'lambdaw': cfg['hparams']['lambdaw']}]

        for var_update in list_var_updates:
            for tid in range(NUM_TRIALS):
                rngs = build_rngs(cfg, device=device)

                trial_dir = ""
                if any(SAVE_DICT.values()):
                    trial_dir = os.path.join(main_dir,"trial:"+str(tid))
                    if not os.path.exists(trial_dir):
                        os.makedirs(trial_dir)
                        print(f"Directory '{trial_dir}' created.")
                    else:
                        print(f"Directory '{trial_dir}' already exists.")

                #copy cfg and update with experimental params
                local_cfg = copy.deepcopy(cfg)
                for v,v_val in var_update.items():
                    local_cfg['hparams'][v] = v_val
                hparams=local_cfg['hparams']
                trainingparams=local_cfg['training']
                    
                d = Dataset.build_dataset(local_cfg, 
                                        data_cfg,
                                        rngs,
                                        week, 
                                        device,)
                gan = WGAN_GP(
                            rngs=rngs,
                            dataset=d,
                            cfg=local_cfg,
                            save_dir = trial_dir,
                        )

                synthetic = (weeks[0] == 'SYN')
                if synthetic: #synthetic data experiments
                    writer = SummaryWriter(comment='Variables=' + str(d.column_names) + '||Cell=' + str(d.upscaled_cell)+'||seed:'+str(rngs['seed_bias']))
                else: #non synthetic data experiments
                    writer = SummaryWriter(comment='iter: ' + str(tid) + '||Week='+week+'||VAR = '+str(var_update)+
                                        '||seed:'+str(rngs['seed_bias']))
                
                for key, item in hparams.items():
                    if isinstance(item, list):
                        hparams[key] = str(hparams[key])
                
                writer.add_hparams(hparams,{})
                c_names = ""
                if week == "Syn":
                    c_names = d.column_names
                
                _ = gan.train(trainingparams['epochs'],
                            trainingparams['gtrainingfactor'],
                            trainingparams['dtrainingfactor'],
                            writer,
                            tid,
                            synthetic=(week=='Syn'),
                            synthetic_col_names=c_names)
                writer.close()
        