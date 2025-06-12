from alg.gan2 import GAN
from dataset.pulse_dataset import PulseDataset
from utils import *
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import random
import pandas as pd
import logging
import warnings
import os
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').disabled = True

# For learning
BATCHS_IN_EPOCH = 10
BIAS_BATCH_SIZE = 100
TRUTH_BATCH_SIZE = 100
EPOCHS = 100  # the stream is infinite so one epoch will be defined as BATCHS_IN_EPOCH * BATCH_SIZE
GENERATOR_TRAINING_FACTOR = 20
DISCRIMINATOR_TRAINING_FACTOR = 1
LEARNING_RATE = 1e-5
TEMPERATURE_START = 0.1
TEMPERATURE_END = 0.1
TEMPERATURE = 0.01
LOSS_TYPE = "Wasserstein"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pulse_path = ["dataset/pulse_dataset/cleaned/week26/week26.csv"] 
#                "dataset/pulse_dataset/cleaned/week23/week23.csv"]
ipums_path = "dataset/ipums_dataset/cleaned/ipums.csv"

# logging
filename = 'Pulse_log.log'
if os.path.exists(filename):
    os.remove(filename)
    print(f"'{filename}' has been removed.")
else:
    print(f"'{filename}' does not exist.")
logger = logging.getLogger(__name__)
logging.basicConfig(filename='Pulse_log.log', level=logging.INFO)

for path in pulse_path:

    # Disable print
    #enable_print = print
    #print = lambda *args, **kwargs: None

    dataset = PulseDataset(path, ipums_path)
    gan = GAN(dataset, LEARNING_RATE, TRUTH_BATCH_SIZE, BIAS_BATCH_SIZE, TEMPERATURE, EPOCHS, BATCHS_IN_EPOCH, GENERATOR_TRAINING_FACTOR, DISCRIMINATOR_TRAINING_FACTOR, LOSS_TYPE, device)
    info, debug_info = gan.train()

    save_data(logger, "learning_rate", LEARNING_RATE)
    save_data(logger, "temperature", TEMPERATURE)
    save_data(logger, "total_iterations", BATCHS_IN_EPOCH * EPOCHS)
    save_data(logger, "training_frequency", (GENERATOR_TRAINING_FACTOR, DISCRIMINATOR_TRAINING_FACTOR))
    save_data(logger, "generator_losses", info["generator loss"])
    save_data(logger, "discriminator_losses", info["discriminator loss"])
    save_data(logger, "hard predicts", info["hard predict"])
    save_data(logger, "soft predicts", info["soft predict"])
    save_data(logger, "generator weights", debug_info["generator weights"])
    save_data(logger, "generator increment", debug_info["generator increment"])
    save_data(logger, "discriminator weights", debug_info["discriminator weights"])
    save_data(logger, "discriminator increment", debug_info["discriminator increment"])

    g_prop = []
    d_prop = []
    for ind, val in enumerate(info["generator loss"]):
        if ind != 0:
            g_prop.append(abs(val-info["generator loss"][ind-1])/info["generator loss"][ind-1])
    for ind, val in enumerate(info["discriminator loss"]):
        if ind != 0:
            d_prop.append(abs(val-info["discriminator loss"][ind-1])/info["discriminator loss"][ind-1])
    save_data(logger, "generator rate", g_prop)
    save_data(logger, "discriminator rate", d_prop)

    # Re-enable print
    #print = enable_print

    print("Predict for this week:", gan.predict(10))


