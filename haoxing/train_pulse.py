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
BIAS_BATCH_SIZE = 1000
TRUTH_BATCH_SIZE = 1000
EPOCHS = 500  # the stream is infinite so one epoch will be defined as BATCHS_IN_EPOCH * BATCH_SIZE
GENERATOR_TRAINING_FACTOR = 1
DISCRIMINATOR_TRAINING_FACTOR = 1
LEARNING_RATE = 1e-5
TEMPERATURE_START = 0.1
TEMPERATURE_END = 0.1
TEMPERATURE = 0.1
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
    gan = GAN(dataset, LEARNING_RATE, TRUTH_BATCH_SIZE, BIAS_BATCH_SIZE, TEMPERATURE, EPOCHS, BATCHS_IN_EPOCH, GENERATOR_TRAINING_FACTOR, DISCRIMINATOR_TRAINING_FACTOR, device)
    generator_losses, discriminator_losses, predicts = gan.train()

    save_data(logger, "learning_rate", LEARNING_RATE)
    save_data(logger, "total_iterations", BATCHS_IN_EPOCH * EPOCHS)
    save_data(logger, "training_frequency", (GENERATOR_TRAINING_FACTOR, DISCRIMINATOR_TRAINING_FACTOR))
    save_data(logger, "generator_losses", generator_losses)
    save_data(logger, "discriminator_losses", discriminator_losses)
    save_data(logger, "predicts", predicts)

    # Re-enable print
    #print = enable_print

    print("Predict for this week:", gan.predict(10))


