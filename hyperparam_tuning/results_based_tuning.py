'''
This file is meant to autotune parameters in a results oriented manner:

Given a set of Datasets, {D}, find a parameter search over a given 
network architecture such that for d in {D}, the prediction error
is minimized. 

uses optuna multi-processing to train a function on each data set simultaneously
'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
BASE_DIR = Path(__file__).parent.parent

from Dataset import build_dataset
from util import build_rngs
from GAN import WGAN_GP

import tqdm
from functools import partialmethod
tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)

import optuna
import numpy as np
import yaml
import copy
import torch
from torch.utils.tensorboard import SummaryWriter
import logging

logging.basicConfig(
    filename=BASE_DIR / "hyperparam_tuning" / "optuna_log.txt",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

def log_callback(study, trial):
    value_str = f"{trial.value:.4f}" if trial.value is not None else "FAILED"
    logging.info(f"Trial {trial.number} | Value: {value_str} | Params: {trial.params}")

#global base config load for optuna objective function
with open(BASE_DIR/"configs"/"all_datasets.yaml", "r") as f:
    data_config = yaml.safe_load(f)
with open(BASE_DIR/"configs"/"default_config.yaml", "r") as f:
    base_config = yaml.safe_load(f)


# ── data ────────────────────────────────────────────────────────────────────
def load_all_datasets():
    datasets = []
    rngs = []
    for d_name, d_info in dataset_info.items():
        if d_name == 'household_pulse':
            for _ in range(base_config['training']['NUM_TRIALS']):
                local_base_config = copy.deepcopy(base_config)
                local_base_config['data']['dataset_name'] = d_name
                local_base_config['data']['weeks'] = [d_info[0]]
                rngs.append(build_rngs(local_base_config, device=d_info[1]))
                datasets.append(build_dataset(local_base_config, 
                            data_config,
                            rngs[-1],
                            week=d_info[0], 
                            device=d_info[1],))
        else:
            local_base_config = copy.deepcopy(base_config)
            local_base_config['data']['dataset_name'] = d_name
            local_base_config['data']['weeks'] = [d_info[0]]
            rngs.append(build_rngs(local_base_config, device=d_info[1]))
            datasets.append(build_dataset(local_base_config, 
                        data_config,
                        rngs[-1],
                        week=d_info[0], 
                        device=d_info[1],))
    return datasets, rngs

dataset_info = { #last element is how many unique data sets are loaded
    #'d4p': ['25', torch.device('cuda:0'), 1],
    'household_pulse': ['29', torch.device('cuda:1'), base_config['training']['NUM_TRIALS']],
} 
dataset_names = [name for name, info in dataset_info.items() for _ in range(info[2])]

datasets, all_rngs = load_all_datasets()
true_targets = [.605 for _ in range(base_config['training']['NUM_TRIALS'])]   # corresponding ground truth targets

# ── training function ────────────────────────────────────────────────────────
def train_wgan_gp(dataset, data_set_info, config, rngs):
    """
    Train WGAN-GP with fixed base params and given regularization coefficients.
    Returns array of target variable predictions over training, length T.
    """
    #create data set
    trainingparams = config['training']
    hparams = config["hparams"]
    ld, ljsd, lw = hparams['lambdad'], hparams['lambdaJSD'], hparams['lambdaw']
    hp_str = f"lambdad={ld:.4f}||lambdaJSD={ljsd:.4f}||lambdaw={lw:.4f}"
    week=data_set_info[0]
    device=data_set_info[1]
    predictions = []
    jsd_history = []
    num_trials = config['training']['NUM_TRIALS']
    if data_set_info[2] != 1: #data is loaded in pieces
        num_trials=1
    for tid in range(num_trials):
        writer = SummaryWriter(comment=f"iter:{tid}||Week={week}||{hp_str}||seed:{rngs['seed_bias']}")
        gan = WGAN_GP(
                rngs=rngs,
                dataset=dataset,
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
                lambda_JSD=hparams["lambdaJSD"],
                gen_history_length=0,
                temperature=hparams["tau"],
                warmup_length=0,
                lambda_regularizer=0,
                lambda_first_layer=0,
                generator_dropout=hparams["generator_dropout"],
                discriminator_dropout=hparams["discriminator_dropout"],
                KLIEP_downsample=-1,
                save_dict= config['SAVE_DICT'],
                save_dir = "./saves/",
                device=device,
            )

        gan_train_output = gan.train(trainingparams['epochs'],
                trainingparams['gtrainingfactor'],
                trainingparams['dtrainingfactor'],
                writer,
                tid,
                synthetic=(week=='Syn'),
                synthetic_col_names=None)
        predictions.append(gan_train_output[-1])
        jsd_history.append(gan_train_output[-2])
    
    predictions=np.array(predictions)
    jsd_history = np.array(jsd_history)
    last_means = predictions[:, int(0.8*config['training']['epochs']):].mean(axis=1)  # shape (4,)
    last_jsds = jsd_history[:, int(0.8*config['training']['epochs']):].mean(axis=1)
    predictions = np.mean(last_means)
    last_jsds = np.mean(last_jsds)
    return predictions, last_jsds  # np.array of shape (T,)

# ── optuna objective ─────────────────────────────────────────────────────────
def objective(trial):
    config = copy.deepcopy(base_config)
    config["hparams"]['lambdad'] = trial.suggest_float("lambdad", 20, 50)
    config["hparams"]['lambdaJSD']= trial.suggest_float("lambdaJSD", 10, 40)
    config["hparams"]['lambdaw']= trial.suggest_float("lambdaw", 0, 2.5)

    errors = {}
    for i, (dataset, dname, true_target) in enumerate(zip(datasets, dataset_names, true_targets)):
        predictions, last_jsds = train_wgan_gp(dataset, dataset_info[dname], config, rngs=all_rngs[i])
        #weight each error by the number of copies of the data set that are used
        if dname not in errors:
            errors[dname] = {}
            errors[dname]['target'] = abs(predictions - true_target) / dataset_info[dname][2]
            errors[dname]['jsd'] =  max(0.0, last_jsds - JSD_TARGET)
        else:
            errors[dname]['target'] += abs(predictions - true_target) / dataset_info[dname][2]
            errors[dname]['jsd'] +=  max(0.0, last_jsds - JSD_TARGET)
    total_error = sum(inner[key] for inner in errors.values() for key in inner)
    return total_error


JSD_TARGET = 0.075
# ── run study ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    T = 600
    study = optuna.create_study(direction="minimize")
    try:
        study.optimize(objective, n_trials=50, n_jobs=2, callbacks=[log_callback])
    except KeyboardInterrupt:
        print("Stopped. Best so far:", study.best_trial)

    print("Best params:", study.best_params)
    print("Best value: ", study.best_value)

    logging.info("="*50)
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best value: {study.best_value:.4f}")
    logging.info(f"Best params: {study.best_params}")
    logging.info("="*50)