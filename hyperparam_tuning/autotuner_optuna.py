"""
autotuner_optuna.py

Optuna-based joint hyperparameter tuning for lambdaJSD, lambdad, and lambdaw.
All three parameters are searched simultaneously in a single study.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
BASE_DIR = Path(__file__).parent.parent

import numpy as np
import os
import logging
import copy
import torch
from datetime import datetime

import optuna
import run_experiment
import Dataset


# ── Helpers ───────────────────────────────────────────────────────────────────

def final_quarter_mean(history):
    arr = np.array(history)
    q = max(1, len(arr) // 4)
    return float(arr[-q:].mean())


def first_quarter_mean(history):
    arr = np.array(history)
    q = max(1, len(arr) // 4)
    return float(arr[:q].mean())


def _setup_logger(run_dir):
    log_path = os.path.join(run_dir, "autotuneLogger.txt")
    logger = logging.getLogger("autotuner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    return logger


# ── Objective ─────────────────────────────────────────────────────────────────

def _make_objective(cfg, dataset, rngs, param_ranges, log):
    """Single objective over all three parameters jointly."""
    (p1_name, (p1_init, p1_end, p1_step)), \
    (p2_name, (p2_init, p2_end, p2_step)), \
    (p3_name, (p3_init, p3_end, p3_step)) = param_ranges.items()

    def objective(trial):
        p1_val = trial.suggest_float(p1_name, p1_init, p1_end, step=p1_step)
        p2_val = trial.suggest_float(p2_name, p2_init, p2_end, step=p2_step)
        p3_val = trial.suggest_float(p3_name, p3_init, p3_end, step=p3_step)

        trial_cfg = copy.deepcopy(cfg)
        trial_cfg['hparams']['lambdaJSD'] = p1_val
        trial_cfg['hparams']['lambdad']   = p2_val
        trial_cfg['hparams']['lambdaw']   = p3_val

        jsd_history, kliep_history = run_experiment.run(
            cfg=trial_cfg, dataset=dataset, rngs=rngs
        )

        fq_jsd       = final_quarter_mean(jsd_history)
        fq_kliep     = final_quarter_mean(kliep_history)
        firstq_kliep = first_quarter_mean(kliep_history)

        log.info(f"  [{p1_name}={p1_val:.4f}, {p2_name}={p2_val:.4f}, {p3_name}={p3_val:.4f}] "
                 f"fq_jsd={fq_jsd:.4f}, fq_kliep={fq_kliep:.4f}")

        # ── TODO: fill in your objective ──────────────────────────────────────
        # Return the scalar value Optuna should optimise.
        # Available variables: fq_jsd, fq_kliep, firstq_kliep
        # Example: return fq_jsd
        raise NotImplementedError("Fill in objective")
        # ─────────────────────────────────────────────────────────────────────

    return objective


# ── Main tuning loop ──────────────────────────────────────────────────────────

def autotune(param_ranges, cfg, dataset, rngs,
             n_trials=30,
             direction="minimize"):
    """
    Joint Optuna tuning over all three parameters in a single study.

    Parameters
    ----------
    param_ranges : dict  {param_name: [init, end, step]}
        Must have exactly three keys: lambdaJSD, lambdad, lambdaw.
    cfg : dict
    dataset : object
    rngs : object
    n_trials : int
        Total number of Optuna trials.
    direction : str
        "minimize" or "maximize".

    Returns
    -------
    dict  {param_name: best_value}
    """
    # ── Directory and logger setup ────────────────────────────────────────────
    base_dir = "./autoTuneDir"
    os.makedirs(base_dir, exist_ok=True)

    dataset_stem = os.path.splitext(os.path.basename(cfg["survey_path"]))[0]
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{dataset_stem}"
    run_dir  = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log = _setup_logger(run_dir)
    log.info(f"autotune (optuna) started — run dir: {run_dir}")
    for name, (init, end, step) in param_ranges.items():
        log.info(f"  {name}: [{init}, {end}, step={step}]")
    log.info(f"  n_trials={n_trials}, direction={direction}")

    cfg["runs_dir"] = os.path.join(run_dir, "runs")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction=direction)
    study.optimize(
        _make_objective(cfg, dataset, rngs, param_ranges, log),
        n_trials=n_trials,
    )

    best = study.best_params
    log.info("=== Tuning complete ===")
    for name, val in best.items():
        log.info(f"  best {name} = {val:.4f}")
    log.info(f"  best objective = {study.best_value:.4f}")

    return best


def analysis(param_dict, cfg, all_exp_rngs):
    """
    Called after autotune to run num_experiments on the tuned parameters.
    """
    from Dataset import HouseholdPulse_dataset
    log = logging.getLogger("autotuner")
    log.info("=== Phase 4: analysis ===")
    log.info("Analyzing: " + str(param_dict))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    analysis_temp_cfg = copy.deepcopy(cfg)
    for name, value in param_dict.items():
        analysis_temp_cfg['hparams'][name] = value
    data_cfg = load_config(BASE_DIR / 'configs' / 'all_datasets.yaml')
    cfg['data']['weeks'] = [w]
    
    for ne in range(len(all_exp_rngs)):
        rngs = all_exp_rngs[ne]
        d = Dataset.build_dataset(cfg, 
                                    data_cfg,
                                    rngs,
                                    w, 
                                    device,)
        _ = run_experiment.run(cfg=analysis_temp_cfg, dataset=d, rngs=rngs, phase=4)


if __name__ == "__main__":
    from run_experiment import load_config
    from Dataset import HouseholdPulse_dataset

    for week in range(29, 30):
        w = str(week)
        cfg = load_config(BASE_DIR / 'configs' / 'default_config.yaml')
        data_cfg = load_config(BASE_DIR / 'configs' / 'all_datasets.yaml')
        cfg['data']['weeks'] = [w]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rngs = run_experiment.build_rngs(cfg["rng"], device)

        d = Dataset.build_dataset(cfg, 
                                data_cfg,
                                rngs,
                                w, 
                                device,)

        tune_results = autotune({
            'lambdaJSD': [10.0, 30.0, 2.0],
            'lambdad':   [20.0, 50.0, 5.0],
            'lambdaw':   [0.0,  2.0,  0.2],
        }, cfg, d, rngs,
        n_trials=30,
        direction="minimize")
