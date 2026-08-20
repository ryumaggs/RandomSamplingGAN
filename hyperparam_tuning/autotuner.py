"""
autotuner.py

Sequential hyperparameter tuning for lambdaJSD and lambdaTVD.

Phase 1: Increase lambdaJSD until no further meaningful improvement
         in the settled jsd metric.
Phase 2: With best lambdaJSD fixed, increase lambdaTVD until the
         settled jsd metric begins to degenerate (rise).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
BASE_DIR = Path(__file__).parent.parent

import numpy as np
import os
import logging
from datetime import datetime
import run_experiment
import copy
import torch

import Dataset

JSD_MAX = 0.07
# ── Helpers ───────────────────────────────────────────────────────────────────

def final_quarter_mean(jsd_history):
    """
    Given a list/array of jsd values recorded over training,
    return the mean of the final quarter.
    """
    arr = np.array(jsd_history)
    q = max(1, len(arr) // 4)
    return float(arr[-q:].mean())

def first_quarter_mean(history):
    arr = np.array(history)
    q = max(1, len(arr) // 4)
    return float(arr[:q].mean())


def phase1_should_stop(history):
    """
    TODO: fill in your Phase 1 stopping condition.

    Called after each Phase 1 experiment. Return True to stop increasing
    lambdaJSD (stabilization detected), False to keep going.

    Parameters
    ----------
    history : list[float]
        final_quarter_mean values collected so far, one per experiment,
        in the order they were run.

    Returns
    -------
    bool
    """
    return history[-1] <= JSD_MAX


def phase2_should_stop(history, first_q_history_2, history_2):
    """
    phase2 is increasing KLIEP. we need to make sure that
    1. JSD does not degenerate beyond 0.06
    2. KLIEP has an appropriate shape of improvement even if only by a little

    Returns
    -------
    bool
    """
    if history[-1] > JSD_MAX: #and first_q_history_2[-1] > history_2[-1]:
        return True, first_q_history_2[-1] > history_2[-1]
    else:
        return False, False


def _setup_logger(run_dir):
    """Create a logger that writes to autotuneLogger.txt inside run_dir."""
    log_path = os.path.join(run_dir, "autotuneLogger.txt")
    logger = logging.getLogger("autotuner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    return logger


# ── Main tuning loop ──────────────────────────────────────────────────────────

def autotune(param_ranges, cfg, dataset, rngs,
             skip_mode=False):
    """
    Run the two-phase sequential tuning procedure.

    Parameters
    ----------
    param_ranges : dict
        Keys are parameter names, values are [init, end, step] (end inclusive).
        Must contain exactly two keys in order:
          - Phase 1 parameter (e.g. 'lambdaJSD')
          - Phase 2 parameter (e.g. 'lambdad')

        Example:
            {
                'lambdaJSD': [1.0, 15.0, 2.0],
                'lambdad':   [1.0, 30.0, 5.0],
            }
    skip_mode: bool - only used to in testing of phase 4 (analysis)
        will set up directories and log files but will not run any
        actual experiments (all "best" values are pre-set in this mode)
    Returns
    -------
    results : dict  {param_name: best_value}
    """
    (p1_name, (p1_init, p1_end, p1_step)), \
    (p2_name, (p2_init, p2_end, p2_step)), \
    (p3_name, (p3_init, p3_end, p3_step)) = param_ranges.items()

    # ── Directory setup ───────────────────────────────────────────────────────
    base_dir = BASE_DIR / "hyperparam_tuning" / "autoTuneDir"
    os.makedirs(base_dir, exist_ok=True)

    dataset_stem = cfg['data']['dataset_name']
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{dataset_stem}"
    run_dir = base_dir / run_name
    os.makedirs(run_dir, exist_ok=True)

    log = _setup_logger(run_dir)
    log.info(f"autotune started — run dir: {run_dir}")

    # point all tensorboard runs into this autotuner session's subdirectory
    cfg["runs_dir"] = os.path.join(run_dir, "runs")
    log.info(f"  {p1_name}: init={p1_init}, end={p1_end}, step={p1_step}")
    log.info(f"  {p2_name}: init={p2_init}, end={p2_end}, step={p2_step}")

    if not skip_mode:
        # ── Phase 1: tune p1_name ─────────────────────────────────────────────────
        log.info(f"=== Phase 1: tuning {p1_name} ===")
        p1_val        = p1_init
        p1_fq_history = []
        p2_fixed      = 0.0
        while p1_val <= p1_end + 1e-9:
            
            log.info(f"  Running: {p1_name}={p1_val:.4f}, {p2_name}={p2_fixed:.4f}")

            phase_1_temp_cfg = copy.deepcopy(cfg)
            phase_1_temp_cfg['hparams']['lambdaJSD'] = p1_val
            jsd_history, kliep_history = run_experiment.run(cfg = phase_1_temp_cfg,
                                        dataset = dataset,
                                        rngs = rngs,
                                        phase=1)

            fq_mean = final_quarter_mean(jsd_history)
            p1_fq_history.append(fq_mean)
            log.info(f"  final-quarter mean jsd = {fq_mean:.4f}")

            p1_val += p1_step

            if len(p1_fq_history) >= 2 and phase1_should_stop(p1_fq_history):
                log.info(f"  Phase 1 stabilized at {p1_name}={p1_val:.4f}")
                break

        best_p1 = p1_val - p1_step
        log.info(f"  Best {p1_name} = {best_p1:.4f}")
    else:
        best_p1 = 22
    if not skip_mode:
        # ── Phase 2: tune p2_name ─────────────────────────────────────────────────
        log.info(f"=== Phase 2: tuning {p2_name} ===")
        p2_val        = p2_init
        p2_fq_history = []
        p2_firstq_history_2 = []
        p2_fq_history_2 = []
        

        while p2_val <= p2_end + 1e-9:
            log.info(f"  Running: {p1_name}={best_p1:.4f}, {p2_name}={p2_val:.4f}")

            phase_2_temp_cfg = copy.deepcopy(cfg)
            phase_2_temp_cfg['hparams']['lambdaJSD'] = best_p1
            phase_2_temp_cfg['hparams']['lambdad'] = p2_val
            jsd_history, kliep_history = run_experiment.run(cfg = phase_2_temp_cfg,
                                        dataset = dataset,
                                        rngs = rngs,
                                        phase=2)

            fq_mean = final_quarter_mean(jsd_history)
            firstq_mean_2 = first_quarter_mean(kliep_history)
            fq_mean_2 = final_quarter_mean(kliep_history)

            p2_fq_history.append(fq_mean)
            p2_firstq_history_2.append(firstq_mean_2)
            p2_fq_history_2.append(fq_mean_2)
            log.info(f"  final-quarter mean jsd = {fq_mean:.4f}, kliep = {fq_mean_2:.4f}")

            jsd_degenerate, kleip_improvement = phase2_should_stop(p2_fq_history,
                                                              p2_firstq_history_2,
                                                              p2_fq_history_2)
            
            if len(p2_fq_history) >= 2 and jsd_degenerate:
                log.info(f"  Phase 2 degeneration detected at {p2_name}={p2_val:.4f}")
                if not kleip_improvement:
                    log.error(f" Phase 2 jsd degenerate but KLEIP no improve at {p2_name}={p2_val:.4f}")
                break

            p2_val += p2_step

        best_p2 = p2_val - p2_step
        log.info(f"  Best {p2_name} = {best_p2:.4f}")
    else:
        best_p2 = 40.0
    
    if not skip_mode:
        # ── Phase 3: tune p3_name ─────────────────────────────────────────────────
        log.info(f"=== Phase 3: tuning {p3_name} ===")
        p3_val        = p3_init
        p3_fq_history = []
        p3_firstq_history_2 = []
        p3_fq_history_2 = []
        p3_early_end = False
        while p3_val <= p3_end + 1e-9:
            log.info(f"  Running: {p1_name}={best_p1:.4f}, {p2_name}={best_p2:.4f}, {p3_name}={p3_val:.4f}")

            phase_3_temp_cfg = copy.deepcopy(cfg)
            phase_3_temp_cfg['hparams']['lambdaJSD'] = best_p1
            phase_3_temp_cfg['hparams']['lambdad'] = best_p2
            phase_3_temp_cfg['hparams']['lambdaw'] = p3_val
            jsd_history, kliep_history = run_experiment.run(cfg = phase_3_temp_cfg,
                                        dataset = dataset,
                                        rngs = rngs,
                                        phase=3)

            fq_mean = final_quarter_mean(jsd_history)
            firstq_mean_2 = first_quarter_mean(kliep_history)
            fq_mean_2 = final_quarter_mean(kliep_history)

            p3_fq_history.append(fq_mean)
            p3_firstq_history_2.append(firstq_mean_2)
            p3_fq_history_2.append(fq_mean_2)
            log.info(f"  final-quarter mean jsd = {fq_mean:.4f}, kliep = {fq_mean_2:.4f}")

            jsd_degenerate, kleip_improvement = phase2_should_stop(p3_fq_history,
                                                                p3_firstq_history_2,
                                                                p3_fq_history_2)

            if len(p3_fq_history) >= 2 and jsd_degenerate:
                if not kleip_improvement:
                    log.error(f" Phase 3 jsd degenerate but KLEIP no improve at {p3_name}={p3_val:.4f}")
                p3_early_end = True
                break

            p3_val += p3_step
        if p3_early_end:
            best_p3 = p3_val - p3_step
            log.info(f"  Best {p3_name} = {best_p3:.4f}, ended early")
        else:
            best_p3 = 0.0
            log.info(f"  Best {p3_name} = {best_p3:.4f}, DID NOT END")
    else:
        best_p3 = 0.0
    
    if not skip_mode:
        log.info("=== Tuning complete ===")
        log.info(f"  best {p1_name} = {best_p1:.4f}")
        log.info(f"  best {p2_name} = {best_p2:.4f}")
        log.info(f"  best {p3_name} = {best_p3:.4f}")

    return {p1_name: best_p1, p2_name: best_p2, p3_name: best_p3}

def analysis(param_dict, cfg, 
             data_cfg,
             all_exp_rngs,
             device):
    '''
    called after autotune to run num_experiments on the tuned parameters
    defined by param_dict

    cfg and dataset are pre-created in the main call
    '''
    log = logging.getLogger("autotuner")
    log.info(f"=== Phase 4: analysis ===")
    log.info("Analyzing: " + str(param_dict))
    num_experiments = len(all_exp_rngs)
    analysis_temp_cfg = copy.deepcopy(cfg)
    for name, value in param_dict.items():
        analysis_temp_cfg['hparams'][name] = value
    
    for ne in range(num_experiments):
        rngs = all_exp_rngs[ne]
        d = Dataset.build_dataset(cfg, 
                                data_cfg,
                                rngs,
                                cfg['data']['weeks'][0], 
                                device,)

        _ = run_experiment.run(cfg = analysis_temp_cfg,
                                    dataset = d,
                                    rngs = rngs,
                                    phase=4)

    


if __name__ == "__main__":
    from run_experiment import load_config
    from Dataset import D4P_dataset, HouseholdPulse_dataset

    
    for week in range(29,30):
        w = str(week)
        cfg = load_config(BASE_DIR / 'configs' / 'HHP_compressed.yaml')
        data_cfg = load_config(BASE_DIR / 'configs' / 'all_datasets.yaml')
        cfg['data']['weeks'] = [w]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rngs = run_experiment.build_rngs(cfg["rng"], device)
        if True:
            d = Dataset.build_dataset(cfg, 
                                        data_cfg,
                                        rngs,
                                        w, 
                                        device,)

            #autotune(param_ranges, cfg, dataset, rngs)
            tune_results = autotune({
                'lambdaJSD': [10.0, 30, 2.0],
                'lambdad':   [20.0, 50.0, 5.0],
                'lambdaw': [0.0,2,0.2]
            }, cfg, d, rngs,
            skip_mode = False)

            del d
        else:
            all_tuned_results = {
                "22": {"lambdaJSD": 18.0, "lambdad": 45.0, "lambdaw": 0.0},
                "23": {"lambdaJSD": 20.0, "lambdad": 50.0, "lambdaw": 0.0},
                "24": {"lambdaJSD": 18.0, "lambdad": 50.0, "lambdaw": 0.0},
                "25": {"lambdaJSD": 20.0, "lambdad": 45.0, "lambdaw": 0.0},
                "26": {"lambdaJSD": 18.0, "lambdad": 45.0, "lambdaw": 0.0},
                "27": {"lambdaJSD": 18.0, "lambdad": 40.0, "lambdaw": 0.0},
                "28": {"lambdaJSD": 20.0, "lambdad": 50.0, "lambdaw": 0.0},
                "29": {"lambdaJSD": 18.0, "lambdad": 40.0, "lambdaw": 0.0},
            }
            tune_results = all_tuned_results[w]
        #analysis
        num_experiments = 3
        analysis_rngs = [run_experiment.build_rngs(cfg["rng"], device) for _ in range(num_experiments)]
        analysis(tune_results, cfg, data_cfg, analysis_rngs, device)
