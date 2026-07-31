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
import hyperparam_tuning.run_experiment as run_experiment
import copy
import torch

import main.Dataset

JSD_MAX = 0.075
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
            jsd_history, kliep_history, _ = run_experiment.run(cfg = phase_3_temp_cfg,
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
    all_target_history = []
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

        _, _, target_history = run_experiment.run(cfg = analysis_temp_cfg,
                                    dataset = d,
                                    rngs = rngs,
                                    phase=4)

        target_history = np.array(target_history)
        print(target_history)
        last_20_target = target_history[int(0.8*cfg['training']['epochs']):].mean(axis=0)  # shape (4,)
        all_target_history.append(last_20_target)
    log.info("All experiment results raw: " + str(all_target_history))
    log.info("Average prediction: " + str(np.mean(all_target_history)))


def notebook_tune_script():
    import sys
    from pathlib import Path
    import yaml
    BASE_DIR = Path().resolve()
    sys.path.insert(0, str(BASE_DIR))

    from hyperparam_tuning.run_experiment import load_config, run, build_rngs
    import hyperparam_tuning.autotuner as autotuner
    from main.Dataset import build_dataset

    #set which configs to use
    cfg = load_config(BASE_DIR / 'configs' / 'product_default_config.yaml')
    data_cfg = load_config(BASE_DIR / 'configs' / 'all_datasets.yaml')

    #variable set up
    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    rngs = build_rngs(cfg["rng"], device)
    w = cfg['data']['weeks'][0]

    print("Operating on: ", cfg['data']['dataset_name'], " | week: ", cfg['data']['weeks'])

    #build data set object
    d = build_dataset(cfg, 
                    data_cfg,
                    rngs,
                    w, 
                    device,)

    #tune on data set object
    #variable ranges are given as: [start, end, step]
    tune_results = autotuner.autotune({
        'lambdaJSD': [0, 50.0, 5.0],
        'lambdad':   [0, 50.0, 5.0],
        'lambdaw': [0.0,2,0.2]
    }, cfg, d, rngs,
    skip_mode = False,)

    del d

    new_cfg = copy.deepcopy(cfg)
    for k, v in tune_results.items():
        new_cfg['hparams'][k] = v
    filename = f"tuned_config_{new_cfg['data']['dataset_name']}_week{new_cfg['data']['weeks'][0]}.yaml"
    with open('./configs/'+filename, 'w') as f:
        yaml.dump(new_cfg, f)

    #analysis
    num_analysis = 5
    analysis_rngs = [build_rngs(cfg["rng"], device) for _ in range(num_analysis)]
    autotuner.analysis(tune_results, new_cfg, data_cfg, analysis_rngs, device)