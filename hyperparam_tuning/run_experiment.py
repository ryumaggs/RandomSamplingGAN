import os
import torch
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
from pathlib import Path


from Dataset import D4P_dataset
from GAN import WGAN_GP
from util import set_seed

def load_config(path=None):
    _DEFAULT_CONFIG_PATH = "./configs/default_config.yaml"
    """Load a YAML config. Falls back to default_config.yaml if no path given."""
    with open(path or _DEFAULT_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def build_rngs(rng_cfg: dict, device):
    """
    Build the rngs dict from the 'rng' block of a config.

    Modes
    -----
    fully_random  — all sources randomised, seed ignored
    fix_data      — same data each run, network re-randomised
    fix_network   — same network init, data randomised
    fully_fixed   — everything fixed
    """
    mode = rng_cfg.get("mode", "fully_random")
    seed = rng_cfg.get("seed") or int(np.random.randint(1e6))

    if mode == "fully_random":
        seed = int(np.random.randint(1e6))
        return set_seed(seed, device,
                        data_init=[False, False],
                        data_gen=False,
                        network_init=False)
    elif mode == "fix_data":
        return set_seed(seed, device,
                        data_init=[True, True],
                        data_gen=True,
                        network_init=False)
    elif mode == "fix_network":
        return set_seed(seed, device,
                        data_init=[False, False],
                        data_gen=False,
                        network_init=True)
    elif mode == "fully_fixed":
        return set_seed(seed, device,
                        data_init=[True, True],
                        data_gen=True,
                        network_init=True)
    else:
        raise ValueError(f"Unknown rng mode: '{mode}'")

def run(cfg: dict,
            dataset: object,
            rngs: object,
            phase: int = 0):
    """
    Load datasets, instantiate WGAN_GP, run training, and return results.

    Parameters
    ----------
    cfg  : dict - config loaded from a YAML file (see default_config.yaml)
        Note: this cfg has already had its experimental params updated
        in its function call in autotuner.py
    dataset: the data set pre-loaded we are trying to tune over

    Returns
    -------
    jsd_history : list[float]
        The jsd metric recorded at each step/epoch during training.
        Used by the tuner to assess convergence.
    
    kleip_history: list[float]
        The KLIEP metric recorded at each step/epoch during training.
        Used by the tuner to assess convergence.
    """
    hp = cfg["hparams"]

    # ── 2. Instantiate WGAN_GP ────────────────────────────────────────────────
    gan = WGAN_GP(
        rngs=rngs,
        dataset=dataset,
        generator_type='deepSet',
        discriminator_type='deepSet',
        gen_learning_rate=hp["glearningrate"],
        disc_learning_rate=hp["dlearningrate"],
        batch_size=hp["batch_size"],
        truth_sample_size=hp["subset_size"],
        gen_layers=hp["gen_layers"],
        disc_layers=hp["disc_layers"],
        bias_sample_size=hp["subset_size"],
        lambda_gp=hp["lambdagp"],
        lambda_weights=hp["lambdaw"],
        lambda_demo=hp["lambdad"],
        lambda_JSD=hp["lambdaJSD"],
        gen_history_length=0,
        temperature=hp["tau"],
        warmup_length=0,
        lambda_regularizer=0,
        lambda_first_layer=0,
        generator_dropout=hp["generator_dropout"],
        discriminator_dropout=hp["discriminator_dropout"],
        KLIEP_downsample=-1,
        save_dict=cfg.get("save_dict", {}),
        save_dir=cfg.get("save_dir", ""),
    )

    # ── 3. Tensorboard writer ─────────────────────────────────────────────────
    run_comment = f"phase={phase}||lambdaJSD={hp['lambdaJSD']}||lambdad={hp['lambdad']}||lambdaw={hp['lambdaw']}||seed:{rngs['seed_bias']}"
    runs_dir = cfg.get("runs_dir")
    if runs_dir:
        import os
        writer = SummaryWriter(log_dir=os.path.join(runs_dir, run_comment))
    else:
        writer = SummaryWriter(comment=run_comment)
    hparam_str = "\n".join(f"{k}: {v}" for k, v in hp.items())
    writer.add_text("hparams", hparam_str, global_step=0)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    jsd_history, kliep_history = gan.autotune_train(
            epochs = cfg['training']["epochs"],
            gen_training_factor = cfg['training']["gtrainingfactor"],
            disc_training_factor = cfg['training']["dtrainingfactor"],
            writer=writer,
        )
    

    writer.close()

    return jsd_history, kliep_history