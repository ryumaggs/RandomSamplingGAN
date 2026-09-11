"""
hyperparam_tuning/analysis.py

Analysis utilities for hyperparameter-tuning experiments produced by
autotuner.py, stored under hyperparam_tuning/autoTuneDir/<experiment>/.

Each experiment folder contains one or more trial_N/ subdirectories (one per
autotune() call), each with an autotuneLogger.txt log and a runs/ directory
of tensorboard event folders (one per autotuner phase/hyperparam combo, see
autotuner.py's phase 1-3 tuning loop and phase 4 analysis() call), plus a
batch_summary.txt listing the tuned params each trial landed on.

This is distinct from the root-level analysis.py, which analyzes
experiments.py's independent training runs.
"""
import ast
import os
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
BASE_DIR = Path(__file__).parent.parent

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Scalars tracked during autotuner.py's Phase 4 (final testing) runs.
PHASE4_SCALARS = [
    'RECVDVACC prediction',
    'jsd',
    'tvd',
    'Truth - Fake scores',
    'Gen Entropy',
]


# ── Tensorboard loading ─────────────────────────────────────────────────────

def load_tensorboard_runs(runs_dir, var_names, filter_by=None, num_runs=None):
    """
    Load scalar histories from tensorboard run folders directly under runs_dir.

    Modeled on the root analysis.py's load_runs_as_numpy.

    Parameters
    ----------
    runs_dir : str or Path
        Directory containing one subfolder per tensorboard run (each holding
        an events.out.tfevents.* file), e.g. a trial's "runs" directory.
    var_names : list[str]
        Scalar tag names to extract.
    filter_by : list[str] or None
        Only run folders whose name contains every string in filter_by are
        loaded (substring match, same semantics as the root version).
        None/[] means no filtering.
    num_runs : int or None
        Cap on the number of matching run folders to load. None = no cap.

    Returns
    -------
    data : dict[str, list[list[float]]]
        var_name -> one scalar history (list of floats) per matched run that
        recorded that scalar.
    run_folders : list[str]
        The run folder paths that were loaded, in matched order.
    """
    runs_dir = str(runs_dir)
    filter_by = filter_by or []

    run_folders = []
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name)
        if not os.path.isdir(path):
            continue
        if any(fb not in name for fb in filter_by):
            continue
        run_folders.append(path)
        if num_runs is not None and len(run_folders) >= num_runs:
            break

    data = {vname: [] for vname in var_names}
    for run_folder in run_folders:
        event_files = [os.path.join(run_folder, n) for n in os.listdir(run_folder)
                        if not os.path.isdir(os.path.join(run_folder, n))]
        for event_file in event_files:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            available = ea.Tags()["scalars"]
            for vname in var_names:
                if vname not in available:
                    continue
                data[vname].append([e.value for e in ea.Scalars(vname)])

    return data, run_folders


# ── Phase 4 analysis ─────────────────────────────────────────────────────────

def final_third_mean(history):
    """Mean of the final third of a scalar's recorded history."""
    arr = np.asarray(history, dtype=float)
    q = max(1, len(arr) // 3)
    return float(arr[-q:].mean())


def phase4_trial_stats(trial_dir, scalar_names=PHASE4_SCALARS):
    """
    Summarize a single trial's Phase 4 (final testing) tensorboard runs.

    Phase 4 re-runs the trial's tuned hyperparameters on fresh data draws —
    one tensorboard run per seed (see autotuner.analysis()). For each scalar,
    this takes the final-third mean of each phase=4 run's history, then
    reports the mean/std of those per-run values across the trial's phase=4
    runs.

    Parameters
    ----------
    trial_dir : str or Path
        A trial directory (e.g. .../<experiment>/trial_0), containing a
        "runs" subdirectory.
    scalar_names : list[str]
        Scalar tags to summarize.

    Returns
    -------
    dict[str, dict]
        scalar_name -> {"mean": float, "std": float, "n_runs": int}
    """
    runs_dir = os.path.join(str(trial_dir), "runs")
    data, run_folders = load_tensorboard_runs(runs_dir, scalar_names, filter_by=["phase=4"])

    stats = {}
    for vname in scalar_names:
        per_run_means = [final_third_mean(h) for h in data.get(vname, []) if len(h) > 0]
        stats[vname] = {
            "mean": float(np.mean(per_run_means)) if per_run_means else float("nan"),
            "std": float(np.std(per_run_means)) if per_run_means else float("nan"),
            "n_runs": len(per_run_means),
        }
    return stats


def analyze_experiment_phase4(experiment_dir, scalar_names=PHASE4_SCALARS):
    """
    Run phase4_trial_stats for every trial_N directory in an experiment.

    Parameters
    ----------
    experiment_dir : str or Path
        e.g. hyperparam_tuning/autoTuneDir/<experiment_folder>

    Returns
    -------
    dict[str, dict]
        trial name (e.g. "trial_0") -> phase4_trial_stats(...) result.
    """
    experiment_dir = str(experiment_dir)
    trial_names = sorted(
        (n for n in os.listdir(experiment_dir)
         if n.startswith("trial_") and os.path.isdir(os.path.join(experiment_dir, n))),
        key=lambda n: int(n.split("_")[1]),
    )
    return {
        trial_name: phase4_trial_stats(os.path.join(experiment_dir, trial_name), scalar_names)
        for trial_name in trial_names
    }


# ── batch_summary.txt parsing ────────────────────────────────────────────────

_BATCH_SUMMARY_LINE_RE = re.compile(r"^trial \d+: (\{.*\})\s+dir=(\S+)$")


def load_batch_summary(experiment_dir, skip_zero=True):
    """
    Parse an experiment's batch_summary.txt into {trial_name: tuned_params}.

    Each line looks like:
        trial 0: {'lambdaJSD': 10.0, 'lambdad': 25.0, 'lambdaw': 0.0}  dir=trial_0

    Parameters
    ----------
    experiment_dir : str or Path
    skip_zero : bool
        Drop params whose tuned value is 0 (i.e. params that weren't
        actually pushed away from their inactive default).

    Returns
    -------
    dict[str, dict]
        trial name (e.g. "trial_0") -> {param_name: value}
    """
    summary_path = os.path.join(str(experiment_dir), "batch_summary.txt")
    trial_params = {}
    with open(summary_path) as f:
        for line in f:
            match = _BATCH_SUMMARY_LINE_RE.match(line.strip())
            if not match:
                continue
            params = ast.literal_eval(match.group(1))
            trial_name = match.group(2)
            if skip_zero:
                params = {k: v for k, v in params.items() if v != 0}
            trial_params[trial_name] = params
    return trial_params


if __name__ == "__main__":
    experiment_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if experiment_dir is None:
        auto_tune_dir = BASE_DIR / "hyperparam_tuning" / "autoTuneDir"
        experiments = sorted(p for p in auto_tune_dir.iterdir() if p.is_dir())
        experiment_dir = experiments[-1]
    results = analyze_experiment_phase4(experiment_dir)
    trial_params = load_batch_summary(experiment_dir)
    for trial_name, stats in results.items():
        params = ", ".join(f"{k}: {v}" for k, v in trial_params.get(trial_name, {}).items())
        print(f"{trial_name} - {params}")
        for vname, s in stats.items():
            print(f"  {vname}: mean={s['mean']:.4f} std={s['std']:.4f} (n={s['n_runs']})")
