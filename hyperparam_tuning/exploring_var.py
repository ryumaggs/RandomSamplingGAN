"""
hyperparam_tuning/exploring_batchsize.py

Analysis of how autotuner.py's tuning behavior scales with batch size, using
the experiment folders already collected under hyperparam_tuning/autoTuneDir/.
(Not autotuner_optuna.py — that's an outdated, unused script.)

Each autoTuneDir/<experiment>/ folder name carries a _<value><var> tag for the
swept variable (e.g. "..._64subset_size", "..._256batchsize_part1"; see
parse_run_name); within it, trial_N/ holds an
autotuneLogger.txt recording the autotune() run described in autotuner.py:

  Phase 1 sweeps lambdaJSD (holding lambdad=0) until the settled
  (final-quarter mean) jsd stabilizes, and picks a "best" lambdaJSD.
  Phase 2 then sweeps lambdad (holding lambdaJSD fixed at that best) until
  jsd degenerates, and picks a "best" lambdad.
  Phase 3 would sweep lambdaw, but it's intentionally inert (range is
  [0.0, 0.0, 0.2], so it only ever runs lambdaw=0.0) and isn't analyzed here.

This parses autotuneLogger.txt directly rather than the tensorboard event
files under trial_N/runs/ — the log already contains the computed
final-quarter mean jsd (and kliep) for every point in the Phase 1/2 sweeps,
so re-deriving it from tensorboard scalars isn't necessary for these
batch-size comparisons.
"""
import os
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
BASE_DIR = Path(__file__).parent.parent

import numpy as np
import matplotlib.pyplot as plt

from analysis import load_tensorboard_runs
from autotuner import final_quarter_mean

AUTOTUNE_DIR = Path(__file__).parent / "autoTuneDir"
VACCINATION_SCALAR = "RECVDVACC prediction"

# folder names are <YYYY-MM-DD_HH-MM-SS>_<dataset_stem>[_<value><var>...][_partN]
# (see autotuner.make_run_name). Each _<value><var> segment starts with a number
# and runs until the next "_<number>" or the end, so underscores inside var
# names (subset_size) and dataset stems (household_pulse) are safe.
_RUN_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.+?)(?:_part(\d+))?$")
_VAR_SEGMENT_RE = re.compile(r"_(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)([A-Za-z]\w*?)(?=_-?\d|$)")
_PHASE_HEADER_RE = re.compile(r"=== Phase (\d+): tuning (\w+) ===")
_RUNNING_RE = re.compile(r"Running: (.+)$")
_PARAM_RE = re.compile(r"(\w+)=([-\d.]+)")
_FQ_JSD_RE = re.compile(r"final-quarter mean jsd = ([-\d.]+)(?:, kliep = ([-\d.]+))?")
_BEST_RE = re.compile(r"Best (\w+) = ([-\d.]+)")


# ── autotuneLogger.txt parsing ──────────────────────────────────────────────

def parse_autotune_log(log_path):
    """
    Parse a trial's autotuneLogger.txt into per-phase sweep records.

    Parameters
    ----------
    log_path : str or Path

    Returns
    -------
    dict[int, dict]
        phase number -> {
            "records": [
                {param_name: value, ..., "final_quarter_jsd": float, "kliep": float or None},
                ...  # one per "Running: ..." / "final-quarter mean jsd = ..." pair, in run order
            ],
            "best": {param_name: value},  # from "Best <param> = <value>" lines
        }
    """
    phases = {}
    current_phase = None
    pending_params = None

    with open(log_path) as f:
        for line in f:
            header = _PHASE_HEADER_RE.search(line)
            if header:
                current_phase = int(header.group(1))
                phases.setdefault(current_phase, {"records": [], "best": {}})
                pending_params = None
                continue

            if current_phase is None:
                continue

            running = _RUNNING_RE.search(line)
            if running:
                pending_params = {k: float(v) for k, v in _PARAM_RE.findall(running.group(1))}
                continue

            fq = _FQ_JSD_RE.search(line)
            if fq and pending_params is not None:
                record = dict(pending_params)
                record["final_quarter_jsd"] = float(fq.group(1))
                record["kliep"] = float(fq.group(2)) if fq.group(2) is not None else None
                phases[current_phase]["records"].append(record)
                pending_params = None
                continue

            best = _BEST_RE.search(line)
            if best:
                phases[current_phase]["best"][best.group(1)] = float(best.group(2))

    return phases


# ── experiment/trial discovery ──────────────────────────────────────────────

def _parse_number(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def parse_run_name(name):
    """
    Parse an autoTuneDir folder name (autotuner.make_run_name format).

    e.g. "2026-10-08_09-38-27_household_pulse_32subset_size_part1" ->
         {"timestamp": "2026-10-08_09-38-27", "dataset": "household_pulse",
          "vars": {"subset_size": 32}, "part": 1}

    Returns None if the name doesn't match the format.
    """
    m = _RUN_NAME_RE.match(name)
    if not m:
        return None
    timestamp, rest, part = m.groups()
    first_var = re.search(r"_-?\d", rest)
    dataset = rest[:first_var.start()] if first_var else rest
    tail = rest[len(dataset):]
    segments = _VAR_SEGMENT_RE.findall(tail)
    # every character after the dataset stem must belong to a segment
    if "".join(f"_{v}{k}" for v, k in segments) != tail:
        return None
    return {
        "timestamp": timestamp,
        "dataset": dataset,
        "vars": {k: _parse_number(v) for v, k in segments},
        "part": int(part) if part else None,
    }


def discover_experiment_dirs(auto_tune_dir=None, var=None):
    """
    Group autoTuneDir experiment folders by the value of one swept variable
    encoded in their name. Folders for the same value that were split across
    multiple runs (e.g. "..._256subset_size_part1" and "..._256subset_size_part2")
    are grouped together under that one value.

    Parameters
    ----------
    auto_tune_dir : str or Path or None
        Defaults to hyperparam_tuning/autoTuneDir.
    var : str or None
        Variable to group by. If None, it's inferred, and every folder that
        carries a variable must carry the same single one.

    Returns
    -------
    dict[int or float, list[Path]]
        value -> sorted list of experiment directories for that value.
    """
    auto_tune_dir = Path(auto_tune_dir) if auto_tune_dir else AUTOTUNE_DIR
    parsed = []
    for entry in sorted(auto_tune_dir.iterdir()):
        info = parse_run_name(entry.name) if entry.is_dir() else None
        if info and info["vars"]:
            parsed.append((entry, info["vars"]))

    if var is None:
        var_sets = {tuple(sorted(vs)) for _, vs in parsed}
        if len(var_sets) > 1 or any(len(vs) != 1 for vs in var_sets):
            raise ValueError(f"can't infer a single swept variable from folder names "
                             f"(found {sorted(var_sets)}); pass var=")
        var = next(iter(var_sets))[0] if var_sets else None

    groups = {}
    for entry, vs in parsed:
        if var in vs:
            groups.setdefault(vs[var], []).append(entry)
    return groups


def iter_trial_log_paths(experiment_dir):
    """Yield (trial_name, autotuneLogger.txt path) for each trial_N under experiment_dir, in trial order."""
    experiment_dir = Path(experiment_dir)
    trial_dirs = sorted(
        (p for p in experiment_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")),
        key=lambda p: int(p.name.split("_")[1]),
    )
    for trial_dir in trial_dirs:
        log_path = trial_dir / "autotuneLogger.txt"
        if log_path.exists():
            yield trial_dir.name, log_path


def collect_batchsize_trials(auto_tune_dir=None, var=None):
    """
    Parse every trial's autotuneLogger.txt, grouped by batch size.

    Returns
    -------
    dict[int, list[dict]]
        batch_size -> list of per-trial records:
            {"experiment": str, "trial": str, "trial_dir": Path,
             "phases": parse_autotune_log(...) output}
    """
    groups = discover_experiment_dirs(auto_tune_dir, var=var)
    out = {}
    for batch_size, experiment_dirs in groups.items():
        trials = []
        for experiment_dir in experiment_dirs:
            for trial_name, log_path in iter_trial_log_paths(experiment_dir):
                trials.append({
                    "experiment": experiment_dir.name,
                    "trial": trial_name,
                    "trial_dir": log_path.parent,
                    "phases": parse_autotune_log(log_path),
                })
        out[batch_size] = trials
    return out


# ── plotting helpers ─────────────────────────────────────────────────────────

def _plot_param_vs_batchsize(summary, param_name, phase_label, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    batch_sizes = sorted(summary)
    means = [summary[bs]["mean"] for bs in batch_sizes]
    stds = [summary[bs]["std"] for bs in batch_sizes]

    x_pos = np.arange(len(batch_sizes))
    fig, ax = plt.subplots(figsize=(max(6, len(batch_sizes) * 1.2), 5))
    ax.bar(x_pos, means, yerr=stds, capsize=4, color="steelblue")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.set_xlabel("batch size")
    ax.set_ylabel(f"best {param_name}")
    ax.set_title(f"{phase_label}: best {param_name} by batch size")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{param_name}_by_batchsize.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def _plot_finalquarter_jsd_vs_lambdad(summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    batch_sizes = sorted(summary)
    colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(batch_sizes), 1)))
    for color, batch_size in zip(colors, batch_sizes):
        per_lambdad = summary[batch_size]
        lambdad_vals = sorted(per_lambdad)
        means = [per_lambdad[l]["mean"] for l in lambdad_vals]
        stds = [per_lambdad[l]["std"] for l in lambdad_vals]
        ax.errorbar(lambdad_vals, means, yerr=stds, marker="o", capsize=3,
                     label=f"batch size {batch_size}", color=color)

    ax.set_xlabel("lambdad")
    ax.set_ylabel("final-quarter mean jsd")
    ax.set_title("Phase 2: final-quarter mean jsd vs lambdad, by batch size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "phase2_finalquarter_jsd_vs_lambdad_by_batchsize.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def _plot_vaccination_vs_lambdad(summary, scalar_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    batch_sizes = sorted(summary)
    colors = plt.cm.viridis(np.linspace(0, 0.9, max(len(batch_sizes), 1)))
    for color, batch_size in zip(colors, batch_sizes):
        by_lambdad = summary[batch_size]["by_lambdad"]
        lambdad_vals = sorted(by_lambdad)
        means = [by_lambdad[l]["mean"] for l in lambdad_vals]
        stds = [by_lambdad[l]["std"] for l in lambdad_vals]
        ax.errorbar(lambdad_vals, means, yerr=stds, marker="o", capsize=3,
                     label=f"batch size {batch_size}", color=color)

        best_lambdad = summary[batch_size]["best_lambdad"]
        if best_lambdad in by_lambdad:
            ax.scatter([best_lambdad], [by_lambdad[best_lambdad]["mean"]],
                       marker="*", s=220, color=color, edgecolor="black",
                       linewidth=0.8, zorder=5)

    ax.set_xlabel("lambdad")
    ax.set_ylabel(scalar_name)
    ax.set_title(f"Phase 2: {scalar_name} vs lambdad, by batch size\n"
                 "(star = lambdad with lowest final-quarter mean jsd)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "phase2_vaccination_vs_lambdad_by_batchsize.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


# ── analysis functions ───────────────────────────────────────────────────────

def analyze_phase1_vs_batchsize(auto_tune_dir=None, plot=True, output_dir="./images/"):
    """
    Phase 1 sweeps lambdaJSD and settles on a "best" value once the
    final-quarter mean jsd stabilizes (autotuner.phase1_should_stop). This
    summarizes, per batch size, the best lambdaJSD each trial landed on.

    Parameters
    ----------
    auto_tune_dir : str or Path or None
    plot : bool
        If True, saves a bar chart of best lambdaJSD vs batch size.
    output_dir : str or Path

    Returns
    -------
    dict[int, dict]
        batch_size -> {"values": [best lambdaJSD per trial], "mean": float,
                        "std": float, "n_trials": int}
    """
    trials_by_bs = collect_batchsize_trials(auto_tune_dir)
    summary = {}
    for batch_size, trials in trials_by_bs.items():
        values = [
            t["phases"][1]["best"]["lambdaJSD"]
            for t in trials
            if 1 in t["phases"] and "lambdaJSD" in t["phases"][1]["best"]
        ]
        if not values:
            continue
        summary[batch_size] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n_trials": len(values),
        }

    for batch_size in sorted(summary):
        s = summary[batch_size]
        print(f"batch_size={batch_size}: best lambdaJSD mean={s['mean']:.2f} "
              f"std={s['std']:.2f} (n={s['n_trials']} trials) -> {s['values']}")

    if plot and summary:
        _plot_param_vs_batchsize(summary, param_name="lambdaJSD", phase_label="Phase 1",
                                  output_dir=output_dir)

    return summary


def analyze_phase2_vs_batchsize(auto_tune_dir=None, plot=True, output_dir="./images/"):
    """
    Phase 2 sweeps lambdad (lambdaJSD fixed at the Phase 1 best) and settles
    on a "best" value once jsd degenerates (autotuner.phase2_should_stop).
    This summarizes, per batch size, the best lambdad each trial landed on.

    Parameters
    ----------
    auto_tune_dir : str or Path or None
    plot : bool
        If True, saves a bar chart of best lambdad vs batch size.
    output_dir : str or Path

    Returns
    -------
    dict[int, dict]
        batch_size -> {"values": [best lambdad per trial], "mean": float,
                        "std": float, "n_trials": int}
    """
    trials_by_bs = collect_batchsize_trials(auto_tune_dir)
    summary = {}
    for batch_size, trials in trials_by_bs.items():
        values = [
            t["phases"][2]["best"]["lambdad"]
            for t in trials
            if 2 in t["phases"] and "lambdad" in t["phases"][2]["best"]
        ]
        if not values:
            continue
        summary[batch_size] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n_trials": len(values),
        }

    for batch_size in sorted(summary):
        s = summary[batch_size]
        print(f"batch_size={batch_size}: best lambdad mean={s['mean']:.2f} "
              f"std={s['std']:.2f} (n={s['n_trials']} trials) -> {s['values']}")

    if plot and summary:
        _plot_param_vs_batchsize(summary, param_name="lambdad", phase_label="Phase 2",
                                  output_dir=output_dir)

    return summary


def analyze_phase2_finalquarter_jsd_vs_batchsize(auto_tune_dir=None, plot=True, output_dir="./images/"):
    """
    Looks at the full Phase 2 lambdad sweep (every point, not just each
    trial's chosen best) and tracks how the final-quarter mean jsd moves as
    lambdad increases, grouped by batch size. This is the view for checking
    e.g. whether larger batch sizes drive final-quarter mean jsd down faster
    as lambdad increases.

    Parameters
    ----------
    auto_tune_dir : str or Path or None
    plot : bool
        If True, saves a line plot (one line per batch size, error bars
        across trials) of final-quarter mean jsd vs lambdad.
    output_dir : str or Path

    Returns
    -------
    dict[int, dict[float, dict]]
        batch_size -> {lambdad_value -> {"values": [...], "mean": float,
                                           "std": float, "n": int}}
    """
    trials_by_bs = collect_batchsize_trials(auto_tune_dir)
    summary = {}
    for batch_size, trials in trials_by_bs.items():
        per_lambdad = {}
        for t in trials:
            phase2 = t["phases"].get(2)
            if not phase2:
                continue
            for rec in phase2["records"]:
                per_lambdad.setdefault(rec["lambdad"], []).append(rec["final_quarter_jsd"])
        if not per_lambdad:
            continue
        summary[batch_size] = {
            lambdad: {
                "values": vals,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
            for lambdad, vals in sorted(per_lambdad.items())
        }

    for batch_size in sorted(summary):
        print(f"batch_size={batch_size}:")
        for lambdad, s in summary[batch_size].items():
            print(f"  lambdad={lambdad:.1f}: final-quarter jsd mean={s['mean']:.4f} "
                  f"std={s['std']:.4f} (n={s['n']})")

    if plot and summary:
        _plot_finalquarter_jsd_vs_lambdad(summary, output_dir=output_dir)

    return summary


def _format_hparam(value):
    """
    Format a swept hparam value the way run_experiment.run's run_comment
    embeds it in tensorboard run-folder names (plain float formatting, e.g.
    "25.0" — not the log's ":.4f" style).
    """
    return str(float(value))


def analyze_phase2_vaccination_by_lambdad(auto_tune_dir=None, plot=True, output_dir="./images/",
                                           scalar_name=VACCINATION_SCALAR):
    """
    For each batch size, find the lambdad value with the lowest Phase 2
    final-quarter mean jsd (per analyze_phase2_finalquarter_jsd_vs_batchsize)
    and compare its vaccination-prediction result against the other lambdad
    values swept in Phase 2.

    Phase 2 only runs each lambdad value once within a given trial, but
    every trial in a batch-size group repeats the same lambdad grid, so
    pooling across a batch size's trials (5-10 trials depending on the
    group) gives several repeats per lambdad value — the same pooling
    analyze_phase2_finalquarter_jsd_vs_batchsize uses for jsd.
    autotuneLogger.txt doesn't record vaccination prediction, so it's read
    straight from each Phase 2 run's tensorboard scalar (final-quarter mean
    of that run's history).

    Parameters
    ----------
    auto_tune_dir : str or Path or None
    plot : bool
        If True, saves a line plot (one line per batch size, error bars
        across trials) of vaccination prediction vs lambdad, with a star
        marking each batch size's lowest-jsd lambdad.
    output_dir : str or Path
    scalar_name : str
        Tensorboard scalar tag to read (default "RECVDVACC prediction").

    Returns
    -------
    dict[int, dict]
        batch_size -> {
            "best_lambdad": float,  # lambdad with the lowest mean final-quarter jsd
            "by_lambdad": {
                lambdad_value: {"values": [...], "mean": float, "std": float, "n": int}
            },
        }
    """
    jsd_summary = analyze_phase2_finalquarter_jsd_vs_batchsize(auto_tune_dir, plot=False)
    trials_by_bs = collect_batchsize_trials(auto_tune_dir)

    summary = {}
    for batch_size, per_lambdad_jsd in jsd_summary.items():
        best_lambdad = min(per_lambdad_jsd, key=lambda l: per_lambdad_jsd[l]["mean"])

        per_lambdad_vacc = {}
        for t in trials_by_bs.get(batch_size, []):
            phase2 = t["phases"].get(2)
            if not phase2:
                continue
            runs_dir = t["trial_dir"] / "runs"
            lambdad_values = sorted({rec["lambdad"] for rec in phase2["records"]})
            for lambdad in lambdad_values:
                data, _ = load_tensorboard_runs(
                    runs_dir, [scalar_name],
                    filter_by=["phase=2", f"||lambdad={_format_hparam(lambdad)}||"],
                )
                for hist in data.get(scalar_name, []):
                    if len(hist) == 0:
                        continue
                    per_lambdad_vacc.setdefault(lambdad, []).append(final_quarter_mean(hist))

        if not per_lambdad_vacc:
            continue
        by_lambdad = {
            lambdad: {
                "values": vals,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
            for lambdad, vals in sorted(per_lambdad_vacc.items())
        }
        summary[batch_size] = {"best_lambdad": best_lambdad, "by_lambdad": by_lambdad}

    for batch_size in sorted(summary):
        s = summary[batch_size]
        print(f"batch_size={batch_size}: lowest-jsd lambdad={s['best_lambdad']:.1f}")
        for lambdad, stats in s["by_lambdad"].items():
            marker = "  <- lowest jsd" if lambdad == s["best_lambdad"] else ""
            print(f"  lambdad={lambdad:.1f}: {scalar_name} mean={stats['mean']:.4f} "
                  f"std={stats['std']:.4f} (n={stats['n']}){marker}")

    if plot and summary:
        _plot_vaccination_vs_lambdad(summary, scalar_name=scalar_name, output_dir=output_dir)

    return summary


if __name__ == "__main__":
    auto_tune_dir = sys.argv[1] if len(sys.argv) > 1 else None

    print("=== Phase 1: best lambdaJSD by batch size ===")
    analyze_phase1_vs_batchsize(auto_tune_dir)

    print("\n=== Phase 2: best lambdad by batch size ===")
    analyze_phase2_vs_batchsize(auto_tune_dir)

    print("\n=== Phase 2: final-quarter mean jsd vs lambdad, by batch size ===")
    analyze_phase2_finalquarter_jsd_vs_batchsize(auto_tune_dir)

    print("\n=== Phase 2: vaccination prediction vs lambdad, by batch size ===")
    analyze_phase2_vaccination_by_lambdad(auto_tune_dir)
