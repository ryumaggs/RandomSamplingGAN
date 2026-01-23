import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Literal

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# =========================
# Config
# =========================

@dataclass
class SmoothingConfig:
    # EMA: higher alpha = less smoothing; lower alpha = more smoothing
    ema_alpha: float = 0.10


@dataclass
class PlateauSearchConfig:
    """
    Plateau is detected via a sliding window that must satisfy BOTH:
      - window_range <= range_pct * total_range
      - |window_slope| <= slope_pct * total_range

    All thresholds are relative to EACH INDIVIDUAL SERIES' (max - min).
    """
    window_frac: float = 0.18           # size of plateau window as fraction of series length
    min_window_points: int = 12         # ensure enough points even for short runs

    range_pct: float = 0.15             # plateau "flatness" via range (% of total range)
    slope_pct: float = 0.05             # plateau "flatness" via slope (% of total range)

    # Where plateau is allowed to start (as fraction of series length)
    # For Gen Entropy / Truth-Fake: "generally later 60%" means start >= 0.40.
    plateau_start_frac_default: float = 0.40

    # Allow small numerical noise by tolerating slight upward drift for "non-increasing" checks, etc.
    overall_change_tol_pct: float = 0.02


@dataclass
class TrendConfig:
    """
    "Meaningful" overall change thresholds, relative to series range.
    Keeps rules from being overly strict on small-range metrics.
    """
    overall_min_change_pct: float = 0.08     # require at least this fraction of range as net change for inc/dec


@dataclass
class BiasTrendConfig:
    """
    Bias score: only requirement is that the last `tail_frac` of the graph
    generally trends upward (ignoring when/where drop happened).
    Second point (if desired) is still "no significant plateau" in that tail.
    """
    tail_frac: float = 0.70

    # Require tail net rise of at least this fraction of total range
    tail_rise_min_pct: float = 0.08

    # Tail should be "generally" increasing: allow some down-steps
    tail_max_downstep_frac: float = 0.30

SecondRule = Literal["plateau", "no_plateau", "none"]
ExpectedType = Literal["dec_plateau", "inc_plateau", "tvd_noninc", "bias_drop_climb"]


@dataclass
class MetricSpec:
    # Metric name used in outputs. User requested "Bias score" exact capitalization.
    name: str

    # One or more possible TensorBoard tags (first match wins).
    tags: List[str]

    expected: ExpectedType

    # Each metric can earn up to two points:
    #   - 1 point for trend/shape expectation
    #   - 1 point for second rule: plateau desired / no-plateau desired / none
    second_rule: SecondRule = "plateau"

    # Plateau start constraint (fraction). If None, uses PlateauSearchConfig default.
    plateau_start_frac: Optional[float] = None


@dataclass
class MetricResult:
    points: int

    trend_ok: bool
    trend_reason: str

    second_ok: Optional[bool]         # None if second_rule == "none"
    second_reason: str

    plateau_start_step: Optional[int] # if plateau found
    used_tag: str


# =========================
# TensorBoard loading
# =========================

def _load_event_accumulator(file_path: str) -> EventAccumulator:
    acc = EventAccumulator(file_path)
    acc.Reload()
    return acc


def load_scalars_for_specs(
    file_path: str,
    specs: List[MetricSpec],
) -> Dict[str, Tuple[np.ndarray, np.ndarray, str]]:
    """
    Returns dict[metric_name] -> (steps, values, used_tag)
    Uses first matching tag from MetricSpec.tags
    """
    acc = _load_event_accumulator(file_path)
    available = set(acc.Tags().get("scalars", []))

    out: Dict[str, Tuple[np.ndarray, np.ndarray, str]] = {}

    for spec in specs:
        used_tag = None
        for t in spec.tags:
            if t in available:
                used_tag = t
                break
        if used_tag is None:
            raise ValueError(
                f"Missing TensorBoard tag for '{spec.name}'. Tried: {spec.tags}. "
                f"Available tags include: {sorted(list(available))[:30]}..."
            )

        events = acc.Scalars(used_tag)
        steps = np.array([e.step for e in events], dtype=np.int64)
        vals = np.array([e.value for e in events], dtype=np.float64)
        out[spec.name] = (steps, vals, used_tag)

    return out


# =========================
# Smoothing + helpers
# =========================

def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    if len(values) == 0:
        return values
    sm = np.empty_like(values, dtype=np.float64)
    sm[0] = values[0]
    for i in range(1, len(values)):
        sm[i] = alpha * values[i] + (1.0 - alpha) * sm[i - 1]
    return sm


def linear_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    m, _b = np.polyfit(x, y, 1)
    return float(m)


def series_range(y: np.ndarray) -> float:
    mn = float(np.min(y))
    mx = float(np.max(y))
    r = mx - mn
    return r if r != 0.0 else 1.0


def approx_total_change(y: np.ndarray) -> float:
    """
    Change from start to end (more direct than slope*n).
    """
    if len(y) < 2:
        return 0.0
    return float(y[-1] - y[0])


# =========================
# Plateau detection (sliding window)
# =========================

def find_plateau_window(
    steps: np.ndarray,
    y: np.ndarray,
    cfg: PlateauSearchConfig,
    start_frac: float,
) -> Tuple[bool, str, Optional[int]]:
    """
    Scans windows starting from index >= start_frac * n.
    Plateau condition is per-window:
      range(window) <= cfg.range_pct * total_range
      abs(slope(window)) <= cfg.slope_pct * total_range

    Returns (found, reason, plateau_start_step).
    """
    n = len(y)
    if n < 6:
        return False, "too_few_points", None

    r = series_range(y)
    w = max(cfg.min_window_points, int(cfg.window_frac * n))
    if w >= n:
        w = max(4, n // 2)

    start_i = int(np.floor(start_frac * n))
    start_i = max(start_i, 0)
    start_i = min(start_i, n - w)
    
    range_thr = cfg.range_pct * r
    slope_thr = cfg.slope_pct * r
    #print('series range: ', r, range_thr, slope_thr)
    # Scan windows; return earliest plateau in the allowed region
    for i in range(start_i, n - w + 1):
        seg = y[i:i + w]
        seg_range = float(np.max(seg) - np.min(seg))
        seg_slope = abs(linear_slope(seg))
        #print(i, i+w, seg_range, seg_slope)
        if seg_range <= range_thr and seg_slope <= slope_thr:
            return True, "plateau_found", int(steps[i])

    return False, "no_plateau_found", None


# =========================
# Trend rules
# =========================

def check_dec_then_plateau_trend(y: np.ndarray, tcfg: TrendConfig) -> Tuple[bool, str]:
    """
    Gen Entropy: generally decreasing.
    We require net change to be negative and "meaningful" relative to range.
    """
    r = series_range(y)
    net = approx_total_change(y)
    if net >= 0:
        return False, f"not_decreasing (net={net:.6g})"

    if abs(net) < tcfg.overall_min_change_pct * r:
        # Loosened behavior: allow small net decrease (still valid), but label it
        return True, f"weak_decrease_ok (net={net:.6g}, range={r:.6g})"

    return True, "decrease_ok"


def check_inc_then_plateau_trend(y: np.ndarray, tcfg: TrendConfig) -> Tuple[bool, str]:
    """
    Truth-Fake: generally increasing.
    """
    r = series_range(y)
    net = approx_total_change(y)
    if net <= 0:
        return False, f"not_increasing (net={net:.6g})"

    if abs(net) < tcfg.overall_min_change_pct * r:
        return True, f"weak_increase_ok (net={net:.6g}, range={r:.6g})"

    return True, "increase_ok"

def check_tvd_not_increasing(y: np.ndarray):
    """
    TVD rule:
    Fits a straight line over the entire series and
    passes if the slope is <= 0 (i.e., decreasing or flat).
    """
    if len(y) < 2:
        return True, "too_few_points_assumed_ok"
    slope = linear_slope(y[y.shape[0]//2:])

    if slope <= 0:
        return True, f"slope_non_positive_ok (slope={slope:.6g})"
    else:
        return False, f"slope_positive_fail (slope={slope:.6g})"


def check_bias_tail_upward(y: np.ndarray, tail_frac: float = 0.70):
    """
    Bias score rule (minimal):
    Returns True if the last `tail_frac` of the series
    has a positive overall trend.
    """
    n = len(y)
    if n < 5:
        return False, "too_few_points"

    tail_n = max(2, int(tail_frac * n))
    tail = y[-tail_n:]

    slope = linear_slope(tail)

    if slope > 0:
        return True, "tail_trend_upward"
    else:
        return False, f"tail_not_upward (slope={slope:.6g})"


def analyze_and_score(
    file_path: str,
    smoothing: SmoothingConfig,
    plateau_cfg: PlateauSearchConfig,
    trend_cfg: TrendConfig,
    bias_cfg: BiasTrendConfig,
    specs: Optional[List[MetricSpec]] = None,
    plot: bool = True,
) -> Dict[str, MetricResult]:
    """
    Returns per-metric results with 0..2 points each (except second_rule='none').
    """
    if specs is None:
        specs = [
            MetricSpec(
                name="Gen Entropy",
                tags=["Gen Entropy", "gen_entropy", "gen/entropy"],
                expected="dec_plateau",
                second_rule="plateau",
                plateau_start_frac=plateau_cfg.plateau_start_frac_default,  # generally later 60% => start>=0.40
            ),
            MetricSpec(
                name="TVD",
                tags=["tvd", "TVD", "eval/tvd"],
                expected="tvd_noninc",
                second_rule="plateau",  # can be plateau from start or later
                plateau_start_frac=0.0,
            ),
            MetricSpec(
                name="Truth-Fake",
                tags=["Truth - Fake scores", "Truth-Fake", "truth_fake", "truth-fake"],
                expected="inc_plateau",
                second_rule="plateau",
                plateau_start_frac=plateau_cfg.plateau_start_frac_default,
            ),
            MetricSpec(
                name="Bias score",  # exact capitalization requested
                tags=["Bias Score", "Bias score", "bias_score", "bias"],
                expected="bias_drop_climb",
                second_rule="no_plateau",  # second point awarded for NOT plateauing
                plateau_start_frac=None,
            ),
            MetricSpec(
                name="All gp grads",  # exact capitalization requested
                tags=["All gp grads"],
                expected="GP_noninc",
                second_rule="plateau",  # second point awarded for NOT plateauing
                plateau_start_frac=None,
            ),
        ]

    raw = load_scalars_for_specs(file_path, specs)
    results: Dict[str, MetricResult] = {}

    for spec in specs:
        steps, vals, used_tag = raw[spec.name]
        y = ema_smooth(vals, smoothing.ema_alpha)
        if plot:
            plt.figure()
            plt.plot(steps, y)
            plt.xlabel("Training Step")
            plt.ylabel(spec.name)
            plt.title(f"{spec.name} (EMA α={smoothing.ema_alpha})")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # ---- Trend point ----
        trend_ok = False
        trend_reason = "uninitialized"

        # ---- Second rule point ----
        second_ok: Optional[bool] = None
        second_reason: str = "uninitialized"
        plateau_step: Optional[int] = None

        if spec.expected == "dec_plateau":
            trend_ok, trend_reason = check_dec_then_plateau_trend(y, trend_cfg)

        elif spec.expected == "inc_plateau":
            trend_ok, trend_reason = check_inc_then_plateau_trend(y, trend_cfg)

        elif spec.expected == "tvd_noninc":
            trend_ok, trend_reason = check_tvd_not_increasing(y)

        elif spec.expected == "bias_drop_climb":
            trend_ok, trend_reason = check_bias_tail_upward(
                y=y, tail_frac=bias_cfg.tail_frac
            )
        elif spec.expected == 'GP_noninc':
            trend_ok, trend_reason = check_inc_then_plateau_trend(y, trend_cfg)
        else:
            assert 1 == 0, "unknown trend detected"
            trend_ok, trend_reason = False, f"unknown_expected({spec.expected})"

        points = 1 if trend_ok else 0

        # ---- Second rule scoring for non-bias metrics (plateau / none) ----
        if spec.expected != "bias_drop_climb":
            if spec.second_rule == "none":
                second_ok = None
                second_reason = "second_rule_none"
            else:
                start_frac = spec.plateau_start_frac
                if start_frac is None:
                    start_frac = plateau_cfg.plateau_start_frac_default

                found, reason, step0 = find_plateau_window(
                    steps=steps, y=y, cfg=plateau_cfg, start_frac=start_frac
                )
                plateau_step = step0

                if spec.second_rule == "plateau":
                    second_ok = found
                    second_reason = reason if found else reason
                elif spec.second_rule == "no_plateau":
                    second_ok = (not found)
                    second_reason = "no_plateau_ok" if (not found) else "plateau_found_but_not_allowed"
                else:
                    second_ok = None
                    second_reason = f"unknown_second_rule({spec.second_rule})"

        # Award second point if applicable and passed
        if second_ok is True:
            points += 1

        results[spec.name] = MetricResult(
            points=points,
            trend_ok=trend_ok,
            trend_reason=trend_reason,
            second_ok=second_ok,
            second_reason=second_reason,
            plateau_start_step=plateau_step,
            used_tag=used_tag,
        )

    return results


def summarize_results(results: Dict[str, MetricResult]) -> None:
    print("\n=== Metric Scores ===")
    total = 0
    max_total = 0

    for name, r in results.items():
        # If second_ok is None, max is 1; else max is 2
        metric_max = 1 if (r.second_ok is None) else 2
        max_total += metric_max
        total += r.points

        print(f"\n{name}  (tag='{r.used_tag}')")
        print(f"  Points: {r.points}/{metric_max}")
        print(f"  Trend:  {r.trend_ok}  ({r.trend_reason})")
        if r.second_ok is None:
            print(f"  Second: n/a  ({r.second_reason})")
        else:
            print(f"  Second: {r.second_ok}  ({r.second_reason})")
            if r.plateau_start_step is not None:
                print(f"  Plateau window start step: {r.plateau_start_step}")

    print(f"\nTOTAL: {total}/{max_total}")

def base_run(gp_search_space):
    '''
    check to make sure ALL_GPGRADS plateus
    '''
    cur_depth = 0
    cur_val = gp_search_space[0]

    #run experiment

    #get feedback on GP

    #if fail, repeat
    

def tune_model():
    '''
    Procedure:
    1. Run base model with GAN + GP, start at 10 and ensure that:
        a. ALL GP GRADS plateus over time
    2. Turn on Lambdad and do a binary search [1,20] starting at:
        a. start 10 - see if tvd is decreasing/decreasing+plateau at 10
        b. if yes - drop to 50% of window
        c. if no - increase by 50% of window
        Repeat until we get smallest lambdad where tvd is decreasing/decreasing+plateau
    3. Turn on Lambdaw: Search [.001, .005, .01, .05] in reverse order
        a. find x = max(list) such that gen-entropy decreases + plateau
        b. 
    '''
    max_search_depth = 4
    GP_search_space=[10,20]
    lambdad_search_space = [1,20]
    lambdaw_search_space = [0.001, 0.05]




# =========================
# Example usage
# =========================

if __name__ == "__main__":
    file_path = "path/to/events.out.tfevents..."

    smoothing_cfg = SmoothingConfig(ema_alpha=0.01)

    plateau_cfg = PlateauSearchConfig(
        window_frac=0.18,
        min_window_points=12,
        range_pct=0.18,                  # loosened: allow a bit more tail movement
        slope_pct=0.07,                  # loosened: allow small ongoing drift in plateau
        plateau_start_frac_default=0.40, # "generally later 60%" => plateau can start after 40%
        overall_change_tol_pct=0.02,     # TVD non-increasing tolerance
    )

    trend_cfg = TrendConfig(
        overall_min_change_pct=0.06      # loosened net-change requirement
    )

    bias_cfg = BiasTrendConfig(
        early_frac=0.20,
        late_frac=0.45,
        early_drop_min_pct=0.05,         # loosened
        late_rise_min_pct=0.08,          # loosened
        late_max_downstep_frac=0.25,     # loosened (allow more wiggle while climbing)
    )

    results = analyze_and_score(
        file_path=file_path,
        smoothing=smoothing_cfg,
        plateau_cfg=plateau_cfg,
        trend_cfg=trend_cfg,
        bias_cfg=bias_cfg,
        plot=True,
    )

    summarize_results(results)
