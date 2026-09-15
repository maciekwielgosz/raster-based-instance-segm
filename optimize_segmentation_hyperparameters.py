#!/usr/bin/env python3
"""Sequentially optimize CHM crown-segmentation hyperparameters against GT.

The optimizer uses a reproducible, site-stratified tune/validation/test split.
It starts with the current baseline and space-filling random candidates, then
fits a Random Forest surrogate and proposes candidates using an upper
confidence bound. The best tuning candidates are compared on validation data;
the winner is evaluated once on the untouched test split and, by default, on
all plots.

Every candidate is run in an isolated directory. Existing segmentation and GT
files are never replaced. A stopped study can be resumed by running the same
command with the same study directory and a larger ``--trials`` value.

Recommended full study, run from ``run_r``::

    python code/optimize_segmentation_hyperparameters.py --trials 40

Quick pipeline check::

    python code/optimize_segmentation_hyperparameters.py \
        --trials 1 --initial-trials 1 --validation-candidates 1 \
        --max-plots 4 --skip-full-evaluation --study-dir /tmp/segm-opt-smoke
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    from scipy.stats import qmc
    from sklearn.ensemble import RandomForestRegressor
except ImportError as error:
    raise SystemExit(
        "Missing optimization dependency. Use the project treescan environment "
        "or install numpy, scipy, and scikit-learn. "
        f"Original import error: {error}"
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent

# =============================================================================
# USER-TUNABLE OPTIMIZATION CONFIGURATION
# =============================================================================
DEFAULT_INPUT_DIR = RUN_DIR / "data_input_from_laz"
DEFAULT_GT_DIR = DEFAULT_INPUT_DIR
DEFAULT_STUDY_DIR = RUN_DIR / "data_output_from_laz" / "parameter_optimization"
SEGMENTATION_SCRIPT = SCRIPT_DIR / "pcopw_chunks_500m_Segmentacja.py"
EVALUATION_SCRIPT = SCRIPT_DIR / "evaluate_crown_segmentation.py"

GT_PREFIX = "topmost_gt_"
GT_LAYER = "crowns_gt_topmost"
CHM_PREFIX = "chm_"
PREDICTION_SUBDIRECTORY = "Segmentation3"

DEFAULT_TRIALS = 40
DEFAULT_INITIAL_TRIALS = 12
DEFAULT_VALIDATION_CANDIDATES = 5
DEFAULT_CANDIDATE_POOL_SIZE = 2048
DEFAULT_RANDOM_SEED = 20260915
DEFAULT_TUNE_FRACTION = 0.60
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_EXPLORATION_WEIGHT = 1.0

# The default objective emphasizes the standard IoU=0.50 operating point but
# also rewards candidates that remain useful under looser and stricter matching.
OBJECTIVE_WEIGHTS = {0.25: 0.20, 0.50: 0.60, 0.75: 0.20}
DEFAULT_OBJECTIVE = "weighted-pq"
PRIMARY_IOU = 0.50

# Search limits use metres where applicable. Candidates are constrained to a
# non-decreasing, mostly continuous LMF window function. The exact values passed
# to the segmenter (slopes and intercepts included) are recorded for every run.
LOW_HEIGHT_RANGE = (6.0, 14.0)
HIGH_HEIGHT_MAX = 32.0
MIN_BREAKPOINT_GAP = 6.0
LOW_WINDOW_RANGE = (1.0, 3.5)
MID_WINDOW_AT_HIGH_MAX = 6.0
HIGH_WINDOW_JUMP_RANGE = (0.0, 2.0)
HIGH_SLOPE_RANGE = (0.03, 0.25)
MIN_HEIGHT_RANGE = (1.0, 4.0)
MEDIAN_FILTER_CHOICES = (1, 3, 5)
WATERSHED_CONNECTIVITY_CHOICES = (4, 8)
FIXED_MIN_HEIGHT = 2.0
FIXED_POLYGON_CONNECTIVITY = 4
# =============================================================================
# END USER-TUNABLE OPTIMIZATION CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CandidateResult:
    phase: str
    candidate_id: str
    status: str
    score: float
    parameters: dict[str, float | int]
    metrics: dict[str, float]
    output_directory: str
    elapsed_seconds: float
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize raster crown-segmentation hyperparameters against GT."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--gt-prefix", default=GT_PREFIX)
    parser.add_argument("--gt-layer", default=GT_LAYER)
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Total tuning candidates, including the unchanged baseline.",
    )
    parser.add_argument(
        "--initial-trials",
        type=int,
        default=DEFAULT_INITIAL_TRIALS,
        help="Baseline plus space-filling candidates before surrogate proposals.",
    )
    parser.add_argument(
        "--validation-candidates",
        type=int,
        default=DEFAULT_VALIDATION_CANDIDATES,
        help="Top tuning candidates compared on the validation split.",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=DEFAULT_CANDIDATE_POOL_SIZE,
        help="Random candidates scored by the surrogate at each iteration.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--tune-fraction", type=float, default=DEFAULT_TUNE_FRACTION
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=0,
        help="Limit plots for a quick experiment; 0 uses every matched plot.",
    )
    parser.add_argument(
        "--objective",
        choices=("weighted-pq", "pq", "f1", "mean-iou", "area-iou"),
        default=DEFAULT_OBJECTIVE,
    )
    parser.add_argument(
        "--exploration-weight",
        type=float,
        default=DEFAULT_EXPLORATION_WEIGHT,
        help="Random-Forest UCB uncertainty weight; default 1.0.",
    )
    parser.add_argument(
        "--tune-min-height",
        action="store_true",
        help=(
            "Also search MIN_HEIGHT. Not recommended with topmost GT generated "
            "at a fixed 2 m threshold."
        ),
    )
    parser.add_argument(
        "--skip-full-evaluation",
        action="store_true",
        help="Do not rerun the selected winner on every selected plot.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and print the split without running segmentation.",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def canonical_parameters(parameters: dict[str, float | int]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for key, value in sorted(parameters.items()):
        if isinstance(value, (int, np.integer)):
            result[key] = int(value)
        else:
            result[key] = round(float(value), 8)
    return result


def candidate_id(parameters: dict[str, float | int]) -> str:
    encoded = json.dumps(canonical_parameters(parameters), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_parameters() -> dict[str, float | int]:
    return {
        "min_height": FIXED_MIN_HEIGHT,
        "median_filter_size": 3,
        "lmf_low_height_limit": 10.0,
        "lmf_high_height_limit": 25.0,
        "lmf_low_window_size": 2.0,
        "lmf_mid_slope": 0.1,
        "lmf_mid_intercept": 0.3,
        "lmf_high_slope": 0.15,
        "lmf_high_intercept": 1.0,
        "watershed_connectivity": 8,
        "polygon_connectivity": FIXED_POLYGON_CONNECTIVITY,
    }


def quantize(value: float, step: float) -> float:
    return round(round(value / step) * step, 8)


def decode_unit_candidate(
    unit: np.ndarray, tune_min_height: bool
) -> dict[str, float | int]:
    low_height = quantize(
        LOW_HEIGHT_RANGE[0]
        + float(unit[0]) * (LOW_HEIGHT_RANGE[1] - LOW_HEIGHT_RANGE[0]),
        0.25,
    )
    high_minimum = max(18.0, low_height + MIN_BREAKPOINT_GAP)
    high_height = quantize(
        high_minimum + float(unit[1]) * (HIGH_HEIGHT_MAX - high_minimum),
        0.25,
    )
    low_window = quantize(
        LOW_WINDOW_RANGE[0]
        + float(unit[2]) * (LOW_WINDOW_RANGE[1] - LOW_WINDOW_RANGE[0]),
        0.25,
    )
    mid_window_at_high = quantize(
        low_window + float(unit[3]) * (MID_WINDOW_AT_HIGH_MAX - low_window),
        0.25,
    )
    high_jump = quantize(
        HIGH_WINDOW_JUMP_RANGE[0]
        + float(unit[4])
        * (HIGH_WINDOW_JUMP_RANGE[1] - HIGH_WINDOW_JUMP_RANGE[0]),
        0.25,
    )
    high_slope = quantize(
        HIGH_SLOPE_RANGE[0]
        + float(unit[5]) * (HIGH_SLOPE_RANGE[1] - HIGH_SLOPE_RANGE[0]),
        0.01,
    )
    mid_slope = (mid_window_at_high - low_window) / (
        high_height - low_height
    )
    mid_intercept = low_window - mid_slope * low_height
    high_window_at_break = mid_window_at_high + high_jump
    high_intercept = high_window_at_break - high_slope * high_height
    median_index = min(
        int(float(unit[6]) * len(MEDIAN_FILTER_CHOICES)),
        len(MEDIAN_FILTER_CHOICES) - 1,
    )
    connectivity_index = min(
        int(float(unit[7]) * len(WATERSHED_CONNECTIVITY_CHOICES)),
        len(WATERSHED_CONNECTIVITY_CHOICES) - 1,
    )
    if tune_min_height:
        min_height = quantize(
            MIN_HEIGHT_RANGE[0]
            + float(unit[8]) * (MIN_HEIGHT_RANGE[1] - MIN_HEIGHT_RANGE[0]),
            0.25,
        )
    else:
        min_height = FIXED_MIN_HEIGHT
    return canonical_parameters(
        {
            "min_height": min_height,
            "median_filter_size": MEDIAN_FILTER_CHOICES[median_index],
            "lmf_low_height_limit": low_height,
            "lmf_high_height_limit": high_height,
            "lmf_low_window_size": low_window,
            "lmf_mid_slope": mid_slope,
            "lmf_mid_intercept": mid_intercept,
            "lmf_high_slope": high_slope,
            "lmf_high_intercept": high_intercept,
            "watershed_connectivity": WATERSHED_CONNECTIVITY_CHOICES[
                connectivity_index
            ],
            "polygon_connectivity": FIXED_POLYGON_CONNECTIVITY,
        }
    )


def feature_vector(parameters: dict[str, float | int]) -> list[float]:
    keys = (
        "min_height",
        "median_filter_size",
        "lmf_low_height_limit",
        "lmf_high_height_limit",
        "lmf_low_window_size",
        "lmf_mid_slope",
        "lmf_mid_intercept",
        "lmf_high_slope",
        "lmf_high_intercept",
        "watershed_connectivity",
    )
    return [float(parameters[key]) for key in keys]


def initial_candidates(count: int, seed: int, tune_min_height: bool) -> list[dict]:
    candidates = [baseline_parameters()]
    if count <= 1:
        return candidates
    dimensions = 9 if tune_min_height else 8
    sampler = qmc.LatinHypercube(d=dimensions, seed=seed)
    seen = {candidate_id(candidates[0])}
    for unit in sampler.random(n=max(count * 2, 16)):
        candidate = decode_unit_candidate(unit, tune_min_height)
        signature = candidate_id(candidate)
        if signature not in seen:
            candidates.append(candidate)
            seen.add(signature)
        if len(candidates) >= count:
            break
    return candidates


def propose_surrogate_candidate(
    completed: list[CandidateResult],
    attempted_ids: set[str],
    seed: int,
    pool_size: int,
    exploration_weight: float,
    tune_min_height: bool,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed + 104729 * len(attempted_ids))
    dimensions = 9 if tune_min_height else 8
    pool: list[dict[str, float | int]] = []
    pool_ids: set[str] = set()
    for unit in rng.random((pool_size * 2, dimensions)):
        candidate = decode_unit_candidate(unit, tune_min_height)
        signature = candidate_id(candidate)
        if signature not in attempted_ids and signature not in pool_ids:
            pool.append(candidate)
            pool_ids.add(signature)
        if len(pool) >= pool_size:
            break
    if not pool:
        raise RuntimeError("Could not generate a new hyperparameter candidate")
    if len(completed) < 5:
        return pool[0]

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=seed + len(completed),
        n_jobs=-1,
    )
    training_x = np.asarray(
        [feature_vector(result.parameters) for result in completed],
        dtype=np.float64,
    )
    training_y = np.asarray([result.score for result in completed])
    candidate_x = np.asarray([feature_vector(item) for item in pool])
    model.fit(training_x, training_y)
    tree_predictions = np.asarray(
        [tree.predict(candidate_x) for tree in model.estimators_]
    )
    predicted_mean = tree_predictions.mean(axis=0)
    predicted_std = tree_predictions.std(axis=0)
    acquisition = predicted_mean + exploration_weight * predicted_std
    return pool[int(np.argmax(acquisition))]


def plot_group(tile_id: str) -> str:
    parts = tile_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else tile_id


def discover_tiles(
    input_dir: Path,
    gt_dir: Path,
    gt_prefix: str,
    max_plots: int,
    seed: int,
) -> tuple[list[str], dict[str, Path]]:
    chm = {
        path.stem.removeprefix(CHM_PREFIX): path
        for path in input_dir.glob(f"{CHM_PREFIX}*.tif")
    }
    ground_truth = {
        path.stem.removeprefix(gt_prefix): path
        for path in gt_dir.glob(f"{gt_prefix}*.gpkg")
    }
    missing_gt = sorted(set(chm) - set(ground_truth))
    missing_chm = sorted(set(ground_truth) - set(chm))
    if missing_gt or missing_chm:
        raise ValueError(
            f"CHM/GT mismatch: missing GT={len(missing_gt)}, missing CHM={len(missing_chm)}"
        )
    tiles = sorted(chm)
    if not tiles:
        raise FileNotFoundError("No matched CHM/GT plots")
    if max_plots < 0:
        raise ValueError("--max-plots must be non-negative")
    if max_plots and max_plots < len(tiles):
        rng = np.random.default_rng(seed)
        groups: dict[str, list[str]] = defaultdict(list)
        for tile in tiles:
            groups[plot_group(tile)].append(tile)
        for values in groups.values():
            rng.shuffle(values)
        selected: list[str] = []
        group_names = sorted(groups)
        while len(selected) < max_plots:
            progressed = False
            for group in group_names:
                if groups[group] and len(selected) < max_plots:
                    selected.append(groups[group].pop())
                    progressed = True
            if not progressed:
                break
        tiles = sorted(selected)
    return tiles, ground_truth


def stratified_split(
    tiles: list[str], tune_fraction: float, validation_fraction: float, seed: int
) -> dict[str, list[str]]:
    if len(tiles) < 3:
        raise ValueError("At least three plots are required for tune/validation/test")
    groups: dict[str, list[str]] = defaultdict(list)
    for tile in tiles:
        groups[plot_group(tile)].append(tile)
    rng = np.random.default_rng(seed)
    split = {"tune": [], "validation": [], "test": []}
    for group in sorted(groups):
        values = sorted(groups[group])
        rng.shuffle(values)
        count = len(values)
        tune_count = max(1, int(round(tune_fraction * count)))
        validation_count = max(1, int(round(validation_fraction * count)))
        if tune_count + validation_count >= count and count >= 3:
            tune_count = max(1, count - validation_count - 1)
        validation_count = min(validation_count, count - tune_count)
        split["tune"].extend(values[:tune_count])
        split["validation"].extend(
            values[tune_count : tune_count + validation_count]
        )
        split["test"].extend(values[tune_count + validation_count :])

    # Small --max-plots smoke studies may contain one plot per site. Rebalance
    # globally while preserving disjointness; normal full studies do not need it.
    for empty_name in ("validation", "test"):
        if not split[empty_name]:
            donor = max(split, key=lambda name: len(split[name]))
            if len(split[donor]) <= 1:
                raise ValueError("Could not construct three non-empty data splits")
            split[empty_name].append(split[donor].pop())
    return {name: sorted(values) for name, values in split.items()}


def prepare_gt_links(
    directory: Path,
    tiles: list[str],
    gt_paths: dict[str, Path],
    gt_prefix: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for tile in tiles:
        source = gt_paths[tile].resolve()
        destination = directory / f"{gt_prefix}{tile}.gpkg"
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source:
                raise ValueError(f"Conflicting GT link: {destination}")
        else:
            destination.symlink_to(source)


def hyperparameter_cli(parameters: dict[str, float | int]) -> list[str]:
    result: list[str] = []
    for name in (
        "min_height",
        "median_filter_size",
        "lmf_low_height_limit",
        "lmf_high_height_limit",
        "lmf_low_window_size",
        "lmf_mid_slope",
        "lmf_mid_intercept",
        "lmf_high_slope",
        "lmf_high_intercept",
        "watershed_connectivity",
        "polygon_connectivity",
    ):
        result.extend((f"--{name.replace('_', '-')}", str(parameters[name])))
    return result


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND: " + " ".join(command) + "\n\n")
        stream.flush()
        process = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(
            f"Command exited with status {process.returncode}; see {log_path}"
        )


def read_metrics(summary_path: Path) -> dict[str, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    flattened: dict[str, float] = {}
    for row in payload["overall_metrics"]:
        threshold = float(row["threshold"])
        suffix = f"{threshold:.2f}"
        for field in (
            "precision",
            "recall",
            "f1",
            "segmentation_quality",
            "panoptic_quality",
            "mean_iou",
            "mean_dice",
            "area_iou",
        ):
            value = row.get(field)
            flattened[f"{field}@{suffix}"] = (
                float(value) if value is not None else math.nan
            )
    return flattened


def objective_score(metrics: dict[str, float], objective: str) -> float:
    def finite_or_zero(name: str) -> float:
        value = metrics[name]
        return value if math.isfinite(value) else 0.0

    if objective == "weighted-pq":
        return sum(
            weight * finite_or_zero(f"panoptic_quality@{threshold:.2f}")
            for threshold, weight in OBJECTIVE_WEIGHTS.items()
        )
    metric_name = {
        "pq": "panoptic_quality",
        "f1": "f1",
        "mean-iou": "mean_iou",
        "area-iou": "area_iou",
    }[objective]
    return finite_or_zero(f"{metric_name}@{PRIMARY_IOU:.2f}")


def result_from_manifest(path: Path) -> CandidateResult | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") not in {"complete", "failed"}:
        return None
    return CandidateResult(
        phase=payload["phase"],
        candidate_id=payload["candidate_id"],
        status=payload["status"],
        score=float(payload.get("score", -math.inf)),
        parameters=payload["parameters"],
        metrics={
            key: float(value) if value is not None else math.nan
            for key, value in payload.get("metrics", {}).items()
        },
        output_directory=payload["output_directory"],
        elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
        error=payload.get("error", ""),
    )


def evaluate_candidate(
    phase: str,
    trial_directory: Path,
    parameters: dict[str, float | int],
    tiles: list[str],
    gt_paths: dict[str, Path],
    args: argparse.Namespace,
) -> CandidateResult:
    manifest_path = trial_directory / "result.json"
    previous = result_from_manifest(manifest_path)
    if previous is not None and previous.status == "complete":
        print(f"Reusing {phase} candidate {previous.candidate_id}: {previous.score:.6f}")
        return previous

    trial_directory.mkdir(parents=True, exist_ok=True)
    gt_subset = trial_directory / "gt"
    segmentation_root = trial_directory / "segmentation"
    prediction_dir = segmentation_root / PREDICTION_SUBDIRECTORY
    metrics_dir = trial_directory / "metrics"
    prepare_gt_links(gt_subset, tiles, gt_paths, args.gt_prefix)
    signature = candidate_id(parameters)
    start = time.monotonic()
    segmentation_command = [
        sys.executable,
        str(SEGMENTATION_SCRIPT),
        "--input-dir",
        str(args.input_dir.resolve()),
        "--output-dir",
        str(segmentation_root),
        "--independent-tiles",
        "--overwrite",
    ]
    for tile in tiles:
        segmentation_command.extend(("--tile", tile))
    segmentation_command.extend(hyperparameter_cli(parameters))
    evaluation_command = [
        sys.executable,
        str(EVALUATION_SCRIPT),
        "--gt-dir",
        str(gt_subset),
        "--gt-prefix",
        args.gt_prefix,
        "--gt-layer",
        args.gt_layer,
        "--prediction-dir",
        str(prediction_dir),
        "--output-dir",
        str(metrics_dir),
        "--overwrite",
    ]
    status = "complete"
    error = ""
    score = -math.inf
    metrics: dict[str, float] = {}
    try:
        run_logged(segmentation_command, trial_directory / "segmentation.log")
        run_logged(evaluation_command, trial_directory / "evaluation.log")
        metrics = read_metrics(metrics_dir / "evaluation_summary.json")
        score = objective_score(metrics, args.objective)
        if not math.isfinite(score):
            raise ValueError("Objective score is not finite")
    except Exception as caught:
        status = "failed"
        error = f"{type(caught).__name__}: {caught}"
    elapsed = time.monotonic() - start
    payload = {
        "phase": phase,
        "candidate_id": signature,
        "status": status,
        "score": score if math.isfinite(score) else -1.0,
        "parameters": canonical_parameters(parameters),
        "metrics": metrics,
        "plots": tiles,
        "output_directory": str(trial_directory),
        "elapsed_seconds": elapsed,
        "error": error,
        "segmentation_command": segmentation_command,
        "evaluation_command": evaluation_command,
    }
    atomic_write_json(manifest_path, payload)
    result = result_from_manifest(manifest_path)
    assert result is not None
    if status == "complete":
        print(
            f"{phase} candidate {signature}: score={score:.6f}, "
            f"plots={len(tiles)}, time={elapsed / 60:.2f} min"
        )
    else:
        print(f"{phase} candidate {signature} FAILED: {error}", file=sys.stderr)
    return result


def write_leaderboard(path: Path, results: list[CandidateResult]) -> None:
    metric_columns = sorted(
        {key for result in results for key in result.metrics}
    )
    parameter_columns = list(baseline_parameters())
    fields = [
        "phase",
        "rank",
        "candidate_id",
        "status",
        "score",
        "elapsed_seconds",
        *parameter_columns,
        *metric_columns,
        "output_directory",
        "error",
    ]
    ordered = sorted(
        results,
        key=lambda result: (
            result.phase,
            -(result.score if result.status == "complete" else -math.inf),
        ),
    )
    ranks: dict[str, int] = defaultdict(int)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for result in ordered:
                ranks[result.phase] += 1
                writer.writerow(
                    {
                        "phase": result.phase,
                        "rank": ranks[result.phase],
                        "candidate_id": result.candidate_id,
                        "status": result.status,
                        "score": result.score,
                        "elapsed_seconds": result.elapsed_seconds,
                        **result.parameters,
                        **result.metrics,
                        "output_directory": result.output_directory,
                        "error": result.error,
                    }
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.trials < 1 or args.initial_trials < 1:
        raise ValueError("--trials and --initial-trials must be positive")
    if args.validation_candidates < 1:
        raise ValueError("--validation-candidates must be positive")
    if args.candidate_pool < 32:
        raise ValueError("--candidate-pool must be at least 32")
    if not 0 < args.tune_fraction < 1:
        raise ValueError("--tune-fraction must be between 0 and 1")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if args.tune_fraction + args.validation_fraction >= 1:
        raise ValueError("Tune and validation fractions must sum to less than 1")
    if not math.isfinite(args.exploration_weight) or args.exploration_weight < 0:
        raise ValueError("--exploration-weight must be finite and non-negative")
    for path, description in (
        (args.input_dir, "input directory"),
        (args.gt_dir, "GT directory"),
        (SEGMENTATION_SCRIPT, "segmentation script"),
        (EVALUATION_SCRIPT, "evaluation script"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")
    if not args.gt_prefix or not args.gt_layer:
        raise ValueError("--gt-prefix and --gt-layer must not be empty")


def study_configuration(
    args: argparse.Namespace, tiles: list[str], split: dict[str, list[str]]
) -> dict:
    return {
        "version": 2,
        "input_directory": str(args.input_dir.resolve()),
        "gt_directory": str(args.gt_dir.resolve()),
        "gt_prefix": args.gt_prefix,
        "gt_layer": args.gt_layer,
        "objective": args.objective,
        "objective_weights": {
            f"{threshold:.2f}": weight
            for threshold, weight in OBJECTIVE_WEIGHTS.items()
        },
        "primary_iou": PRIMARY_IOU,
        "seed": args.seed,
        "tune_fraction": args.tune_fraction,
        "validation_fraction": args.validation_fraction,
        "tune_min_height": args.tune_min_height,
        "tiles": tiles,
        "split": split,
        "code_sha256": {
            "optimizer": file_sha256(Path(__file__).resolve()),
            "segmentation": file_sha256(SEGMENTATION_SCRIPT),
            "evaluation": file_sha256(EVALUATION_SCRIPT),
        },
        "search_space": {
            "low_height_range": list(LOW_HEIGHT_RANGE),
            "high_height_max": HIGH_HEIGHT_MAX,
            "minimum_breakpoint_gap": MIN_BREAKPOINT_GAP,
            "low_window_range": list(LOW_WINDOW_RANGE),
            "mid_window_at_high_max": MID_WINDOW_AT_HIGH_MAX,
            "high_window_jump_range": list(HIGH_WINDOW_JUMP_RANGE),
            "high_slope_range": list(HIGH_SLOPE_RANGE),
            "min_height_range": (
                list(MIN_HEIGHT_RANGE) if args.tune_min_height else None
            ),
            "median_filter_choices": list(MEDIAN_FILTER_CHOICES),
            "watershed_connectivity_choices": list(
                WATERSHED_CONNECTIVITY_CHOICES
            ),
            "polygon_connectivity": FIXED_POLYGON_CONNECTIVITY,
        },
    }


def load_existing_tune_results(study_dir: Path) -> list[CandidateResult]:
    results: list[CandidateResult] = []
    for manifest in sorted((study_dir / "tune").glob("trial_*/result.json")):
        result = result_from_manifest(manifest)
        if result is not None:
            results.append(result)
    return results


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.gt_dir = args.gt_dir.resolve()
    args.study_dir = args.study_dir.resolve()
    try:
        validate_args(args)
        tiles, gt_paths = discover_tiles(
            args.input_dir,
            args.gt_dir,
            args.gt_prefix,
            args.max_plots,
            args.seed,
        )
        split = stratified_split(
            tiles, args.tune_fraction, args.validation_fraction, args.seed
        )
        configuration = study_configuration(args, tiles, split)
        configuration_path = args.study_dir / "study_configuration.json"
        if configuration_path.exists():
            existing = json.loads(configuration_path.read_text(encoding="utf-8"))
            if existing != configuration:
                raise ValueError(
                    "Study configuration differs from the existing study. "
                    "Use another --study-dir."
                )
        elif not args.dry_run:
            args.study_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(configuration_path, configuration)
    except Exception as error:
        print(f"Input validation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    print(f"Matched plots: {len(tiles)}")
    print(
        f"Split: tune={len(split['tune'])}, validation={len(split['validation'])}, "
        f"test={len(split['test'])}"
    )
    print(f"Objective: {args.objective}")
    if args.objective == "weighted-pq":
        print(f"PQ weights: {OBJECTIVE_WEIGHTS}")
    print(f"Tuning trials: {args.trials}; initial candidates: {args.initial_trials}")
    print(f"Study directory: {args.study_dir}")
    if args.tune_min_height:
        print(
            "WARNING: MIN_HEIGHT will be tuned although topmost GT was generated "
            "with a fixed 2 m canopy threshold."
        )
    else:
        print("MIN_HEIGHT fixed at 2 m to match topmost GT.")
    if args.dry_run:
        for name in ("tune", "validation", "test"):
            print(f"{name}: " + ", ".join(split[name]))
        return 0

    initial = initial_candidates(
        args.initial_trials, args.seed, args.tune_min_height
    )
    tune_results = load_existing_tune_results(args.study_dir)
    attempted_ids = {result.candidate_id for result in tune_results}
    while len(tune_results) < args.trials:
        index = len(tune_results)
        if index < len(initial):
            parameters = initial[index]
            if candidate_id(parameters) in attempted_ids:
                parameters = propose_surrogate_candidate(
                    [item for item in tune_results if item.status == "complete"],
                    attempted_ids,
                    args.seed,
                    args.candidate_pool,
                    args.exploration_weight,
                    args.tune_min_height,
                )
        else:
            parameters = propose_surrogate_candidate(
                [item for item in tune_results if item.status == "complete"],
                attempted_ids,
                args.seed,
                args.candidate_pool,
                args.exploration_weight,
                args.tune_min_height,
            )
        signature = candidate_id(parameters)
        directory = args.study_dir / "tune" / f"trial_{index:04d}_{signature}"
        print(f"\nTuning trial {index + 1}/{args.trials}: {signature}")
        result = evaluate_candidate(
            "tune", directory, parameters, split["tune"], gt_paths, args
        )
        tune_results.append(result)
        attempted_ids.add(signature)
        write_leaderboard(args.study_dir / "leaderboard.csv", tune_results)

    completed_tune = sorted(
        (result for result in tune_results if result.status == "complete"),
        key=lambda result: result.score,
        reverse=True,
    )
    if not completed_tune:
        print("No tuning candidate completed successfully.", file=sys.stderr)
        return 1

    validation_count = min(args.validation_candidates, len(completed_tune))
    validation_parameters = [
        result.parameters for result in completed_tune[:validation_count]
    ]
    baseline = baseline_parameters()
    baseline_id = candidate_id(baseline)
    if baseline_id not in {candidate_id(item) for item in validation_parameters}:
        validation_parameters.append(baseline)

    validation_results: list[CandidateResult] = []
    print(f"\nValidating {len(validation_parameters)} candidates...")
    for rank, parameters in enumerate(validation_parameters, start=1):
        signature = candidate_id(parameters)
        directory = (
            args.study_dir / "validation" / f"candidate_{rank:02d}_{signature}"
        )
        validation_results.append(
            evaluate_candidate(
                "validation",
                directory,
                parameters,
                split["validation"],
                gt_paths,
                args,
            )
        )
    completed_validation = [
        result for result in validation_results if result.status == "complete"
    ]
    if not completed_validation:
        print("No validation candidate completed successfully.", file=sys.stderr)
        return 1
    winner = max(completed_validation, key=lambda result: result.score)
    print(f"\nValidation winner: {winner.candidate_id}, score={winner.score:.6f}")

    test_directory = args.study_dir / "test" / f"winner_{winner.candidate_id}"
    test_result = evaluate_candidate(
        "test", test_directory, winner.parameters, split["test"], gt_paths, args
    )
    all_results = [*tune_results, *validation_results, test_result]
    full_result: CandidateResult | None = None
    if not args.skip_full_evaluation:
        full_directory = args.study_dir / "full" / f"winner_{winner.candidate_id}"
        full_result = evaluate_candidate(
            "full", full_directory, winner.parameters, tiles, gt_paths, args
        )
        all_results.append(full_result)

    tune_winner = next(
        (
            result
            for result in completed_tune
            if result.candidate_id == winner.candidate_id
        ),
        None,
    )
    best_payload = {
        "candidate_id": winner.candidate_id,
        "selection_rule": "highest validation objective among top tuning candidates",
        "objective": args.objective,
        "parameters": winner.parameters,
        "scores": {
            "tune": tune_winner.score if tune_winner else None,
            "validation": winner.score,
            "test": test_result.score if test_result.status == "complete" else None,
            "full": (
                full_result.score
                if full_result is not None and full_result.status == "complete"
                else None
            ),
        },
        "metrics": {
            "validation": winner.metrics,
            "test": test_result.metrics,
            "full": full_result.metrics if full_result is not None else None,
        },
        "recommended_segmentation_command": [
            sys.executable,
            str(SEGMENTATION_SCRIPT),
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            "/path/to/new/output",
            "--independent-tiles",
            *hyperparameter_cli(winner.parameters),
        ],
    }
    atomic_write_json(args.study_dir / "best_parameters.json", best_payload)
    write_leaderboard(args.study_dir / "leaderboard.csv", all_results)
    print("\nOptimization complete.")
    print(f"Best parameters: {args.study_dir / 'best_parameters.json'}")
    print(f"Leaderboard: {args.study_dir / 'leaderboard.csv'}")
    if test_result.status == "complete":
        print(f"Held-out test score: {test_result.score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
