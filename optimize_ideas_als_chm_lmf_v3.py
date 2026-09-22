#!/usr/bin/env python3
"""Optimize CHM preprocessing and a flexible LMF curve on IDEAS-ALS."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import qmc

import optimize_segmentation_hyperparameters as optimizer


DEFAULT_TRIALS = 80
DEFAULT_INITIAL_TRIALS = 20
DEFAULT_SEED = 20260923
OBJECTIVE = "source-balanced-pq"
PIT_FILL_CHOICES = (1, 3, 5)
GAUSSIAN_SIGMA_CHOICES = (0.0, 0.5, 0.75, 1.0)
MEDIAN_FILTER_CHOICES = (1, 3)
SAMPLED_HEIGHTS = np.asarray((2, 5, 10, 15, 20, 25, 30, 35, 40, 50), dtype=float)

COMMON_FIELDS = (
    "min_height",
    "chm_pit_fill_size",
    "chm_pit_depth",
    "chm_gaussian_sigma",
    "median_filter_size",
)
LEGACY_LMF_FIELDS = (
    "lmf_low_height_limit",
    "lmf_high_height_limit",
    "lmf_low_window_size",
    "lmf_mid_slope",
    "lmf_mid_intercept",
    "lmf_high_slope",
    "lmf_high_intercept",
)
FLEXIBLE_LMF_FIELDS = (
    "lmf_control_height_1",
    "lmf_control_height_2",
    "lmf_control_height_3",
    "lmf_control_window_0",
    "lmf_control_window_1",
    "lmf_control_window_2",
    "lmf_control_window_3",
)
TAIL_FIELDS = ("watershed_connectivity", "polygon_connectivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Source-balanced IDEAS-ALS optimization of CHM pit filling, "
            "Gaussian smoothing, and a monotonic four-point LMF curve."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--initial-trials", type=int, default=DEFAULT_INITIAL_TRIALS)
    parser.add_argument("--candidate-pool", type=int, default=2048)
    parser.add_argument("--exploration-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def quantize(value: float, step: float) -> float:
    return round(round(value / step) * step, 8)


def choice(unit: float, values: tuple) -> int | float:
    index = min(int(float(unit) * len(values)), len(values) - 1)
    return values[index]


def decode_unit_candidate(unit: np.ndarray) -> dict[str, float | int]:
    pit_size = int(choice(unit[0], PIT_FILL_CHOICES))
    pit_depth = (
        0.0 if pit_size == 1 else quantize(0.5 + float(unit[1]) * 3.5, 0.5)
    )
    gaussian_sigma = float(choice(unit[2], GAUSSIAN_SIGMA_CHOICES))
    median_size = int(choice(unit[3], MEDIAN_FILTER_CHOICES))

    height_1 = quantize(8.0 + float(unit[4]) * 6.0, 0.25)
    height_2_min = max(16.0, height_1 + 5.0)
    height_2 = quantize(
        height_2_min + float(unit[5]) * (30.0 - height_2_min), 0.25
    )
    height_3_min = max(28.0, height_2 + 5.0)
    height_3 = quantize(
        height_3_min + float(unit[6]) * (48.0 - height_3_min), 0.25
    )

    window_0 = quantize(1.5 + float(unit[7]) * 2.0, 0.25)
    window_1 = window_0 + quantize(float(unit[8]) * 1.25, 0.25)
    window_2 = window_1 + quantize(float(unit[9]) * 2.0, 0.25)
    window_3 = window_2 + quantize(float(unit[10]) * 2.5, 0.25)
    return optimizer.canonical_parameters(
        {
            "min_height": 2.0,
            "chm_pit_fill_size": pit_size,
            "chm_pit_depth": pit_depth,
            "chm_gaussian_sigma": gaussian_sigma,
            "median_filter_size": median_size,
            "lmf_control_height_1": height_1,
            "lmf_control_height_2": height_2,
            "lmf_control_height_3": height_3,
            "lmf_control_window_0": window_0,
            "lmf_control_window_1": window_1,
            "lmf_control_window_2": window_2,
            "lmf_control_window_3": window_3,
            "watershed_connectivity": 4,
            "polygon_connectivity": 4,
        }
    )


def with_preprocessing_defaults(
    parameters: dict[str, float | int],
) -> dict[str, float | int]:
    result = dict(parameters)
    result.setdefault("chm_pit_fill_size", 1)
    result.setdefault("chm_pit_depth", 0.0)
    result.setdefault("chm_gaussian_sigma", 0.0)
    return optimizer.canonical_parameters(result)


def lmf_windows(parameters: dict[str, float | int]) -> np.ndarray:
    if "lmf_control_height_1" in parameters:
        heights = np.asarray(
            (
                float(parameters["min_height"]),
                float(parameters["lmf_control_height_1"]),
                float(parameters["lmf_control_height_2"]),
                float(parameters["lmf_control_height_3"]),
            )
        )
        windows = np.asarray(
            tuple(float(parameters[f"lmf_control_window_{i}"]) for i in range(4))
        )
        return np.interp(SAMPLED_HEIGHTS, heights, windows)
    return np.where(
        SAMPLED_HEIGHTS < float(parameters["lmf_low_height_limit"]),
        float(parameters["lmf_low_window_size"]),
        np.where(
            SAMPLED_HEIGHTS < float(parameters["lmf_high_height_limit"]),
            SAMPLED_HEIGHTS * float(parameters["lmf_mid_slope"])
            + float(parameters["lmf_mid_intercept"]),
            SAMPLED_HEIGHTS * float(parameters["lmf_high_slope"])
            + float(parameters["lmf_high_intercept"]),
        ),
    )


def feature_vector(parameters: dict[str, float | int]) -> list[float]:
    values = with_preprocessing_defaults(parameters)
    return [
        float(values["chm_pit_fill_size"]),
        float(values["chm_pit_depth"]),
        float(values["chm_gaussian_sigma"]),
        float(values["median_filter_size"]),
        *lmf_windows(values).tolist(),
        float(values["watershed_connectivity"]),
    ]


def read_source_mapping(manifest: Path, tiles: list[str]) -> dict[str, str]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    mapping = {
        row["dataset_id"]: row["collection"]
        for row in rows
        if row.get("split", "").lower() == "dev"
    }
    missing = sorted(set(tiles) - set(mapping))
    extra = sorted(set(mapping) - set(tiles))
    if missing or extra:
        raise ValueError(
            f"Manifest/CHM mismatch: missing mappings={missing}, extra={extra}"
        )
    return {tile: mapping[tile] for tile in tiles}


def source_scores(
    result: optimizer.CandidateResult, sources: list[str]
) -> np.ndarray:
    return np.asarray(
        [
            result.metrics[f"source_pq@{source}@{optimizer.PRIMARY_IOU:.2f}"]
            for source in sources
        ],
        dtype=np.float64,
    )


def initial_candidates(
    count: int, seed: int, incumbent_path: Path
) -> list[dict[str, float | int]]:
    incumbent_payload = json.loads(incumbent_path.read_text(encoding="utf-8"))
    incumbent = with_preprocessing_defaults(incumbent_payload["parameters"])
    baseline = with_preprocessing_defaults(optimizer.baseline_parameters())
    candidates = [incumbent, baseline]
    sampler = qmc.LatinHypercube(d=11, seed=seed)
    seen = {optimizer.candidate_id(item) for item in candidates}
    for unit in sampler.random(n=max(count * 2, 32)):
        candidate = decode_unit_candidate(unit)
        signature = optimizer.candidate_id(candidate)
        if signature not in seen:
            candidates.append(candidate)
            seen.add(signature)
        if len(candidates) >= count:
            break
    return candidates


def propose_candidate(
    completed: list[optimizer.CandidateResult],
    attempted_ids: set[str],
    sources: list[str],
    seed: int,
    pool_size: int,
    exploration_weight: float,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed + 104729 * len(attempted_ids))
    pool: list[dict[str, float | int]] = []
    pool_ids: set[str] = set()
    for unit in rng.random((pool_size * 2, 11)):
        candidate = decode_unit_candidate(unit)
        signature = optimizer.candidate_id(candidate)
        if signature not in attempted_ids and signature not in pool_ids:
            pool.append(candidate)
            pool_ids.add(signature)
        if len(pool) >= pool_size:
            break
    if not pool:
        raise RuntimeError("Could not generate a new candidate")
    if len(completed) < 5:
        return pool[0]

    training_x = np.asarray(
        [feature_vector(result.parameters) for result in completed], dtype=np.float64
    )
    candidate_x = np.asarray(
        [feature_vector(parameters) for parameters in pool], dtype=np.float64
    )
    scores = np.asarray(
        [source_scores(result, sources) for result in completed], dtype=np.float64
    )
    fold_means: list[np.ndarray] = []
    fold_uncertainties: list[np.ndarray] = []
    for held_out_index in range(len(sources)):
        training_y = np.mean(np.delete(scores, held_out_index, axis=1), axis=1)
        model = optimizer.RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=seed + len(completed) * 101 + held_out_index,
            n_jobs=-1,
        )
        model.fit(training_x, training_y)
        tree_predictions = np.asarray(
            [tree.predict(candidate_x) for tree in model.estimators_]
        )
        fold_means.append(tree_predictions.mean(axis=0))
        fold_uncertainties.append(tree_predictions.std(axis=0))
    predicted = np.asarray(fold_means)
    uncertainty = np.asarray(fold_uncertainties)
    acquisition = (
        predicted.mean(axis=0)
        - 0.25 * predicted.std(axis=0)
        + exploration_weight * uncertainty.mean(axis=0)
    )
    return pool[int(np.argmax(acquisition))]


def loso_selection(
    completed: list[optimizer.CandidateResult], sources: list[str]
) -> list[dict]:
    rows: list[dict] = []
    for held_out_index, held_out in enumerate(sources):
        ranked = []
        for result in completed:
            values = source_scores(result, sources)
            ranked.append(
                (
                    float(np.mean(np.delete(values, held_out_index))),
                    float(values[held_out_index]),
                    result,
                )
            )
        training_score, held_out_score, winner = max(ranked, key=lambda item: item[0])
        rows.append(
            {
                "held_out_source": held_out,
                "selected_candidate_id": winner.candidate_id,
                "training_sources_mean_pq": training_score,
                "held_out_source_pq": held_out_score,
            }
        )
    return rows


def atomic_write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_leaderboard(
    path: Path,
    completed: list[optimizer.CandidateResult],
    sources: list[str],
) -> None:
    rows: list[dict] = []
    for result in completed:
        source_values = source_scores(result, sources)
        rows.append(
            {
                "candidate_id": result.candidate_id,
                "mean_source_pq": float(np.mean(source_values)),
                "std_source_pq": float(np.std(source_values)),
                "minimum_source_pq": float(np.min(source_values)),
                "maximum_source_pq": float(np.max(source_values)),
                "elapsed_seconds": result.elapsed_seconds,
                **{
                    f"pq_{source}": float(value)
                    for source, value in zip(sources, source_values, strict=True)
                },
                **result.parameters,
                "lmf_parameterization": (
                    "flexible" if "lmf_control_height_1" in result.parameters else "legacy"
                ),
                "output_directory": result.output_directory,
            }
        )
    rows.sort(key=lambda row: row["mean_source_pq"], reverse=True)
    ranked = [{"rank": index, **row} for index, row in enumerate(rows, start=1)]
    fields = [
        "rank",
        "candidate_id",
        "mean_source_pq",
        "std_source_pq",
        "minimum_source_pq",
        "maximum_source_pq",
        "elapsed_seconds",
        *[f"pq_{source}" for source in sources],
        *COMMON_FIELDS,
        *LEGACY_LMF_FIELDS,
        *FLEXIBLE_LMF_FIELDS,
        *TAIL_FIELDS,
        "lmf_parameterization",
        "output_directory",
    ]
    atomic_write_csv(path, ranked, fields)


def main() -> int:
    args = parse_args()
    for name in ("input_dir", "gt_dir", "manifest", "study_dir", "incumbent"):
        setattr(args, name, getattr(args, name).resolve())
    if args.trials < 5 or not 2 <= args.initial_trials <= args.trials:
        raise ValueError("Require trials >= 5 and 2 <= initial-trials <= trials")
    if args.candidate_pool < 32:
        raise ValueError("--candidate-pool must be at least 32")

    tiles, gt_paths = optimizer.discover_tiles(
        args.input_dir, args.gt_dir, optimizer.GT_PREFIX, 0, args.seed
    )
    source_by_tile = read_source_mapping(args.manifest, tiles)
    sources = sorted(set(source_by_tile.values()))
    source_counts = {
        source: sum(value == source for value in source_by_tile.values())
        for source in sources
    }
    configuration = {
        "version": 1,
        "experiment": "ideas-als-chm-flexible-lmf-v3",
        "input_directory": str(args.input_dir),
        "gt_directory": str(args.gt_dir),
        "manifest": str(args.manifest),
        "manifest_sha256": optimizer.file_sha256(args.manifest),
        "objective": OBJECTIVE,
        "primary_iou": optimizer.PRIMARY_IOU,
        "trials": args.trials,
        "initial_trials": args.initial_trials,
        "candidate_pool": args.candidate_pool,
        "exploration_weight": args.exploration_weight,
        "seed": args.seed,
        "tiles": tiles,
        "source_by_tile": source_by_tile,
        "source_counts": source_counts,
        "incumbent": str(args.incumbent),
        "search_space": {
            "pit_fill_choices": list(PIT_FILL_CHOICES),
            "pit_depth_range": [0.5, 4.0],
            "gaussian_sigma_choices": list(GAUSSIAN_SIGMA_CHOICES),
            "median_filter_choices": list(MEDIAN_FILTER_CHOICES),
            "lmf_parameterization": "monotonic-four-control-point",
        },
        "code_sha256": {
            "driver": optimizer.file_sha256(Path(__file__).resolve()),
            "optimizer": optimizer.file_sha256(Path(optimizer.__file__).resolve()),
            "segmentation": optimizer.file_sha256(optimizer.SEGMENTATION_SCRIPT),
            "evaluation": optimizer.file_sha256(optimizer.EVALUATION_SCRIPT),
        },
    }
    print(f"Plots: {len(tiles)}; sources: {len(sources)}", flush=True)
    for source in sources:
        print(f"  {source}: {source_counts[source]}", flush=True)
    print(
        f"Trials: {args.trials}; initial: {args.initial_trials}; "
        f"objective: equal-source mean PQ@{optimizer.PRIMARY_IOU:.2f}",
        flush=True,
    )
    if args.dry_run:
        return 0

    args.study_dir.mkdir(parents=True, exist_ok=True)
    configuration_path = args.study_dir / "study_configuration.json"
    if configuration_path.exists():
        existing = json.loads(configuration_path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise ValueError(
                "Study configuration differs from existing study; use a new directory"
            )
    else:
        optimizer.atomic_write_json(configuration_path, configuration)

    evaluation_args = SimpleNamespace(
        input_dir=args.input_dir,
        gt_dir=args.gt_dir,
        gt_prefix=optimizer.GT_PREFIX,
        gt_layer=optimizer.GT_LAYER,
        objective=OBJECTIVE,
        source_by_tile=source_by_tile,
    )
    initial = initial_candidates(args.initial_trials, args.seed, args.incumbent)
    completed = optimizer.load_existing_tune_results(args.study_dir)
    attempted_ids = {result.candidate_id for result in completed}
    while len(completed) < args.trials:
        index = len(completed)
        if index < len(initial) and optimizer.candidate_id(initial[index]) not in attempted_ids:
            parameters = initial[index]
        else:
            parameters = propose_candidate(
                [result for result in completed if result.status == "complete"],
                attempted_ids,
                sources,
                args.seed,
                args.candidate_pool,
                args.exploration_weight,
            )
        signature = optimizer.candidate_id(parameters)
        directory = args.study_dir / "tune" / f"trial_{index:04d}_{signature}"
        print(f"\nTrial {index + 1}/{args.trials}: {signature}", flush=True)
        result = optimizer.evaluate_candidate(
            "tune", directory, parameters, tiles, gt_paths, evaluation_args
        )
        completed.append(result)
        attempted_ids.add(signature)
        write_leaderboard(
            args.study_dir / "candidate_source_leaderboard.csv",
            [item for item in completed if item.status == "complete"],
            sources,
        )

    successful = [result for result in completed if result.status == "complete"]
    if not successful:
        raise RuntimeError("No candidate completed successfully")
    loso_rows = loso_selection(successful, sources)
    atomic_write_csv(
        args.study_dir / "leave_one_source_out_selection.csv",
        loso_rows,
        list(loso_rows[0]),
    )
    winner = max(successful, key=lambda result: result.score)
    winner_values = source_scores(winner, sources)
    payload = {
        "candidate_id": winner.candidate_id,
        "selection_rule": "highest equal-source mean PQ@0.50 after LOSO stability analysis",
        "objective": OBJECTIVE,
        "parameters": winner.parameters,
        "scores": {
            "equal_source_mean_pq": float(np.mean(winner_values)),
            "source_pq_std": float(np.std(winner_values)),
            "minimum_source_pq": float(np.min(winner_values)),
            "maximum_source_pq": float(np.max(winner_values)),
        },
        "source_scores": {
            source: float(value)
            for source, value in zip(sources, winner_values, strict=True)
        },
        "overall_metrics": winner.metrics,
        "leave_one_source_out": loso_rows,
        "loso_mean_held_out_pq": float(
            np.mean([row["held_out_source_pq"] for row in loso_rows])
        ),
        "output_directory": winner.output_directory,
        "ideas_official_test_evaluated": False,
        "for_instance_evaluated": False,
    }
    optimizer.atomic_write_json(args.study_dir / "best_parameters.json", payload)
    print(
        f"\nWinner {winner.candidate_id}: equal-source mean PQ="
        f"{np.mean(winner_values):.6f}; LOSO mean held-out PQ="
        f"{payload['loso_mean_held_out_pq']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
