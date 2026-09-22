#!/usr/bin/env python3
"""Source-balanced IDEAS-ALS optimization with leave-one-source-out checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import optimize_segmentation_hyperparameters as optimizer


DEFAULT_TRIALS = 64
DEFAULT_INITIAL_TRIALS = 16
DEFAULT_SEED = 20260922
SEARCH_PROFILE = "ideas-als-small-windows-v2"
OBJECTIVE = "source-balanced-pq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize IDEAS-ALS with equal source weights and report "
            "leave-one-source-out parameter-selection performance."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--initial-trials", type=int, default=DEFAULT_INITIAL_TRIALS)
    parser.add_argument("--candidate-pool", type=int, default=2048)
    parser.add_argument("--exploration-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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
        [result.metrics[f"source_pq@{source}@{optimizer.PRIMARY_IOU:.2f}"] for source in sources],
        dtype=np.float64,
    )


def initial_candidates(
    count: int, seed: int, incumbent: Path | None
) -> list[dict[str, float | int]]:
    candidates = optimizer.initial_candidates(count, seed, False)
    if incumbent is None:
        return candidates
    payload = json.loads(incumbent.read_text(encoding="utf-8"))
    incumbent_parameters = optimizer.canonical_parameters(payload["parameters"])
    ordered = [candidates[0], incumbent_parameters, *candidates[1:]]
    unique: list[dict[str, float | int]] = []
    seen: set[str] = set()
    for parameters in ordered:
        signature = optimizer.candidate_id(parameters)
        if signature not in seen:
            unique.append(parameters)
            seen.add(signature)
        if len(unique) >= count:
            break
    return unique


def propose_loso_candidate(
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
    for unit in rng.random((pool_size * 2, 8)):
        parameters = optimizer.decode_unit_candidate(unit, False)
        signature = optimizer.candidate_id(parameters)
        if signature not in attempted_ids and signature not in pool_ids:
            pool.append(parameters)
            pool_ids.add(signature)
        if len(pool) >= pool_size:
            break
    if not pool:
        raise RuntimeError("Could not generate a new candidate")
    if len(completed) < 5:
        return pool[0]

    training_x = np.asarray(
        [optimizer.feature_vector(result.parameters) for result in completed],
        dtype=np.float64,
    )
    candidate_x = np.asarray(
        [optimizer.feature_vector(parameters) for parameters in pool],
        dtype=np.float64,
    )
    score_matrix = np.asarray(
        [source_scores(result, sources) for result in completed],
        dtype=np.float64,
    )
    fold_means: list[np.ndarray] = []
    fold_uncertainties: list[np.ndarray] = []
    for held_out_index in range(len(sources)):
        training_y = np.mean(
            np.delete(score_matrix, held_out_index, axis=1), axis=1
        )
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


def write_candidate_leaderboard(
    path: Path,
    completed: list[optimizer.CandidateResult],
    sources: list[str],
) -> None:
    rows: list[dict] = []
    for result in completed:
        values = source_scores(result, sources)
        rows.append(
            {
                "candidate_id": result.candidate_id,
                "mean_source_pq": float(np.mean(values)),
                "std_source_pq": float(np.std(values)),
                "minimum_source_pq": float(np.min(values)),
                "maximum_source_pq": float(np.max(values)),
                "elapsed_seconds": result.elapsed_seconds,
                **{
                    f"pq_{source}": float(value)
                    for source, value in zip(sources, values, strict=True)
                },
                **result.parameters,
                "output_directory": result.output_directory,
            }
        )
    rows.sort(key=lambda row: row["mean_source_pq"], reverse=True)
    fields = [
        "rank",
        "candidate_id",
        "mean_source_pq",
        "std_source_pq",
        "minimum_source_pq",
        "maximum_source_pq",
        "elapsed_seconds",
        *[f"pq_{source}" for source in sources],
        *optimizer.baseline_parameters().keys(),
        "output_directory",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for rank, row in enumerate(rows, start=1):
                writer.writerow({"rank": rank, **row})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def loso_selection(
    completed: list[optimizer.CandidateResult], sources: list[str]
) -> list[dict]:
    rows: list[dict] = []
    for held_out_index, held_out in enumerate(sources):
        ranked: list[tuple[float, float, optimizer.CandidateResult]] = []
        for result in completed:
            values = source_scores(result, sources)
            training_score = float(
                np.mean(np.delete(values, held_out_index))
            )
            ranked.append((training_score, float(values[held_out_index]), result))
        training_score, held_out_score, winner = max(
            ranked, key=lambda item: item[0]
        )
        rows.append(
            {
                "held_out_source": held_out,
                "selected_candidate_id": winner.candidate_id,
                "training_sources_mean_pq": training_score,
                "held_out_source_pq": held_out_score,
            }
        )
    return rows


def write_loso(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.gt_dir = args.gt_dir.resolve()
    args.manifest = args.manifest.resolve()
    args.study_dir = args.study_dir.resolve()
    if args.incumbent is not None:
        args.incumbent = args.incumbent.resolve()
    if args.trials < 5 or not 1 <= args.initial_trials <= args.trials:
        raise ValueError("Require trials >= 5 and 1 <= initial-trials <= trials")
    if args.candidate_pool < 32:
        raise ValueError("--candidate-pool must be at least 32")
    optimizer.apply_search_profile(SEARCH_PROFILE)
    tiles, gt_paths = optimizer.discover_tiles(
        args.input_dir, args.gt_dir, optimizer.GT_PREFIX, 0, args.seed
    )
    source_by_tile = read_source_mapping(args.manifest, tiles)
    sources = sorted(set(source_by_tile.values()))
    if len(sources) < 3:
        raise ValueError("At least three sources are required for LOSO")
    source_counts = {
        source: sum(value == source for value in source_by_tile.values())
        for source in sources
    }
    configuration = {
        "version": 1,
        "input_directory": str(args.input_dir),
        "gt_directory": str(args.gt_dir),
        "manifest": str(args.manifest),
        "manifest_sha256": optimizer.file_sha256(args.manifest),
        "objective": OBJECTIVE,
        "primary_iou": optimizer.PRIMARY_IOU,
        "search_profile": SEARCH_PROFILE,
        "trials": args.trials,
        "initial_trials": args.initial_trials,
        "candidate_pool": args.candidate_pool,
        "exploration_weight": args.exploration_weight,
        "seed": args.seed,
        "tiles": tiles,
        "source_by_tile": source_by_tile,
        "source_counts": source_counts,
        "incumbent": str(args.incumbent) if args.incumbent else None,
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
            parameters = propose_loso_candidate(
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
        write_candidate_leaderboard(
            args.study_dir / "candidate_source_leaderboard.csv",
            [result for result in completed if result.status == "complete"],
            sources,
        )

    successful = [result for result in completed if result.status == "complete"]
    if not successful:
        raise RuntimeError("No candidate completed successfully")
    loso_rows = loso_selection(successful, sources)
    write_loso(args.study_dir / "leave_one_source_out_selection.csv", loso_rows)
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
