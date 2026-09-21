#!/usr/bin/env python3
"""Cross-validate pooled crown-segmentation candidates on development plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import optimize_segmentation_hyperparameters as optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a stable candidate across disjoint development folds."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, action="append", required=True)
    parser.add_argument("--incumbent", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-per-study", type=int, default=5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def collection_name(tile_id: str) -> str:
    return tile_id.split("_", 1)[0]


def stratified_folds(tiles: list[str], count: int, seed: int) -> list[list[str]]:
    if count < 2 or count > len(tiles):
        raise ValueError("--folds must be between 2 and the number of plots")
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    for tile in tiles:
        grouped[collection_name(tile)].append(tile)
    folds: list[list[str]] = [[] for _ in range(count)]
    for collection in sorted(grouped):
        values = sorted(grouped[collection])
        rng.shuffle(values)
        start = min(range(count), key=lambda index: len(folds[index]))
        for offset, tile in enumerate(values):
            folds[(start + offset) % count].append(tile)
    return [sorted(values) for values in folds]


def candidate_pool(
    study_dirs: list[Path], top_per_study: int, incumbent: Path | None
) -> dict[str, dict]:
    if top_per_study < 1:
        raise ValueError("--top-per-study must be positive")
    pooled: dict[str, dict] = {}
    for study_dir in study_dirs:
        completed = sorted(
            (
                result
                for result in optimizer.load_existing_tune_results(study_dir)
                if result.status == "complete"
            ),
            key=lambda result: result.score,
            reverse=True,
        )
        if not completed:
            raise ValueError(f"No completed tuning candidates in {study_dir}")
        for result in completed[:top_per_study]:
            pooled.setdefault(
                result.candidate_id,
                {
                    "parameters": result.parameters,
                    "sources": [],
                },
            )["sources"].append(str(study_dir))
    if incumbent is not None:
        payload = json.loads(incumbent.read_text(encoding="utf-8"))
        parameters = optimizer.canonical_parameters(payload["parameters"])
        signature = optimizer.candidate_id(parameters)
        pooled.setdefault(
            signature,
            {"parameters": parameters, "sources": []},
        )["sources"].append(f"incumbent:{incumbent}")
    return pooled


def write_leaderboard(path: Path, rows: list[dict]) -> None:
    parameter_columns = list(optimizer.baseline_parameters())
    fields = [
        "rank",
        "candidate_id",
        "mean_weighted_pq",
        "std_weighted_pq",
        "minimum_weighted_pq",
        "maximum_weighted_pq",
        *[f"fold_{index + 1}_weighted_pq" for index in range(len(rows[0]["scores"]))],
        *parameter_columns,
        "sources",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for rank, row in enumerate(rows, start=1):
                writer.writerow(
                    {
                        "rank": rank,
                        "candidate_id": row["candidate_id"],
                        "mean_weighted_pq": row["mean"],
                        "std_weighted_pq": row["std"],
                        "minimum_weighted_pq": row["minimum"],
                        "maximum_weighted_pq": row["maximum"],
                        **{
                            f"fold_{index + 1}_weighted_pq": score
                            for index, score in enumerate(row["scores"])
                        },
                        **row["parameters"],
                        "sources": " | ".join(row["sources"]),
                    }
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.gt_dir = args.gt_dir.resolve()
    args.study_dir = [path.resolve() for path in args.study_dir]
    args.output_dir = args.output_dir.resolve()
    if args.incumbent is not None:
        args.incumbent = args.incumbent.resolve()

    tiles, gt_paths = optimizer.discover_tiles(
        args.input_dir,
        args.gt_dir,
        optimizer.GT_PREFIX,
        0,
        args.seed,
    )
    folds = stratified_folds(tiles, args.folds, args.seed)
    pooled = candidate_pool(args.study_dir, args.top_per_study, args.incumbent)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_args = SimpleNamespace(
        input_dir=args.input_dir,
        gt_dir=args.gt_dir,
        gt_prefix=optimizer.GT_PREFIX,
        gt_layer=optimizer.GT_LAYER,
        objective=optimizer.DEFAULT_OBJECTIVE,
    )

    print(f"Development plots: {len(tiles)}")
    print(f"Candidate pool: {len(pooled)}")
    for index, fold in enumerate(folds, start=1):
        print(f"Fold {index} ({len(fold)} plots): " + ", ".join(fold))

    rows: list[dict] = []
    for candidate_index, (signature, item) in enumerate(sorted(pooled.items()), start=1):
        print(f"Candidate {candidate_index}/{len(pooled)}: {signature}", flush=True)
        scores: list[float] = []
        for fold_index, fold in enumerate(folds, start=1):
            directory = (
                args.output_dir
                / "fold_evaluations"
                / f"fold_{fold_index:02d}"
                / f"candidate_{signature}"
            )
            result = optimizer.evaluate_candidate(
                "cross_validation",
                directory,
                item["parameters"],
                fold,
                gt_paths,
                evaluation_args,
            )
            if result.status != "complete" or not math.isfinite(result.score):
                raise RuntimeError(
                    f"Candidate {signature} failed on fold {fold_index}: {result.error}"
                )
            scores.append(result.score)
        rows.append(
            {
                "candidate_id": signature,
                "parameters": item["parameters"],
                "sources": item["sources"],
                "scores": scores,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "minimum": float(np.min(scores)),
                "maximum": float(np.max(scores)),
            }
        )

    rows.sort(key=lambda row: (row["mean"], -row["std"]), reverse=True)
    write_leaderboard(args.output_dir / "cross_validation_leaderboard.csv", rows)
    winner = rows[0]
    full_result = optimizer.evaluate_candidate(
        "full_dev",
        args.output_dir / "full_dev" / f"winner_{winner['candidate_id']}",
        winner["parameters"],
        tiles,
        gt_paths,
        evaluation_args,
    )
    payload = {
        "candidate_id": winner["candidate_id"],
        "selection_rule": "highest mean weighted PQ across three disjoint dev folds",
        "parameters": winner["parameters"],
        "folds": folds,
        "fold_scores": winner["scores"],
        "mean_weighted_pq": winner["mean"],
        "std_weighted_pq": winner["std"],
        "minimum_weighted_pq": winner["minimum"],
        "maximum_weighted_pq": winner["maximum"],
        "full_dev_score": full_result.score,
        "full_dev_metrics": full_result.metrics,
        "sources": winner["sources"],
        "official_test_evaluated": False,
    }
    optimizer.atomic_write_json(args.output_dir / "best_refined_parameters.json", payload)
    print(
        f"Winner: {winner['candidate_id']}, mean={winner['mean']:.6f}, "
        f"std={winner['std']:.6f}, full_dev={full_result.score:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
