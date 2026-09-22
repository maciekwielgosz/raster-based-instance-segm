#!/usr/bin/env python3
"""Route CHM plots to structure-specific frozen segmentation parameters."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

import optimize_segmentation_hyperparameters as optimizer


SCRIPT_DIR = Path(__file__).resolve().parent
SEGMENTER = SCRIPT_DIR / "pcopw_chunks_500m_Segmentacja.py"
EVALUATOR = SCRIPT_DIR / "evaluate_crown_segmentation.py"
CLASS_NAMES = ("open_low", "tall_complex", "closed_medium")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a CHM-only structural router and evaluate three frozen parameter sets."
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--parameter-file",
        type=Path,
        action="append",
        required=True,
        help="Frozen JSON for classes ordered by canopy cover; repeat exactly three times.",
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def chm_features(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        raster = source.read(1, masked=True).filled(np.nan).astype(np.float64)
    finite = raster[np.isfinite(raster)]
    if finite.size == 0:
        raise ValueError(f"CHM contains no finite cells: {path}")
    canopy = finite[finite >= 2.0]
    if canopy.size == 0:
        return np.zeros(4, dtype=np.float64)
    return np.asarray(
        (
            canopy.size / finite.size,
            *np.percentile(canopy, (50, 95)),
            np.std(canopy),
        ),
        dtype=np.float64,
    )


def discover(directory: Path) -> tuple[list[str], list[Path], np.ndarray]:
    paths = sorted(directory.glob("chm_*.tif"))
    if not paths:
        raise FileNotFoundError(f"No chm_*.tif files in {directory}")
    identifiers = [path.stem.removeprefix("chm_") for path in paths]
    features = np.asarray([chm_features(path) for path in paths])
    return identifiers, paths, features


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    args.reference_dir = args.reference_dir.resolve()
    args.input_dir = args.input_dir.resolve()
    args.gt_dir = args.gt_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.parameter_file = [path.resolve() for path in args.parameter_file]
    if len(args.parameter_file) != 3:
        raise ValueError("Exactly three --parameter-file arguments are required")

    reference_ids, reference_paths, reference_features = discover(args.reference_dir)
    target_ids, target_paths, target_features = discover(args.input_dir)
    scaler = RobustScaler().fit(reference_features)
    model = KMeans(n_clusters=3, random_state=args.seed, n_init=50).fit(
        scaler.transform(reference_features)
    )
    mean_cover = np.asarray(
        [reference_features[model.labels_ == index, 0].mean() for index in range(3)]
    )
    ordered_raw_classes = np.argsort(mean_cover)
    raw_to_ordered = {
        int(raw): ordered for ordered, raw in enumerate(ordered_raw_classes)
    }
    reference_classes = np.asarray(
        [raw_to_ordered[int(raw)] for raw in model.labels_], dtype=int
    )
    target_raw = model.predict(scaler.transform(target_features))
    target_classes = np.asarray(
        [raw_to_ordered[int(raw)] for raw in target_raw], dtype=int
    )

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.parameter_file]
    parameters = [optimizer.canonical_parameters(payload["parameters"]) for payload in payloads]
    assignments = [
        {
            "tile_id": tile,
            "class_index": int(class_index),
            "class_name": CLASS_NAMES[int(class_index)],
            "canopy_cover": float(values[0]),
            "height_p50": float(values[1]),
            "height_p95": float(values[2]),
            "height_sd": float(values[3]),
        }
        for tile, class_index, values in zip(
            target_ids, target_classes, target_features, strict=True
        )
    ]
    configuration = {
        "version": 1,
        "method": "three-class CHM-only KMeans structural router",
        "feature_order": ["canopy_cover", "height_p50", "height_p95", "height_sd"],
        "seed": args.seed,
        "class_names": list(CLASS_NAMES),
        "reference_directory": str(args.reference_dir),
        "reference_tiles": reference_ids,
        "reference_class_counts": [
            int(np.sum(reference_classes == index)) for index in range(3)
        ],
        "input_directory": str(args.input_dir),
        "gt_directory": str(args.gt_dir),
        "parameter_files": [str(path) for path in args.parameter_file],
        "parameter_file_sha256": [optimizer.file_sha256(path) for path in args.parameter_file],
        "candidate_ids": [payload.get("candidate_id") for payload in payloads],
        "parameters": parameters,
        "assignments": assignments,
    }
    configuration_path = args.output_dir / "routing_configuration.json"
    summary = args.output_dir / "quality_metrics_topmost" / "evaluation_summary.json"
    if configuration_path.exists():
        existing = json.loads(configuration_path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise ValueError(f"Existing routing configuration differs: {configuration_path}")
        if summary.is_file() and not args.overwrite:
            print(f"Reusing complete structural evaluation: {summary}")
            return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.atomic_write_json(configuration_path, configuration)
    assignment_path = args.output_dir / "structural_class_assignments.csv"
    with assignment_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(assignments[0]))
        writer.writeheader()
        writer.writerows(assignments)

    for class_index in range(3):
        tiles = [
            tile for tile, assigned in zip(target_ids, target_classes, strict=True)
            if assigned == class_index
        ]
        if not tiles:
            continue
        command = [
            sys.executable,
            str(SEGMENTER),
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(args.output_dir),
            "--independent-tiles",
        ]
        for tile in tiles:
            command.extend(("--tile", tile))
        command.extend(optimizer.hyperparameter_cli(parameters[class_index]))
        if args.overwrite:
            command.append("--overwrite")
        run(command)

    evaluation = [
        sys.executable,
        str(EVALUATOR),
        "--gt-dir",
        str(args.gt_dir),
        "--gt-prefix",
        "topmost_gt_",
        "--gt-layer",
        "crowns_gt_topmost",
        "--prediction-dir",
        str(args.output_dir / optimizer.PREDICTION_SUBDIRECTORY),
        "--output-dir",
        str(args.output_dir / "quality_metrics_topmost"),
    ]
    if args.overwrite:
        evaluation.append("--overwrite")
    run(evaluation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
