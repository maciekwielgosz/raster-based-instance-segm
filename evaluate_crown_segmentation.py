#!/usr/bin/env python3
"""Evaluate predicted tree-crown instances against LAZ-derived ground truth.

By default, the script pairs ``crowns_<plot>.gpkg`` predictions with
``gt_<plot>.gpkg`` references, performs one-to-one matching by polygon IoU,
and writes dataset, tile, match, and unmatched-object reports. Alternative GT
products can be selected with ``--gt-prefix`` and ``--gt-layer``.

Ground-truth crowns with ``evaluation_eligible=0`` are treated as ignore
regions by default. An unmatched prediction is not counted as a false positive
when at least half of its area lies in one of those incomplete edge crowns.

Run the complete TreeScan evaluation from ``run_r``:

    python code/evaluate_crown_segmentation.py

Evaluate one plot:

    python code/evaluate_crown_segmentation.py \
        --file Rem_Herby_2016_0702506
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

# Make direct interpreter invocation behave like an activated conda
# environment before GeoPandas/Rasterio load GDAL and PROJ.
_environment_prefix = Path(sys.prefix)
_proj_data = _environment_prefix / "share" / "proj"
_gdal_data = _environment_prefix / "share" / "gdal"
_gdal_plugins = _environment_prefix / "lib" / "gdalplugins"
if _proj_data.is_dir():
    os.environ.setdefault("PROJ_DATA", str(_proj_data))
if _gdal_data.is_dir():
    os.environ.setdefault("GDAL_DATA", str(_gdal_data))
if _gdal_plugins.is_dir():
    os.environ.setdefault("GDAL_DRIVER_PATH", str(_gdal_plugins))

try:
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
    from shapely.geometry.base import BaseGeometry
    from tqdm import tqdm
except ImportError as error:
    raise SystemExit(
        "Missing Python dependency. Use the project treescan environment or "
        "install geopandas, shapely, scipy, numpy, pandas, and tqdm. "
        f"Original import error: {error}"
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent

# =============================================================================
# USER-TUNABLE CONFIGURATION
# =============================================================================
DEFAULT_GT_DIR = RUN_DIR / "data_input_from_laz"
DEFAULT_PREDICTION_DIR = RUN_DIR / "data_output_from_laz" / "Segmentation3"
DEFAULT_OUTPUT_DIR = RUN_DIR / "data_output_from_laz" / "quality_metrics"

GT_PREFIX = "gt_"
PREDICTION_PREFIX = "crowns_"
GEOPACKAGE_SUFFIX = ".gpkg"
GT_LAYER_NAME = "crowns_gt"

DEFAULT_IOU_THRESHOLDS = (0.25, 0.50, 0.75)
DEFAULT_PRIMARY_IOU_THRESHOLD = 0.50
DEFAULT_IGNORE_OVERLAP_THRESHOLD = 0.50
DEFAULT_BOUNDARY_SAMPLE_SPACING_METRES = 0.25
DEFAULT_EXCLUDE_INCOMPLETE_GT = True
OVERWRITE_EXISTING_REPORTS = False

SUMMARY_FILENAME = "overall_metrics.csv"
PER_TILE_FILENAME = "per_tile_metrics.csv"
MATCHES_FILENAME = "matched_crowns.csv"
OBJECTS_FILENAME = "unmatched_objects.csv"
JSON_FILENAME = "evaluation_summary.json"
# =============================================================================
# END USER-TUNABLE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class PlotPair:
    tile_id: str
    gt_path: Path
    prediction_path: Path
    gt_layer: str


@dataclass(frozen=True)
class CountMetrics:
    threshold: float
    gt_count: int
    ignored_gt_count: int
    prediction_count: int
    ignored_prediction_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    recognition_quality: float
    segmentation_quality: float
    panoptic_quality: float
    mean_iou: float
    mean_dice: float
    matched_intersection_m2: float
    evaluated_gt_area_m2: float
    evaluated_prediction_area_m2: float
    area_precision: float
    area_recall: float
    area_dice: float
    area_iou: float


@dataclass(frozen=True)
class TileEvaluation:
    tile_id: str
    metrics_by_threshold: dict[float, CountMetrics]
    match_rows: list[dict]
    object_rows: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted crown instances against crown GT."
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=DEFAULT_GT_DIR,
        help="Directory containing ground-truth GeoPackages.",
    )
    parser.add_argument(
        "--gt-prefix",
        default=GT_PREFIX,
        help=f"Ground-truth filename prefix; default: {GT_PREFIX!r}.",
    )
    parser.add_argument(
        "--gt-layer",
        default=GT_LAYER_NAME,
        help=f"Ground-truth GeoPackage layer; default: {GT_LAYER_NAME!r}.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=DEFAULT_PREDICTION_DIR,
        help="Directory containing crowns_<plot>.gpkg files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and JSON evaluation reports.",
    )
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_IOU_THRESHOLDS),
        metavar="IOU",
        help="IoU thresholds for detection metrics; default: 0.25 0.50 0.75.",
    )
    parser.add_argument(
        "--primary-iou",
        type=float,
        default=DEFAULT_PRIMARY_IOU_THRESHOLD,
        help="IoU threshold used for per-tile, match, and object reports.",
    )
    parser.add_argument(
        "--ignore-overlap",
        type=float,
        default=DEFAULT_IGNORE_OVERLAP_THRESHOLD,
        help=(
            "Ignore an unmatched prediction when this fraction of its area "
            "overlaps incomplete GT; default 0.50."
        ),
    )
    parser.add_argument(
        "--boundary-spacing",
        type=float,
        default=DEFAULT_BOUNDARY_SAMPLE_SPACING_METRES,
        help="Boundary sampling interval in metres; default 0.25.",
    )
    parser.add_argument(
        "--include-incomplete-gt",
        action="store_true",
        help="Evaluate incomplete edge crowns instead of treating them as ignore regions.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="NAME",
        help="Evaluate only this plot/file name or stem. Repeatable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE_EXISTING_REPORTS,
        help="Replace existing evaluation reports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate file pairs and configuration without loading geometries.",
    )
    return parser.parse_args()


def normalize_requested_name(name: str, gt_prefix: str) -> str:
    normalized = Path(name).name
    if normalized.lower().endswith(GEOPACKAGE_SUFFIX):
        normalized = normalized[: -len(GEOPACKAGE_SUFFIX)]
    for prefix in (gt_prefix, PREDICTION_PREFIX):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def discover_pairs(
    gt_dir: Path,
    prediction_dir: Path,
    requested: list[str],
    gt_prefix: str,
    gt_layer: str,
) -> list[PlotPair]:
    ground_truth = {
        path.stem.removeprefix(gt_prefix): path
        for path in gt_dir.glob(f"{gt_prefix}*{GEOPACKAGE_SUFFIX}")
    }
    predictions = {
        path.stem.removeprefix(PREDICTION_PREFIX): path
        for path in prediction_dir.glob(
            f"{PREDICTION_PREFIX}*{GEOPACKAGE_SUFFIX}"
        )
    }
    if not ground_truth:
        raise FileNotFoundError(f"No {gt_prefix}*.gpkg files in {gt_dir}")
    if not predictions:
        raise FileNotFoundError(
            f"No {PREDICTION_PREFIX}*.gpkg files in {prediction_dir}"
        )

    missing_predictions = sorted(set(ground_truth) - set(predictions))
    missing_ground_truth = sorted(set(predictions) - set(ground_truth))
    if missing_predictions or missing_ground_truth:
        details: list[str] = []
        if missing_predictions:
            details.append(
                "missing predictions: " + ", ".join(missing_predictions)
            )
        if missing_ground_truth:
            details.append(
                "missing GT: " + ", ".join(missing_ground_truth)
            )
        raise ValueError("Unpaired evaluation files; " + "; ".join(details))

    available = {
        tile_id: PlotPair(
            tile_id,
            ground_truth[tile_id],
            predictions[tile_id],
            gt_layer,
        )
        for tile_id in ground_truth
    }
    if not requested:
        return [available[tile_id] for tile_id in sorted(available)]

    selected: list[PlotPair] = []
    missing: list[str] = []
    for name in requested:
        tile_id = normalize_requested_name(name, gt_prefix)
        pair = available.get(tile_id)
        if pair is None:
            missing.append(name)
        elif pair not in selected:
            selected.append(pair)
    if missing:
        raise FileNotFoundError(
            "Requested evaluation pairs not found: " + ", ".join(missing)
        )
    return selected


def validate_fraction(value: float, name: str, allow_zero: bool = True) -> None:
    lower_ok = value >= 0 if allow_zero else value > 0
    if not math.isfinite(value) or not lower_ok or value > 1:
        lower = "0" if allow_zero else "0 (exclusive)"
        raise ValueError(f"{name} must be between {lower} and 1")


def validate_args(args: argparse.Namespace) -> list[float]:
    if not args.gt_prefix:
        raise ValueError("--gt-prefix must not be empty")
    if not args.gt_layer:
        raise ValueError("--gt-layer must not be empty")
    validate_fraction(args.primary_iou, "--primary-iou", allow_zero=False)
    validate_fraction(args.ignore_overlap, "--ignore-overlap", allow_zero=False)
    if not math.isfinite(args.boundary_spacing) or args.boundary_spacing <= 0:
        raise ValueError("--boundary-spacing must be greater than zero")
    thresholds = set(args.iou_thresholds)
    thresholds.add(args.primary_iou)
    for threshold in thresholds:
        validate_fraction(threshold, "--iou-thresholds", allow_zero=False)
    return sorted(float(value) for value in thresholds)


def validate_geodataframe(
    frame: gpd.GeoDataFrame,
    path: Path,
    required_columns: set[str],
) -> None:
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path.name} lacks columns: {', '.join(sorted(missing))}"
        )
    if frame.crs is None:
        raise ValueError(f"{path.name} has no CRS")
    if frame["treeID"].isna().any() or frame["treeID"].duplicated().any():
        raise ValueError(f"{path.name} has missing or duplicate treeID values")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any():
        raise ValueError(f"{path.name} has missing or empty geometries")
    if not frame.geometry.is_valid.all():
        raise ValueError(f"{path.name} has invalid geometries")
    if not frame.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError(f"{path.name} contains non-polygon geometry")


def read_pair(pair: PlotPair) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    ground_truth = gpd.read_file(pair.gt_path, layer=pair.gt_layer)
    predictions = gpd.read_file(pair.prediction_path)
    validate_geodataframe(
        ground_truth,
        pair.gt_path,
        {"treeID", "evaluation_eligible", "geometry"},
    )
    validate_geodataframe(
        predictions,
        pair.prediction_path,
        {"treeID", "geometry"},
    )
    if ground_truth.crs != predictions.crs:
        raise ValueError(
            f"CRS mismatch: GT {ground_truth.crs}, prediction {predictions.crs}"
        )
    if ground_truth.crs.is_geographic:
        raise ValueError("Evaluation requires a projected CRS for metre-based metrics")
    return ground_truth, predictions


def pairwise_overlap(
    ground_truth: gpd.GeoDataFrame,
    predictions: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt_areas = ground_truth.geometry.area.to_numpy(dtype=np.float64)
    prediction_areas = predictions.geometry.area.to_numpy(dtype=np.float64)
    intersections = np.zeros(
        (len(ground_truth), len(predictions)), dtype=np.float64
    )
    for gt_index, gt_geometry in enumerate(ground_truth.geometry):
        for prediction_index, prediction_geometry in enumerate(
            predictions.geometry
        ):
            if gt_geometry.intersects(prediction_geometry):
                intersections[gt_index, prediction_index] = (
                    gt_geometry.intersection(prediction_geometry).area
                )
    unions = (
        gt_areas[:, np.newaxis]
        + prediction_areas[np.newaxis, :]
        - intersections
    )
    iou = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )
    return iou, intersections, gt_areas, prediction_areas


def ignored_prediction_overlap(
    ignored_ground_truth: gpd.GeoDataFrame,
    predictions: gpd.GeoDataFrame,
    prediction_areas: np.ndarray,
) -> np.ndarray:
    maximum = np.zeros(len(predictions), dtype=np.float64)
    for ignored_geometry in ignored_ground_truth.geometry:
        for prediction_index, prediction_geometry in enumerate(
            predictions.geometry
        ):
            if ignored_geometry.intersects(prediction_geometry):
                overlap = ignored_geometry.intersection(prediction_geometry).area
                if prediction_areas[prediction_index] > 0:
                    maximum[prediction_index] = max(
                        maximum[prediction_index],
                        overlap / prediction_areas[prediction_index],
                    )
    return maximum


def match_instances(
    iou: np.ndarray,
    threshold: float,
) -> list[tuple[int, int]]:
    if iou.shape[0] == 0 or iou.shape[1] == 0:
        return []
    # A +1 bonus makes the assignment maximize the number of threshold-valid
    # matches first, then their summed IoU. Invalid pairs contribute zero.
    score = np.where(iou >= threshold, 1.0 + iou, 0.0)
    gt_indices, prediction_indices = linear_sum_assignment(-score)
    return [
        (int(gt_index), int(prediction_index))
        for gt_index, prediction_index in zip(
            gt_indices, prediction_indices, strict=True
        )
        if iou[gt_index, prediction_index] >= threshold
    ]


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else math.nan


def calculate_count_metrics(
    threshold: float,
    matches: list[tuple[int, int]],
    iou: np.ndarray,
    intersections: np.ndarray,
    gt_areas: np.ndarray,
    prediction_areas: np.ndarray,
    ignored_gt_count: int,
    ignore_overlap: np.ndarray,
    ignore_threshold: float,
) -> tuple[CountMetrics, set[int], set[int], set[int]]:
    matched_gt = {gt_index for gt_index, _ in matches}
    matched_predictions = {prediction_index for _, prediction_index in matches}
    ignored_predictions = {
        prediction_index
        for prediction_index, overlap in enumerate(ignore_overlap)
        if prediction_index not in matched_predictions
        and overlap >= ignore_threshold
    }
    false_predictions = (
        set(range(len(prediction_areas)))
        - matched_predictions
        - ignored_predictions
    )
    missed_gt = set(range(len(gt_areas))) - matched_gt

    true_positives = len(matches)
    false_positives = len(false_predictions)
    false_negatives = len(missed_gt)
    precision = safe_ratio(true_positives, true_positives + false_positives)
    recall = safe_ratio(true_positives, true_positives + false_negatives)
    f1 = safe_ratio(
        2 * true_positives,
        2 * true_positives + false_positives + false_negatives,
    )
    matched_ious = [iou[gt_index, pred_index] for gt_index, pred_index in matches]
    matched_dice = [
        safe_ratio(
            2 * intersections[gt_index, pred_index],
            gt_areas[gt_index] + prediction_areas[pred_index],
        )
        for gt_index, pred_index in matches
    ]
    segmentation_quality = (
        float(np.mean(matched_ious)) if matched_ious else math.nan
    )
    recognition_quality = f1
    panoptic_quality = (
        segmentation_quality * recognition_quality
        if math.isfinite(segmentation_quality)
        and math.isfinite(recognition_quality)
        else math.nan
    )

    matched_intersection = float(
        sum(intersections[gt_index, pred_index] for gt_index, pred_index in matches)
    )
    evaluated_gt_area = float(gt_areas.sum())
    evaluated_prediction_area = float(
        sum(
            prediction_areas[index]
            for index in range(len(prediction_areas))
            if index not in ignored_predictions
        )
    )
    area_precision = safe_ratio(matched_intersection, evaluated_prediction_area)
    area_recall = safe_ratio(matched_intersection, evaluated_gt_area)
    area_dice = safe_ratio(
        2 * matched_intersection,
        evaluated_prediction_area + evaluated_gt_area,
    )
    area_iou = safe_ratio(
        matched_intersection,
        evaluated_prediction_area + evaluated_gt_area - matched_intersection,
    )

    metrics = CountMetrics(
        threshold=threshold,
        gt_count=len(gt_areas),
        ignored_gt_count=ignored_gt_count,
        prediction_count=len(prediction_areas),
        ignored_prediction_count=len(ignored_predictions),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        recognition_quality=recognition_quality,
        segmentation_quality=segmentation_quality,
        panoptic_quality=panoptic_quality,
        mean_iou=segmentation_quality,
        mean_dice=float(np.mean(matched_dice)) if matched_dice else math.nan,
        matched_intersection_m2=matched_intersection,
        evaluated_gt_area_m2=evaluated_gt_area,
        evaluated_prediction_area_m2=evaluated_prediction_area,
        area_precision=area_precision,
        area_recall=area_recall,
        area_dice=area_dice,
        area_iou=area_iou,
    )
    return metrics, missed_gt, false_predictions, ignored_predictions


def boundary_coordinates(
    geometry: BaseGeometry,
    spacing: float,
) -> np.ndarray:
    boundary = geometry.boundary
    lines = list(boundary.geoms) if hasattr(boundary, "geoms") else [boundary]
    coordinates: list[tuple[float, float]] = []
    for line in lines:
        if line.is_empty or line.length == 0:
            continue
        distances = np.arange(0.0, line.length, spacing)
        if distances.size == 0 or not math.isclose(distances[-1], line.length):
            distances = np.append(distances, line.length)
        coordinates.extend(
            (point.x, point.y) for point in (line.interpolate(d) for d in distances)
        )
    if not coordinates:
        raise ValueError("Cannot sample an empty crown boundary")
    return np.asarray(coordinates, dtype=np.float64)


def mean_symmetric_boundary_distance(
    first: BaseGeometry,
    second: BaseGeometry,
    spacing: float,
) -> tuple[float, float]:
    first_points = boundary_coordinates(first, spacing)
    second_points = boundary_coordinates(second, spacing)
    first_to_second = cKDTree(second_points).query(first_points, k=1)[0]
    second_to_first = cKDTree(first_points).query(second_points, k=1)[0]
    combined = np.concatenate((first_to_second, second_to_first))
    return float(combined.mean()), float(np.quantile(combined, 0.95))


def build_match_rows(
    pair: PlotPair,
    ground_truth: gpd.GeoDataFrame,
    predictions: gpd.GeoDataFrame,
    matches: list[tuple[int, int]],
    iou: np.ndarray,
    intersections: np.ndarray,
    gt_areas: np.ndarray,
    prediction_areas: np.ndarray,
    boundary_spacing: float,
) -> list[dict]:
    rows: list[dict] = []
    for gt_index, prediction_index in matches:
        gt_row = ground_truth.iloc[gt_index]
        prediction_row = predictions.iloc[prediction_index]
        gt_geometry = gt_row.geometry
        prediction_geometry = prediction_row.geometry
        mean_boundary, p95_boundary = mean_symmetric_boundary_distance(
            gt_geometry, prediction_geometry, boundary_spacing
        )
        union_area = (
            gt_areas[gt_index]
            + prediction_areas[prediction_index]
            - intersections[gt_index, prediction_index]
        )
        row = {
            "tile_id": pair.tile_id,
            "gt_treeID": int(gt_row["treeID"]),
            "prediction_treeID": int(prediction_row["treeID"]),
            "iou": float(iou[gt_index, prediction_index]),
            "dice": safe_ratio(
                2 * intersections[gt_index, prediction_index],
                gt_areas[gt_index] + prediction_areas[prediction_index],
            ),
            "intersection_m2": float(intersections[gt_index, prediction_index]),
            "union_m2": float(union_area),
            "gt_area_m2": float(gt_areas[gt_index]),
            "prediction_area_m2": float(prediction_areas[prediction_index]),
            "area_error_m2": float(
                prediction_areas[prediction_index] - gt_areas[gt_index]
            ),
            "absolute_area_error_m2": float(
                abs(prediction_areas[prediction_index] - gt_areas[gt_index])
            ),
            "centroid_distance_m": float(
                gt_geometry.centroid.distance(prediction_geometry.centroid)
            ),
            "hausdorff_distance_m": float(
                gt_geometry.hausdorff_distance(prediction_geometry)
            ),
            "mean_symmetric_boundary_distance_m": mean_boundary,
            "p95_symmetric_boundary_distance_m": p95_boundary,
        }
        for column in ("species_code", "species", "height_m"):
            if column in ground_truth.columns:
                row[column] = gt_row[column]
        rows.append(row)
    return rows


def build_object_rows(
    pair: PlotPair,
    ground_truth: gpd.GeoDataFrame,
    predictions: gpd.GeoDataFrame,
    missed_gt: set[int],
    false_predictions: set[int],
    ignored_predictions: set[int],
    gt_areas: np.ndarray,
    prediction_areas: np.ndarray,
    ignore_overlap: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    for gt_index in sorted(missed_gt):
        gt_row = ground_truth.iloc[gt_index]
        rows.append(
            {
                "tile_id": pair.tile_id,
                "status": "false_negative",
                "treeID": int(gt_row["treeID"]),
                "area_m2": float(gt_areas[gt_index]),
                "ignore_overlap_fraction": math.nan,
                "species_code": gt_row.get("species_code", pd.NA),
                "species": gt_row.get("species", pd.NA),
                "height_m": gt_row.get("height_m", math.nan),
            }
        )
    for prediction_index, status in [
        *((index, "false_positive") for index in sorted(false_predictions)),
        *((index, "ignored_prediction") for index in sorted(ignored_predictions)),
    ]:
        prediction_row = predictions.iloc[prediction_index]
        rows.append(
            {
                "tile_id": pair.tile_id,
                "status": status,
                "treeID": int(prediction_row["treeID"]),
                "area_m2": float(prediction_areas[prediction_index]),
                "ignore_overlap_fraction": float(ignore_overlap[prediction_index]),
                "species_code": pd.NA,
                "species": pd.NA,
                "height_m": math.nan,
            }
        )
    return rows


def evaluate_pair(
    pair: PlotPair,
    thresholds: list[float],
    primary_threshold: float,
    ignore_threshold: float,
    boundary_spacing: float,
    exclude_incomplete: bool,
) -> TileEvaluation:
    all_ground_truth, predictions = read_pair(pair)
    if exclude_incomplete:
        eligible_mask = all_ground_truth["evaluation_eligible"].astype(bool)
        ground_truth = all_ground_truth.loc[eligible_mask].reset_index(drop=True)
        ignored_ground_truth = all_ground_truth.loc[~eligible_mask].reset_index(
            drop=True
        )
    else:
        ground_truth = all_ground_truth.reset_index(drop=True)
        ignored_ground_truth = all_ground_truth.iloc[0:0].copy()
    predictions = predictions.reset_index(drop=True)

    iou, intersections, gt_areas, prediction_areas = pairwise_overlap(
        ground_truth, predictions
    )
    ignore_overlap = ignored_prediction_overlap(
        ignored_ground_truth, predictions, prediction_areas
    )

    metrics_by_threshold: dict[float, CountMetrics] = {}
    primary_state: tuple[
        list[tuple[int, int]], set[int], set[int], set[int]
    ] | None = None
    for threshold in thresholds:
        matches = match_instances(iou, threshold)
        metrics, missed_gt, false_predictions, ignored_predictions = (
            calculate_count_metrics(
                threshold,
                matches,
                iou,
                intersections,
                gt_areas,
                prediction_areas,
                len(ignored_ground_truth),
                ignore_overlap,
                ignore_threshold,
            )
        )
        metrics_by_threshold[threshold] = metrics
        if math.isclose(threshold, primary_threshold):
            primary_state = (
                matches,
                missed_gt,
                false_predictions,
                ignored_predictions,
            )

    if primary_state is None:
        raise RuntimeError("Primary IoU threshold was not evaluated")
    matches, missed_gt, false_predictions, ignored_predictions = primary_state
    match_rows = build_match_rows(
        pair,
        ground_truth,
        predictions,
        matches,
        iou,
        intersections,
        gt_areas,
        prediction_areas,
        boundary_spacing,
    )
    object_rows = build_object_rows(
        pair,
        ground_truth,
        predictions,
        missed_gt,
        false_predictions,
        ignored_predictions,
        gt_areas,
        prediction_areas,
        ignore_overlap,
    )
    return TileEvaluation(pair.tile_id, metrics_by_threshold, match_rows, object_rows)


def aggregate_metrics(
    evaluations: list[TileEvaluation],
    threshold: float,
) -> CountMetrics:
    tile_metrics = [item.metrics_by_threshold[threshold] for item in evaluations]
    gt_count = sum(item.gt_count for item in tile_metrics)
    ignored_gt_count = sum(item.ignored_gt_count for item in tile_metrics)
    prediction_count = sum(item.prediction_count for item in tile_metrics)
    ignored_prediction_count = sum(
        item.ignored_prediction_count for item in tile_metrics
    )
    true_positives = sum(item.true_positives for item in tile_metrics)
    false_positives = sum(item.false_positives for item in tile_metrics)
    false_negatives = sum(item.false_negatives for item in tile_metrics)
    matched_intersection = sum(
        item.matched_intersection_m2 for item in tile_metrics
    )
    gt_area = sum(item.evaluated_gt_area_m2 for item in tile_metrics)
    prediction_area = sum(
        item.evaluated_prediction_area_m2 for item in tile_metrics
    )
    weighted_iou_sum = sum(
        item.mean_iou * item.true_positives
        for item in tile_metrics
        if math.isfinite(item.mean_iou)
    )
    weighted_dice_sum = sum(
        item.mean_dice * item.true_positives
        for item in tile_metrics
        if math.isfinite(item.mean_dice)
    )
    mean_iou = safe_ratio(weighted_iou_sum, true_positives)
    mean_dice = safe_ratio(weighted_dice_sum, true_positives)
    precision = safe_ratio(true_positives, true_positives + false_positives)
    recall = safe_ratio(true_positives, true_positives + false_negatives)
    f1 = safe_ratio(
        2 * true_positives,
        2 * true_positives + false_positives + false_negatives,
    )
    return CountMetrics(
        threshold=threshold,
        gt_count=gt_count,
        ignored_gt_count=ignored_gt_count,
        prediction_count=prediction_count,
        ignored_prediction_count=ignored_prediction_count,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        recognition_quality=f1,
        segmentation_quality=mean_iou,
        panoptic_quality=(
            f1 * mean_iou
            if math.isfinite(f1) and math.isfinite(mean_iou)
            else math.nan
        ),
        mean_iou=mean_iou,
        mean_dice=mean_dice,
        matched_intersection_m2=matched_intersection,
        evaluated_gt_area_m2=gt_area,
        evaluated_prediction_area_m2=prediction_area,
        area_precision=safe_ratio(matched_intersection, prediction_area),
        area_recall=safe_ratio(matched_intersection, gt_area),
        area_dice=safe_ratio(2 * matched_intersection, prediction_area + gt_area),
        area_iou=safe_ratio(
            matched_intersection,
            prediction_area + gt_area - matched_intersection,
        ),
    )


def ensure_reports_can_be_written(output_dir: Path, overwrite: bool) -> None:
    paths = [
        output_dir / filename
        for filename in (
            SUMMARY_FILENAME,
            PER_TILE_FILENAME,
            MATCHES_FILENAME,
            OBJECTS_FILENAME,
            JSON_FILENAME,
        )
    ]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Evaluation reports already exist; use --overwrite: "
            + ", ".join(path.name for path in existing)
        )


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    start = time.monotonic()
    gt_dir = args.gt_dir.resolve()
    prediction_dir = args.prediction_dir.resolve()
    output_dir = args.output_dir.resolve()
    try:
        if not gt_dir.is_dir():
            raise FileNotFoundError(f"GT directory does not exist: {gt_dir}")
        if not prediction_dir.is_dir():
            raise FileNotFoundError(
                f"Prediction directory does not exist: {prediction_dir}"
            )
        thresholds = validate_args(args)
        pairs = discover_pairs(
            gt_dir,
            prediction_dir,
            args.file,
            args.gt_prefix,
            args.gt_layer,
        )
        ensure_reports_can_be_written(output_dir, args.overwrite)
    except Exception as error:
        print(
            f"Input validation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2

    exclude_incomplete = (
        DEFAULT_EXCLUDE_INCOMPLETE_GT and not args.include_incomplete_gt
    )
    print(f"Evaluation pairs: {len(pairs)}")
    print(f"GT directory: {gt_dir}")
    print(f"GT filename prefix: {args.gt_prefix}")
    print(f"GT GeoPackage layer: {args.gt_layer}")
    print(f"Prediction directory: {prediction_dir}")
    print(f"Output directory: {output_dir}")
    print("IoU thresholds: " + ", ".join(f"{value:g}" for value in thresholds))
    print(f"Primary IoU threshold: {args.primary_iou:g}")
    print(f"Exclude incomplete GT: {exclude_incomplete}")
    if args.dry_run:
        for pair in pairs:
            print(f"{pair.gt_path.name} + {pair.prediction_path.name}")
        return 0

    evaluations: list[TileEvaluation] = []
    errors: list[str] = []
    for pair in tqdm(pairs, unit="plot", desc="Quality metrics"):
        try:
            evaluations.append(
                evaluate_pair(
                    pair,
                    thresholds,
                    args.primary_iou,
                    args.ignore_overlap,
                    args.boundary_spacing,
                    exclude_incomplete,
                )
            )
        except Exception as error:
            errors.append(f"{pair.tile_id}: {type(error).__name__}: {error}")
    if errors:
        print(f"Evaluation errors: {len(errors)}", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    overall = [aggregate_metrics(evaluations, value) for value in thresholds]
    primary = next(
        metrics
        for metrics in overall
        if math.isclose(metrics.threshold, args.primary_iou)
    )
    overall_frame = pd.DataFrame([asdict(metrics) for metrics in overall])
    per_tile_frame = pd.DataFrame(
        [
            {"tile_id": evaluation.tile_id, **asdict(metrics)}
            for evaluation in evaluations
            for threshold, metrics in evaluation.metrics_by_threshold.items()
            if math.isclose(threshold, args.primary_iou)
        ]
    )
    matches_frame = pd.DataFrame(
        [row for evaluation in evaluations for row in evaluation.match_rows]
    )
    objects_frame = pd.DataFrame(
        [row for evaluation in evaluations for row in evaluation.object_rows]
    )

    shape_metric_columns = {
        "mean_absolute_area_error_m2": "absolute_area_error_m2",
        "mean_centroid_distance_m": "centroid_distance_m",
        "mean_hausdorff_distance_m": "hausdorff_distance_m",
        "mean_symmetric_boundary_distance_m": (
            "mean_symmetric_boundary_distance_m"
        ),
        "mean_p95_symmetric_boundary_distance_m": (
            "p95_symmetric_boundary_distance_m"
        ),
    }
    primary_shape_metrics = {
        output_column: (
            float(matches_frame[source_column].mean())
            if source_column in matches_frame.columns and not matches_frame.empty
            else math.nan
        )
        for output_column, source_column in shape_metric_columns.items()
    }
    for column in shape_metric_columns:
        overall_frame[column] = math.nan
    primary_row = np.isclose(overall_frame["threshold"], args.primary_iou)
    for column, value in primary_shape_metrics.items():
        overall_frame.loc[primary_row, column] = value

    if not matches_frame.empty:
        per_tile_shape = (
            matches_frame.groupby("tile_id")[
                list(shape_metric_columns.values())
            ]
            .mean()
            .rename(
                columns={
                    source: output
                    for output, source in shape_metric_columns.items()
                }
            )
            .reset_index()
        )
        per_tile_frame = per_tile_frame.merge(
            per_tile_shape, on="tile_id", how="left"
        )
    else:
        for column in shape_metric_columns:
            per_tile_frame[column] = math.nan

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(overall_frame, output_dir / SUMMARY_FILENAME)
    atomic_write_csv(per_tile_frame, output_dir / PER_TILE_FILENAME)
    atomic_write_csv(matches_frame, output_dir / MATCHES_FILENAME)
    atomic_write_csv(objects_frame, output_dir / OBJECTS_FILENAME)
    payload = json_safe(
        {
            "configuration": {
                "gt_directory": str(gt_dir),
                "gt_filename_prefix": args.gt_prefix,
                "gt_geopackage_layer": args.gt_layer,
                "prediction_directory": str(prediction_dir),
                "plots": len(pairs),
                "iou_thresholds": thresholds,
                "primary_iou_threshold": args.primary_iou,
                "exclude_incomplete_gt": exclude_incomplete,
                "ignored_prediction_overlap_threshold": args.ignore_overlap,
                "boundary_sample_spacing_metres": args.boundary_spacing,
            },
            "overall_metrics": [asdict(metrics) for metrics in overall],
            "primary_match_shape_metrics": primary_shape_metrics,
            "reports": {
                "overall": SUMMARY_FILENAME,
                "per_tile": PER_TILE_FILENAME,
                "matches": MATCHES_FILENAME,
                "unmatched_objects": OBJECTS_FILENAME,
            },
        }
    )
    atomic_write_json(payload, output_dir / JSON_FILENAME)

    print(f"GT crowns evaluated: {primary.gt_count}")
    print(f"Incomplete GT crowns ignored: {primary.ignored_gt_count}")
    print(f"Predicted crowns: {primary.prediction_count}")
    print(f"Predictions ignored at incomplete edges: {primary.ignored_prediction_count}")
    print(
        f"IoU >= {args.primary_iou:g}: TP={primary.true_positives}, "
        f"FP={primary.false_positives}, FN={primary.false_negatives}"
    )
    print(
        f"Precision={primary.precision:.4f}, Recall={primary.recall:.4f}, "
        f"F1/RQ={primary.f1:.4f}"
    )
    print(
        f"SQ/mean IoU={primary.segmentation_quality:.4f}, "
        f"PQ={primary.panoptic_quality:.4f}, "
        f"mean Dice={primary.mean_dice:.4f}"
    )
    if math.isfinite(primary_shape_metrics["mean_centroid_distance_m"]):
        print(
            "Mean matched-crown errors: "
            f"centroid={primary_shape_metrics['mean_centroid_distance_m']:.3f} m, "
            "boundary="
            f"{primary_shape_metrics['mean_symmetric_boundary_distance_m']:.3f} m, "
            "Hausdorff="
            f"{primary_shape_metrics['mean_hausdorff_distance_m']:.3f} m"
        )
    print(f"Reports written to: {output_dir}")
    print(f"Elapsed: {(time.monotonic() - start) / 60.0:.2f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
