#!/usr/bin/env python3
"""Create instance-segmentation crown ground truth from TreeScan LAZ labels.

For every ``chm_<LAZ-stem>.tif``, this script reads the matching LAZ file and
creates ``gt_<LAZ-stem>.gpkg``. The GeoPackage contains one georeferenced crown
polygon per positive LAS ``point_source_id`` that has a robust height record
in ``individual_tree_summary.csv``. This field is the stable tree identifier;
the TreeScan-specific extra ``treeID`` field is only an internal re-numbering.

Each crown is the top-down projection of its labeled 3D points onto the exact
grid of the matching CHM. High-Z annotation outliers are rejected using the
same per-tree height limits as ``laz_to_chm_tif.py``. Internal holes are filled
and only the largest connected component is retained, removing isolated label
noise without inventing a convex crown boundary.

The resulting polygons can be compared directly with predicted crowns using
IoU, Dice, boundary distance, precision/recall, or area-based metrics.

Run all CHM/LAZ pairs:

    python code/laz_to_crown_gt.py

Run one plot for testing:

    python code/laz_to_crown_gt.py --file Rem_Herby_2016_0702506.laz
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Importing the converter first configures GDAL/PROJ for the active environment
# and keeps LAZ metadata matching and height normalization consistent.
from laz_to_chm_tif import (
    CHUNK_SIZE_POINTS,
    DEFAULT_LAZ_DIR,
    DEFAULT_TREE_SUMMARY_FILENAME,
    HEIGHT_OUTLIER_TOLERANCE_METRES,
    TREE_ID_DIMENSION,
    find_tree_bases,
    year_insensitive_plot_key,
)

try:
    import geopandas as gpd
    import laspy
    import numpy as np
    import pyogrio
    import rasterio
    from rasterio.features import shapes
    from scipy.ndimage import (
        binary_closing,
        binary_fill_holes,
        generate_binary_structure,
        label,
    )
    from shapely.geometry import MultiPolygon, Polygon, shape
    from shapely.ops import unary_union
    from tqdm import tqdm
except ImportError as error:
    raise SystemExit(
        "Missing Python dependency. Use the project treescan environment or "
        "install laspy, rasterio, numpy, scipy, geopandas, shapely, pyogrio, "
        f"and tqdm. Original import error: {error}"
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent

# =============================================================================
# USER-TUNABLE CONFIGURATION
# =============================================================================
DEFAULT_CHM_DIR = RUN_DIR / "data_input_from_laz"
DEFAULT_GT_DIR = DEFAULT_CHM_DIR
CHM_PREFIX = "chm_"
CHM_SUFFIX = ".tif"
GT_PREFIX = "gt_"
GT_SUFFIX = ".gpkg"
GT_LAYER_NAME = "crowns_gt"

# Raster-mask cleanup. A value of 0 disables morphological closing.
MASK_CONNECTIVITY = 8  # Allowed values: 4 or 8.
MORPHOLOGICAL_CLOSING_ITERATIONS = 1
FILL_INTERNAL_HOLES = True
KEEP_LARGEST_COMPONENT = True
MIN_CROWN_AREA_M2 = 0.25

OVERWRITE_EXISTING_OUTPUTS = False
DEFAULT_WORKERS = 1
# =============================================================================
# END USER-TUNABLE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class TreeRecord:
    tree_id: int
    species_code: int
    species: str
    completely_inside: bool
    height_m: float
    reference_canopy_area_m2: float


@dataclass(frozen=True)
class PlotPair:
    chm_path: Path
    laz_path: Path
    source_stem: str


@dataclass(frozen=True)
class GroundTruthResult:
    pair: PlotPair
    output_path: Path
    status: str
    crowns: int = 0
    eligible_crowns: int = 0
    omitted_without_summary: int = 0
    omitted_small_or_empty: int = 0
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create crown ground-truth polygons from TreeScan labels."
    )
    parser.add_argument(
        "--chm-dir",
        type=Path,
        default=DEFAULT_CHM_DIR,
        help="Directory containing chm_<LAZ-stem>.tif files.",
    )
    parser.add_argument(
        "--laz-dir",
        type=Path,
        default=DEFAULT_LAZ_DIR,
        help="Directory containing matching LAZ files and tree summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GT_DIR,
        help="Directory for gt_<LAZ-stem>.gpkg; defaults beside the CHMs.",
    )
    parser.add_argument(
        "--tree-summary",
        type=Path,
        help=(
            "Per-tree summary CSV. Defaults to individual_tree_summary.csv "
            "inside --laz-dir."
        ),
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="NAME",
        help="Process only this LAZ, CHM, GT filename, or source stem. Repeatable.",
    )
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Write only trees flagged completely_inside=1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE_EXISTING_OUTPUTS,
        help="Replace an existing GT GeoPackage.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of LAZ plots processed concurrently; default 1. "
            "Each worker can use substantial memory."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pairs and metadata without reading LAZ points.",
    )
    return parser.parse_args()


def optional_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def read_tree_records(
    summary_path: Path,
) -> dict[str, dict[int, TreeRecord]]:
    records: dict[str, dict[int, TreeRecord]] = {}
    with summary_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "source_file",
            "treeID",
            "species_code",
            "species",
            "completely_inside",
            "height_m",
            "canopy_area_m2",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Missing tree-summary columns: " + ", ".join(sorted(missing))
            )

        for row in reader:
            source_file = row["source_file"].strip()
            tree_id = int(row["treeID"])
            height_m = optional_float(row["height_m"])
            if tree_id <= 0 or not math.isfinite(height_m) or height_m < 0:
                continue
            source_records = records.setdefault(source_file, {})
            if tree_id in source_records:
                raise ValueError(
                    f"Duplicate tree summary: {source_file}, treeID {tree_id}"
                )
            source_records[tree_id] = TreeRecord(
                tree_id=tree_id,
                species_code=int(row["species_code"]),
                species=row["species"].strip(),
                completely_inside=row["completely_inside"].strip() == "1",
                height_m=height_m,
                reference_canopy_area_m2=optional_float(row["canopy_area_m2"]),
            )
    return records


def match_plot_records(
    laz_name: str,
    records_by_file: dict[str, dict[int, TreeRecord]],
) -> tuple[dict[int, TreeRecord] | None, str | None]:
    if laz_name in records_by_file:
        return records_by_file[laz_name], None

    plot_key = year_insensitive_plot_key(laz_name)
    matches = [
        name
        for name in records_by_file
        if year_insensitive_plot_key(name) == plot_key
    ]
    if len(matches) == 1:
        return records_by_file[matches[0]], matches[0]
    return None, None


def normalize_requested_name(name: str) -> str:
    normalized = Path(name).name
    for suffix in (GT_SUFFIX, CHM_SUFFIX, ".laz"):
        if normalized.lower().endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    for prefix in (GT_PREFIX, CHM_PREFIX):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized


def discover_pairs(
    chm_dir: Path,
    laz_dir: Path,
    requested: list[str],
) -> list[PlotPair]:
    chm_paths = sorted(
        path
        for path in chm_dir.iterdir()
        if path.is_file()
        and path.name.startswith(CHM_PREFIX)
        and path.suffix.lower() == CHM_SUFFIX
    )
    pairs_by_stem: dict[str, PlotPair] = {}
    missing_laz: list[str] = []
    for chm_path in chm_paths:
        source_stem = chm_path.stem.removeprefix(CHM_PREFIX)
        laz_path = laz_dir / f"{source_stem}.laz"
        if not laz_path.is_file():
            missing_laz.append(laz_path.name)
            continue
        pairs_by_stem[source_stem] = PlotPair(chm_path, laz_path, source_stem)
    if missing_laz:
        raise FileNotFoundError(
            "Matching LAZ files not found: " + ", ".join(missing_laz)
        )

    if not requested:
        return list(pairs_by_stem.values())

    selected: list[PlotPair] = []
    missing_requested: list[str] = []
    for name in requested:
        source_stem = normalize_requested_name(name)
        pair = pairs_by_stem.get(source_stem)
        if pair is None:
            missing_requested.append(name)
        elif pair not in selected:
            selected.append(pair)
    if missing_requested:
        raise FileNotFoundError(
            "Requested CHM/LAZ pairs not found: " + ", ".join(missing_requested)
        )
    return selected


def connectivity_structure() -> np.ndarray:
    if MASK_CONNECTIVITY == 4:
        return generate_binary_structure(2, 1)
    if MASK_CONNECTIVITY == 8:
        return generate_binary_structure(2, 2)
    raise ValueError("MASK_CONNECTIVITY must be 4 or 8")


def build_tree_masks(
    pair: PlotPair,
    tree_bases: np.ndarray,
    tree_records: dict[int, TreeRecord],
) -> tuple[np.ndarray, np.ndarray]:
    """Project accepted tree points onto the exact matching CHM grid."""
    with rasterio.open(pair.chm_path) as chm_source:
        width = chm_source.width
        height = chm_source.height
        bounds = chm_source.bounds
        x_resolution, y_resolution = chm_source.res

    centre_x = (bounds.left + bounds.right) / 2.0
    centre_y = (bounds.bottom + bounds.top) / 2.0
    cell_count = width * height
    masks = np.zeros((tree_bases.size, height, width), dtype=bool)
    point_counts = np.zeros(tree_bases.size, dtype=np.int64)
    height_limits = np.full(tree_bases.size, -np.inf, dtype=np.float64)
    for tree_id, record in tree_records.items():
        if tree_id < tree_bases.size:
            height_limits[tree_id] = record.height_m

    with laspy.open(pair.laz_path) as reader:
        dimensions = set(reader.header.point_format.dimension_names)
        if TREE_ID_DIMENSION not in dimensions:
            raise ValueError(f"LAZ has no {TREE_ID_DIMENSION!r} dimension")

        for points in reader.chunk_iterator(CHUNK_SIZE_POINTS):
            x_local = np.asarray(points.x, dtype=np.float64)
            y_local = np.asarray(points.y, dtype=np.float64)
            z_values = np.asarray(points.z, dtype=np.float64)
            tree_ids = np.asarray(points[TREE_ID_DIMENSION], dtype=np.int64)

            in_id_range = (tree_ids > 0) & (tree_ids < tree_bases.size)
            positions = np.flatnonzero(
                in_id_range
                & np.isfinite(x_local)
                & np.isfinite(y_local)
                & np.isfinite(z_values)
            )
            if positions.size == 0:
                continue

            selected_ids = tree_ids[positions]
            normalized_heights = (
                z_values[positions] - tree_bases[selected_ids]
            )
            within_height = (
                normalized_heights >= 0
            ) & (
                normalized_heights
                <= height_limits[selected_ids] + HEIGHT_OUTLIER_TOLERANCE_METRES
            )
            positions = positions[within_height]
            selected_ids = selected_ids[within_height]
            if positions.size == 0:
                continue

            x_global = centre_x + x_local[positions]
            y_global = centre_y + y_local[positions]
            inside = (
                (x_global >= bounds.left)
                & (x_global < bounds.right)
                & (y_global > bounds.bottom)
                & (y_global <= bounds.top)
            )
            positions = positions[inside]
            selected_ids = selected_ids[inside]
            x_global = x_global[inside]
            y_global = y_global[inside]
            if positions.size == 0:
                continue

            columns = np.floor(
                (x_global - bounds.left) / x_resolution
            ).astype(np.int64)
            rows = np.floor(
                (bounds.top - y_global) / y_resolution
            ).astype(np.int64)
            rows = np.minimum(rows, height - 1)
            cells = rows * width + columns

            np.add.at(point_counts, selected_ids, 1)
            mask_indices = selected_ids * cell_count + cells
            masks.reshape(-1)[mask_indices] = True

    return masks, point_counts


def largest_component(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    labeled, component_count = label(mask, structure=structure)
    if component_count <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def clean_crown_mask(mask: np.ndarray) -> np.ndarray:
    structure = connectivity_structure()
    result = mask
    if KEEP_LARGEST_COMPONENT:
        result = largest_component(result, structure)
    if MORPHOLOGICAL_CLOSING_ITERATIONS > 0:
        result = binary_closing(
            result,
            structure=structure,
            iterations=MORPHOLOGICAL_CLOSING_ITERATIONS,
            border_value=1,
        )
    if FILL_INTERNAL_HOLES:
        result = binary_fill_holes(result)
    if KEEP_LARGEST_COMPONENT:
        result = largest_component(result, structure)
    return np.asarray(result, dtype=bool)


def polygonal_geometry(mask: np.ndarray, transform: rasterio.Affine) -> MultiPolygon:
    pieces = [
        shape(geometry_mapping)
        for geometry_mapping, value in shapes(
            mask.astype(np.uint8),
            mask=mask,
            transform=transform,
            connectivity=MASK_CONNECTIVITY,
        )
        if int(value) == 1
    ]
    if not pieces:
        return MultiPolygon()

    merged = unary_union(pieces)
    if not merged.is_valid:
        merged = merged.buffer(0)
    polygons: list[Polygon] = []
    if isinstance(merged, Polygon):
        polygons = [merged]
    elif isinstance(merged, MultiPolygon):
        polygons = list(merged.geoms)
    elif hasattr(merged, "geoms"):
        polygons = [geometry for geometry in merged.geoms if isinstance(geometry, Polygon)]
    return MultiPolygon(polygons)


def build_ground_truth(
    pair: PlotPair,
    tree_records: dict[int, TreeRecord],
    complete_only: bool,
) -> tuple[gpd.GeoDataFrame, int, int]:
    tree_bases, _ = find_tree_bases(pair.laz_path)
    present_tree_ids = set(
        int(tree_id)
        for tree_id in np.flatnonzero(np.isfinite(tree_bases))
        if tree_id > 0
    )
    missing_records = sorted(present_tree_ids - set(tree_records))
    if missing_records:
        preview = ", ".join(str(tree_id) for tree_id in missing_records[:10])
        if len(missing_records) > 10:
            preview += ", ..."
        tqdm.write(
            f"{pair.laz_path.name}: omitting tree IDs without summaries: {preview}"
        )

    masks, point_counts = build_tree_masks(pair, tree_bases, tree_records)
    with rasterio.open(pair.chm_path) as chm_source:
        transform = chm_source.transform
        crs = chm_source.crs
        pixel_area = abs(transform.a * transform.e)

    rows: list[dict] = []
    geometries: list[MultiPolygon] = []
    omitted_small_or_empty = 0
    for tree_id in sorted(present_tree_ids & set(tree_records)):
        record = tree_records[tree_id]
        if complete_only and not record.completely_inside:
            continue
        raw_mask = masks[tree_id]
        clean_mask = clean_crown_mask(raw_mask)
        clean_area = float(np.count_nonzero(clean_mask) * pixel_area)
        if clean_area < MIN_CROWN_AREA_M2:
            omitted_small_or_empty += 1
            continue
        geometry = polygonal_geometry(clean_mask, transform)
        if geometry.is_empty or not geometry.is_valid:
            omitted_small_or_empty += 1
            continue

        rows.append(
            {
                "treeID": tree_id,
                "species_code": record.species_code,
                "species": record.species,
                "height_m": record.height_m,
                "completely_inside": int(record.completely_inside),
                "evaluation_eligible": int(record.completely_inside),
                "point_count_used": int(point_counts[tree_id]),
                "raw_cells": int(np.count_nonzero(raw_mask)),
                "clean_cells": int(np.count_nonzero(clean_mask)),
                "reference_canopy_area_m2": record.reference_canopy_area_m2,
                "gt_area_m2": float(geometry.area),
                "source_laz": pair.laz_path.name,
            }
        )
        geometries.append(geometry)

    if not rows:
        raise ValueError("No crown polygons remained after filtering")
    ground_truth = gpd.GeoDataFrame(rows, geometry=geometries, crs=crs)
    ground_truth = ground_truth.sort_values("treeID").reset_index(drop=True)
    return ground_truth, len(missing_records), omitted_small_or_empty


def verify_ground_truth(path: Path, expected_count: int, expected_crs) -> None:
    info = pyogrio.read_info(path, layer=GT_LAYER_NAME)
    if int(info["features"]) != expected_count:
        raise RuntimeError(
            f"GT verification found {info['features']} features, expected {expected_count}"
        )
    if info["crs"] != expected_crs.to_string():
        raise RuntimeError(
            f"GT verification found CRS {info['crs']}, expected {expected_crs}"
        )
    check = gpd.read_file(path, layer=GT_LAYER_NAME)
    if check["treeID"].duplicated().any():
        raise RuntimeError("GT verification found duplicate tree IDs")
    if check.geometry.is_empty.any() or not check.geometry.is_valid.all():
        raise RuntimeError("GT verification found empty or invalid geometries")


def write_ground_truth_atomically(
    output_path: Path,
    ground_truth: gpd.GeoDataFrame,
) -> None:
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}{GT_SUFFIX}"
    )
    try:
        ground_truth.to_file(
            temporary_path,
            layer=GT_LAYER_NAME,
            driver="GPKG",
            engine="pyogrio",
            index=False,
        )
        verify_ground_truth(temporary_path, len(ground_truth), ground_truth.crs)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def process_pair(
    pair: PlotPair,
    output_dir: Path,
    tree_records: dict[int, TreeRecord],
    complete_only: bool,
    overwrite: bool,
) -> GroundTruthResult:
    output_path = output_dir / f"{GT_PREFIX}{pair.source_stem}{GT_SUFFIX}"
    if output_path.exists() and not overwrite:
        try:
            feature_count = int(
                pyogrio.read_info(output_path, layer=GT_LAYER_NAME)["features"]
            )
        except Exception:
            feature_count = 0
        return GroundTruthResult(
            pair,
            output_path,
            "skipped",
            crowns=feature_count,
            detail="exists",
        )

    try:
        ground_truth, missing_records, omitted_small = build_ground_truth(
            pair, tree_records, complete_only
        )
        write_ground_truth_atomically(output_path, ground_truth)
        eligible = int(ground_truth["evaluation_eligible"].sum())
        return GroundTruthResult(
            pair,
            output_path,
            "created",
            crowns=len(ground_truth),
            eligible_crowns=eligible,
            omitted_without_summary=missing_records,
            omitted_small_or_empty=omitted_small,
        )
    except Exception as error:
        return GroundTruthResult(
            pair,
            output_path,
            "error",
            detail=f"{type(error).__name__}: {error}",
        )


def main() -> int:
    args = parse_args()
    start = time.monotonic()
    chm_dir = args.chm_dir.resolve()
    laz_dir = args.laz_dir.resolve()
    output_dir = args.output_dir.resolve()
    summary_path = (
        args.tree_summary.resolve()
        if args.tree_summary is not None
        else laz_dir / DEFAULT_TREE_SUMMARY_FILENAME
    )

    for path, description in (
        (chm_dir, "CHM directory"),
        (laz_dir, "LAZ directory"),
    ):
        if not path.is_dir():
            print(f"{description} does not exist: {path}", file=sys.stderr)
            return 2
    if not summary_path.is_file():
        print(f"Tree summary does not exist: {summary_path}", file=sys.stderr)
        return 2

    try:
        pairs = discover_pairs(chm_dir, laz_dir, args.file)
        if not pairs:
            raise FileNotFoundError(f"No CHM files found in {chm_dir}")
        records_by_file = read_tree_records(summary_path)
        records_for_pair: dict[str, dict[int, TreeRecord]] = {}
        aliases: list[tuple[str, str]] = []
        missing_summaries: list[str] = []
        for pair in pairs:
            records, alias = match_plot_records(pair.laz_path.name, records_by_file)
            if records is None:
                missing_summaries.append(pair.laz_path.name)
                continue
            records_for_pair[pair.source_stem] = records
            if alias is not None:
                aliases.append((pair.laz_path.name, alias))
        if missing_summaries:
            raise ValueError(
                "Missing tree summaries for: " + ", ".join(missing_summaries)
            )
        if args.workers < 1:
            raise ValueError("--workers must be at least 1")
        connectivity_structure()
    except Exception as error:
        print(f"Input validation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    print(f"CHM/LAZ pairs: {len(pairs)}")
    print(f"CHM directory: {chm_dir}")
    print(f"LAZ directory: {laz_dir}")
    print(f"Tree summary: {summary_path}")
    print(f"GT output directory: {output_dir}")
    print(f"Workers: {args.workers}")
    for laz_name, summary_name in aliases:
        print(
            f"Note: using tree summary {summary_name} for {laz_name} "
            "(unique plot ID; year differs)."
        )

    if args.dry_run:
        for pair in pairs:
            output_path = output_dir / f"{GT_PREFIX}{pair.source_stem}{GT_SUFFIX}"
            print(f"{pair.chm_path.name} + {pair.laz_path.name} -> {output_path.name}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.workers == 1:
        results = [
            process_pair(
                pair,
                output_dir,
                records_for_pair[pair.source_stem],
                args.complete_only,
                args.overwrite,
            )
            for pair in tqdm(pairs, unit="plot", desc="Crown GT")
        ]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_pair,
                    pair,
                    output_dir,
                    records_for_pair[pair.source_stem],
                    args.complete_only,
                    args.overwrite,
                )
                for pair in pairs
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                unit="plot",
                desc="Crown GT",
            ):
                results.append(future.result())

    created = [result for result in results if result.status == "created"]
    skipped = [result for result in results if result.status == "skipped"]
    errors = [result for result in results if result.status == "error"]
    print(f"Created: {len(created)}")
    print(f"Skipped existing: {len(skipped)}")
    if created:
        print(f"Crown polygons created: {sum(item.crowns for item in created)}")
        print(
            "Evaluation-eligible complete crowns: "
            f"{sum(item.eligible_crowns for item in created)}"
        )
        omitted_summary = sum(item.omitted_without_summary for item in created)
        omitted_geometry = sum(item.omitted_small_or_empty for item in created)
        if omitted_summary:
            print(f"Trees omitted without summaries: {omitted_summary}")
        if omitted_geometry:
            print(f"Trees omitted with empty/small contours: {omitted_geometry}")
    if errors:
        print(f"Errors: {len(errors)}", file=sys.stderr)
        for result in errors:
            print(
                f"  - {result.pair.source_stem}: {result.detail}",
                file=sys.stderr,
            )

    elapsed_minutes = (time.monotonic() - start) / 60.0
    print(f"Elapsed: {elapsed_minutes:.2f} min")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
