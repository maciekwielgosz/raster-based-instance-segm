#!/usr/bin/env python3
"""Create exclusive top-view crown GT from labeled TreeScan LAZ points.

For every matching CHM/LAZ pair, each 0.5 m canopy cell is assigned to exactly
one canonical tree ID: the tree whose accepted labeled point supplies the
highest normalized canopy height in that cell. Ties are resolved by choosing
the smaller tree ID. The result is saved both as an integer label GeoTIFF and
as non-overlapping crown polygons:

    topmost_gt_<LAZ-stem>.tif
    topmost_gt_<LAZ-stem>.gpkg  (layer: crowns_gt_topmost)

The label raster uses -1 for CHM NoData, 0 for background/below-threshold
cells, and positive ``point_source_id`` values for visible tree instances.
Unlike ``laz_to_crown_gt.py``, this representation cannot contain overlapping
crowns and is therefore directly compatible with single-label watershed
segmentation of the CHM.

Run all plots from ``run_r``:

    python code/laz_to_topmost_crown_gt.py --workers 3
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from laz_to_chm_tif import (
    CHUNK_SIZE_POINTS,
    DEFAULT_LAZ_DIR,
    DEFAULT_TREE_SUMMARY_FILENAME,
    HEIGHT_OUTLIER_TOLERANCE_METRES,
    TREE_ID_DIMENSION,
    find_tree_bases,
)
from laz_to_crown_gt import TreeRecord, match_plot_records, read_tree_records

try:
    import geopandas as gpd
    import laspy
    import numpy as np
    import pyogrio
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import MultiPolygon, Polygon, shape
    from shapely.ops import unary_union
    from tqdm import tqdm
except ImportError as error:
    raise SystemExit(
        "Missing Python dependency. Use the project treescan environment or "
        "install laspy, rasterio, numpy, geopandas, shapely, pyogrio, and tqdm. "
        f"Original import error: {error}"
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent

# =============================================================================
# USER-TUNABLE CONFIGURATION
# =============================================================================
DEFAULT_CHM_DIR = RUN_DIR / "data_input_from_laz"
DEFAULT_OUTPUT_DIR = DEFAULT_CHM_DIR

CHM_PREFIX = "chm_"
CHM_SUFFIX = ".tif"
OUTPUT_PREFIX = "topmost_gt_"
RASTER_SUFFIX = ".tif"
VECTOR_SUFFIX = ".gpkg"
VECTOR_LAYER_NAME = "crowns_gt_topmost"

# Must normally equal the segmentation script's MIN_HEIGHT. Setting this to
# zero retains every labeled tree cell, including trunk/base-only cells.
MIN_CANOPY_HEIGHT_METRES = 2.0
POLYGON_CONNECTIVITY = 4
LABEL_NODATA_VALUE = -1
DEFAULT_WORKERS = 1
OVERWRITE_EXISTING_OUTPUTS = False
# =============================================================================
# END USER-TUNABLE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class PlotPair:
    chm_path: Path
    laz_path: Path
    source_stem: str


@dataclass(frozen=True)
class TopmostResult:
    pair: PlotPair
    raster_path: Path
    vector_path: Path
    status: str
    visible_trees: int = 0
    visible_cells: int = 0
    occluded_or_below_threshold_trees: int = 0
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create exclusive topmost-point crown GT from TreeScan LAZ labels."
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
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for topmost_gt_<stem>.tif/.gpkg outputs.",
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
        help="Process only this LAZ/CHM/output name or source stem. Repeatable.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of plots processed concurrently; default 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE_EXISTING_OUTPUTS,
        help="Replace existing topmost GT raster/vector pairs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pairs and metadata without reading LAZ points.",
    )
    return parser.parse_args()


def normalize_requested_name(name: str) -> str:
    normalized = Path(name).name
    for suffix in (VECTOR_SUFFIX, RASTER_SUFFIX, ".laz"):
        if normalized.lower().endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    for prefix in (OUTPUT_PREFIX, CHM_PREFIX):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized


def discover_pairs(
    chm_dir: Path,
    laz_dir: Path,
    requested: list[str],
) -> list[PlotPair]:
    pairs: dict[str, PlotPair] = {}
    missing_laz: list[str] = []
    for chm_path in sorted(chm_dir.glob(f"{CHM_PREFIX}*{CHM_SUFFIX}")):
        source_stem = chm_path.stem.removeprefix(CHM_PREFIX)
        laz_path = laz_dir / f"{source_stem}.laz"
        if not laz_path.is_file():
            missing_laz.append(laz_path.name)
            continue
        pairs[source_stem] = PlotPair(chm_path, laz_path, source_stem)
    if missing_laz:
        raise FileNotFoundError(
            "Matching LAZ files not found: " + ", ".join(missing_laz)
        )
    if not pairs:
        raise FileNotFoundError(f"No {CHM_PREFIX}*.tif files in {chm_dir}")
    if not requested:
        return list(pairs.values())

    selected: list[PlotPair] = []
    missing: list[str] = []
    for name in requested:
        source_stem = normalize_requested_name(name)
        pair = pairs.get(source_stem)
        if pair is None:
            missing.append(name)
        elif pair not in selected:
            selected.append(pair)
    if missing:
        raise FileNotFoundError(
            "Requested CHM/LAZ pairs not found: " + ", ".join(missing)
        )
    return selected


def validate_configuration() -> None:
    if MIN_CANOPY_HEIGHT_METRES < 0 or not math.isfinite(
        MIN_CANOPY_HEIGHT_METRES
    ):
        raise ValueError("MIN_CANOPY_HEIGHT_METRES must be finite and non-negative")
    if POLYGON_CONNECTIVITY not in (4, 8):
        raise ValueError("POLYGON_CONNECTIVITY must be 4 or 8")
    if LABEL_NODATA_VALUE >= 0:
        raise ValueError("LABEL_NODATA_VALUE must be negative")


def build_topmost_labels(
    pair: PlotPair,
    tree_bases: np.ndarray,
    tree_records: dict[int, TreeRecord],
) -> tuple[np.ndarray, np.ndarray, rasterio.Affine, object, dict]:
    """Assign each canopy cell to the tree supplying its highest CHM point."""
    with rasterio.open(pair.chm_path) as source:
        chm = source.read(1, masked=True)
        transform = source.transform
        crs = source.crs
        profile = source.profile.copy()
        bounds = source.bounds
        width = source.width
        height = source.height
        x_resolution, y_resolution = source.res

    if crs is None:
        raise ValueError(f"CHM has no CRS: {pair.chm_path.name}")
    if width * height == 0:
        raise ValueError(f"CHM is empty: {pair.chm_path.name}")
    centre_x = (bounds.left + bounds.right) / 2.0
    centre_y = (bounds.bottom + bounds.top) / 2.0
    cell_count = width * height
    best_height = np.full(cell_count, -np.inf, dtype=np.float64)
    best_tree_id = np.zeros(cell_count, dtype=np.int64)

    height_limits = np.full(tree_bases.size, -np.inf, dtype=np.float64)
    for tree_id, record in tree_records.items():
        if tree_id < height_limits.size:
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

            valid_ids = (tree_ids > 0) & (tree_ids < tree_bases.size)
            positions = np.flatnonzero(
                valid_ids
                & np.isfinite(x_local)
                & np.isfinite(y_local)
                & np.isfinite(z_values)
            )
            if positions.size == 0:
                continue
            selected_ids = tree_ids[positions]
            normalized_heights = z_values[positions] - tree_bases[selected_ids]
            accepted = (
                (normalized_heights >= MIN_CANOPY_HEIGHT_METRES)
                & (
                    normalized_heights
                    <= height_limits[selected_ids]
                    + HEIGHT_OUTLIER_TOLERANCE_METRES
                )
            )
            positions = positions[accepted]
            selected_ids = selected_ids[accepted]
            normalized_heights = normalized_heights[accepted]
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
            selected_ids = selected_ids[inside]
            normalized_heights = normalized_heights[inside]
            x_global = x_global[inside]
            y_global = y_global[inside]
            if selected_ids.size == 0:
                continue

            columns = np.floor(
                (x_global - bounds.left) / x_resolution
            ).astype(np.int64)
            rows = np.floor(
                (bounds.top - y_global) / y_resolution
            ).astype(np.int64)
            rows = np.minimum(rows, height - 1)
            cells = rows * width + columns

            chunk_maximum = np.full(cell_count, -np.inf, dtype=np.float64)
            np.maximum.at(chunk_maximum, cells, normalized_heights)
            is_chunk_maximum = normalized_heights == chunk_maximum[cells]
            chunk_tree_id = np.full(
                cell_count, np.iinfo(np.int64).max, dtype=np.int64
            )
            np.minimum.at(
                chunk_tree_id,
                cells[is_chunk_maximum],
                selected_ids[is_chunk_maximum],
            )

            higher = chunk_maximum > best_height
            tied_lower_id = (
                (chunk_maximum == best_height)
                & (chunk_tree_id < best_tree_id)
            )
            replace = higher | tied_lower_id
            best_height[replace] = chunk_maximum[replace]
            best_tree_id[replace] = chunk_tree_id[replace]

    chm_data = chm.filled(np.nan).astype(np.float64)
    valid_chm = ~np.ma.getmaskarray(chm)
    expected_canopy = valid_chm & (chm_data >= MIN_CANOPY_HEIGHT_METRES)
    winner_canopy = best_tree_id.reshape((height, width)) > 0
    if not np.array_equal(expected_canopy, winner_canopy):
        difference = int(np.count_nonzero(expected_canopy != winner_canopy))
        raise RuntimeError(
            f"Topmost labels disagree with the CHM canopy in {difference} cells"
        )
    if np.any(winner_canopy):
        winner_heights = best_height.reshape((height, width))[winner_canopy]
        if not np.allclose(
            winner_heights,
            chm_data[winner_canopy],
            atol=1e-5,
            rtol=0,
        ):
            maximum_difference = float(
                np.max(np.abs(winner_heights - chm_data[winner_canopy]))
            )
            raise RuntimeError(
                "Topmost-point heights do not reproduce CHM values; "
                f"maximum difference {maximum_difference:g} m"
            )

    labels = np.where(valid_chm, 0, LABEL_NODATA_VALUE).astype(np.int32)
    labels[winner_canopy] = best_tree_id.reshape((height, width))[
        winner_canopy
    ].astype(np.int32)
    return labels, best_height.reshape((height, width)), transform, crs, profile


def polygonize_labels(
    pair: PlotPair,
    labels: np.ndarray,
    heights: np.ndarray,
    transform: rasterio.Affine,
    crs,
    tree_records: dict[int, TreeRecord],
) -> gpd.GeoDataFrame:
    pieces: dict[int, list[Polygon]] = {}
    for geometry_mapping, value in shapes(
        labels,
        mask=labels > 0,
        transform=transform,
        connectivity=POLYGON_CONNECTIVITY,
    ):
        tree_id = int(value)
        geometry = shape(geometry_mapping)
        if isinstance(geometry, Polygon):
            pieces.setdefault(tree_id, []).append(geometry)

    rows: list[dict] = []
    geometries: list[MultiPolygon] = []
    for tree_id in sorted(pieces):
        record = tree_records.get(tree_id)
        if record is None:
            raise ValueError(f"Visible tree ID {tree_id} has no summary record")
        merged = unary_union(pieces[tree_id])
        polygons: list[Polygon]
        if isinstance(merged, Polygon):
            polygons = [merged]
        elif isinstance(merged, MultiPolygon):
            polygons = list(merged.geoms)
        else:
            polygons = [
                geometry
                for geometry in getattr(merged, "geoms", [])
                if isinstance(geometry, Polygon)
            ]
        geometry = MultiPolygon(polygons)
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"Invalid polygonized geometry for tree ID {tree_id}")
        tree_cells = labels == tree_id
        rows.append(
            {
                "treeID": tree_id,
                "species_code": record.species_code,
                "species": record.species,
                "height_m": record.height_m,
                "completely_inside": int(record.completely_inside),
                "evaluation_eligible": int(record.completely_inside),
                "visible_cells": int(np.count_nonzero(tree_cells)),
                "topmost_max_height_m": float(np.max(heights[tree_cells])),
                "gt_area_m2": float(geometry.area),
                "source_laz": pair.laz_path.name,
                "gt_method": "topmost_chm_cell",
            }
        )
        geometries.append(geometry)
    if not rows:
        raise ValueError("No topmost crown instances were created")
    result = gpd.GeoDataFrame(rows, geometry=geometries, crs=crs)
    return result.sort_values("treeID").reset_index(drop=True)


def write_outputs_atomically(
    raster_path: Path,
    vector_path: Path,
    labels: np.ndarray,
    profile: dict,
    polygons: gpd.GeoDataFrame,
    source_laz: str,
) -> None:
    temporary_raster = raster_path.with_name(
        f".{raster_path.stem}.{uuid.uuid4().hex}{RASTER_SUFFIX}"
    )
    temporary_vector = vector_path.with_name(
        f".{vector_path.stem}.{uuid.uuid4().hex}{VECTOR_SUFFIX}"
    )
    raster_profile = profile.copy()
    raster_profile.update(
        dtype="int32",
        count=1,
        nodata=LABEL_NODATA_VALUE,
        compress="lzw",
        predictor=2,
    )
    try:
        with rasterio.open(temporary_raster, "w", **raster_profile) as destination:
            destination.write(labels, 1)
            destination.set_band_description(1, "topmost crown instance ID")
            destination.update_tags(
                GT_METHOD="topmost_chm_cell",
                SOURCE_LAZ=source_laz,
                BACKGROUND_VALUE="0",
                NODATA_VALUE=str(LABEL_NODATA_VALUE),
                MIN_CANOPY_HEIGHT_METRES=str(MIN_CANOPY_HEIGHT_METRES),
                TREE_ID_SOURCE=TREE_ID_DIMENSION,
            )
        with rasterio.open(temporary_raster) as check:
            raster_checks = (
                check.count == 1,
                check.dtypes[0] == "int32",
                check.nodata == LABEL_NODATA_VALUE,
                check.width == labels.shape[1],
                check.height == labels.shape[0],
                check.crs == polygons.crs,
                np.array_equal(check.read(1), labels),
            )
        if not all(raster_checks):
            raise RuntimeError("Topmost label raster verification failed")

        polygons.to_file(
            temporary_vector,
            layer=VECTOR_LAYER_NAME,
            driver="GPKG",
            engine="pyogrio",
            index=False,
        )
        info = pyogrio.read_info(temporary_vector, layer=VECTOR_LAYER_NAME)
        check_polygons = gpd.read_file(
            temporary_vector, layer=VECTOR_LAYER_NAME
        )
        if int(info["features"]) != len(polygons):
            raise RuntimeError("Topmost polygon feature-count verification failed")
        if check_polygons["treeID"].duplicated().any():
            raise RuntimeError("Topmost polygons contain duplicate tree IDs")
        if (
            check_polygons.geometry.is_empty.any()
            or not check_polygons.geometry.is_valid.all()
        ):
            raise RuntimeError("Topmost polygon geometry verification failed")

        os.replace(temporary_raster, raster_path)
        os.replace(temporary_vector, vector_path)
    finally:
        temporary_raster.unlink(missing_ok=True)
        temporary_vector.unlink(missing_ok=True)


def process_pair(
    pair: PlotPair,
    output_dir: Path,
    tree_records: dict[int, TreeRecord],
    overwrite: bool,
) -> TopmostResult:
    raster_path = output_dir / f"{OUTPUT_PREFIX}{pair.source_stem}{RASTER_SUFFIX}"
    vector_path = output_dir / f"{OUTPUT_PREFIX}{pair.source_stem}{VECTOR_SUFFIX}"
    raster_exists = raster_path.exists()
    vector_exists = vector_path.exists()
    if raster_exists and vector_exists and not overwrite:
        try:
            visible_trees = int(
                pyogrio.read_info(vector_path, layer=VECTOR_LAYER_NAME)["features"]
            )
        except Exception:
            visible_trees = 0
        return TopmostResult(
            pair,
            raster_path,
            vector_path,
            "skipped",
            visible_trees=visible_trees,
            detail="paired outputs exist",
        )
    if raster_exists != vector_exists and not overwrite:
        return TopmostResult(
            pair,
            raster_path,
            vector_path,
            "error",
            detail="asymmetric raster/vector outputs; use --overwrite",
        )

    try:
        tree_bases, annotated_tree_count = find_tree_bases(pair.laz_path)
        present_tree_ids = {
            int(tree_id)
            for tree_id in np.flatnonzero(np.isfinite(tree_bases))
            if tree_id > 0
        }
        missing_records = sorted(present_tree_ids - set(tree_records))
        if missing_records:
            raise ValueError(
                "Tree IDs lack summary records: "
                + ", ".join(str(value) for value in missing_records[:20])
            )
        labels, heights, transform, crs, profile = build_topmost_labels(
            pair, tree_bases, tree_records
        )
        polygons = polygonize_labels(
            pair, labels, heights, transform, crs, tree_records
        )
        visible_tree_count = len(polygons)
        write_outputs_atomically(
            raster_path,
            vector_path,
            labels,
            profile,
            polygons,
            pair.laz_path.name,
        )
        return TopmostResult(
            pair,
            raster_path,
            vector_path,
            "created",
            visible_trees=visible_tree_count,
            visible_cells=int(np.count_nonzero(labels > 0)),
            occluded_or_below_threshold_trees=(
                annotated_tree_count - visible_tree_count
            ),
        )
    except Exception as error:
        return TopmostResult(
            pair,
            raster_path,
            vector_path,
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
    try:
        if not chm_dir.is_dir():
            raise FileNotFoundError(f"CHM directory does not exist: {chm_dir}")
        if not laz_dir.is_dir():
            raise FileNotFoundError(f"LAZ directory does not exist: {laz_dir}")
        if not summary_path.is_file():
            raise FileNotFoundError(f"Tree summary does not exist: {summary_path}")
        if args.workers < 1:
            raise ValueError("--workers must be at least 1")
        validate_configuration()
        pairs = discover_pairs(chm_dir, laz_dir, args.file)
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
    except Exception as error:
        print(
            f"Input validation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2

    print(f"CHM/LAZ pairs: {len(pairs)}")
    print(f"CHM directory: {chm_dir}")
    print(f"LAZ directory: {laz_dir}")
    print(f"Tree summary: {summary_path}")
    print(f"Topmost GT output directory: {output_dir}")
    print(f"Minimum canopy height: {MIN_CANOPY_HEIGHT_METRES:g} m")
    print(f"Workers: {args.workers}")
    for laz_name, summary_name in aliases:
        print(
            f"Note: using tree summary {summary_name} for {laz_name} "
            "(unique plot ID; year differs)."
        )
    if args.dry_run:
        for pair in pairs:
            print(
                f"{pair.chm_path.name} + {pair.laz_path.name} -> "
                f"{OUTPUT_PREFIX}{pair.source_stem}.tif/.gpkg"
            )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.workers == 1:
        results = [
            process_pair(
                pair,
                output_dir,
                records_for_pair[pair.source_stem],
                args.overwrite,
            )
            for pair in tqdm(pairs, unit="plot", desc="Topmost crown GT")
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
                    args.overwrite,
                )
                for pair in pairs
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                unit="plot",
                desc="Topmost crown GT",
            ):
                results.append(future.result())

    created = [result for result in results if result.status == "created"]
    skipped = [result for result in results if result.status == "skipped"]
    errors = [result for result in results if result.status == "error"]
    print(f"Created pairs: {len(created)}")
    print(f"Skipped existing pairs: {len(skipped)}")
    if created:
        print(
            "Visible topmost crown instances: "
            f"{sum(result.visible_trees for result in created)}"
        )
        print(
            "Exclusive labeled canopy cells: "
            f"{sum(result.visible_cells for result in created)}"
        )
        print(
            "Annotated trees not visible above the height threshold: "
            f"{sum(result.occluded_or_below_threshold_trees for result in created)}"
        )
    if errors:
        print(f"Errors: {len(errors)}", file=sys.stderr)
        for result in errors:
            print(f"  - {result.pair.source_stem}: {result.detail}", file=sys.stderr)
    print(f"Elapsed: {(time.monotonic() - start) / 60.0:.2f} min")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
