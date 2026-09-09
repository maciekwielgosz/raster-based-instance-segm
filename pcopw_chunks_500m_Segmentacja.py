#!/usr/bin/env python3
"""Tree-top detection and marker-controlled crown segmentation from CHM tiles.

Python port of ``pcopw_chunks_500m_Segmentacja.R``.  The workflow keeps the
same input/output names and parameters while using Rasterio, SciPy,
GeoPandas, Shapely, and Pyogrio.  The watershed routine directly reproduces
the CImg algorithm called by ``ForestTools::mcws``.

Run from the repository root with:

    .tools/miniforge3/envs/treescan/bin/python \
        run_r/code/pcopw_chunks_500m_Segmentacja.py

Existing paired crown/tree-top outputs are skipped by default.  Use
``--overwrite`` only when replacing those outputs is intentional.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path

# Conda activation normally supplies these paths.  Set them from the active
# Python prefix as well so direct invocation of the environment's interpreter
# has the same GDAL/PROJ behavior as ``conda run``.
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

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from rasterio.features import shapes
from rasterio.transform import xy
from rasterio.windows import from_bounds
from scipy.ndimage import generic_filter, maximum_filter
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from tqdm import tqdm


PROJECT_CRS = "EPSG:2180"
BUFFER_METRES = 25.0
MIN_HEIGHT = 2.0
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent


@dataclass(frozen=True)
class TileResult:
    tile_id: str
    status: str
    trees_count: int = 0
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect tree tops and segment crowns from CHM GeoTIFF tiles."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RUN_DIR / "data_input",
        help="Directory containing chm_*.tif and 00_CHM_Full.vrt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_DIR / "data_output",
        help="Base output directory; files are written below Segmentation3.",
    )
    parser.add_argument(
        "--tile",
        action="append",
        default=[],
        metavar="ID",
        help="Only process this tile ID (for example 764000_197500). Repeatable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing paired GeoPackage outputs.",
    )
    return parser.parse_args()


def dbh_naslund(height: np.ndarray, a: float = 10.0, b: float = 0.6) -> np.ndarray:
    """Estimate diameter at breast height using the formula from the R script."""
    hm = np.maximum(height - 1.3, 0.0)
    coefficient_a = -hm * b
    coefficient_c = -hm * a
    discriminant = coefficient_a**2 - 4.0 * coefficient_c
    return (-coefficient_a + np.sqrt(discriminant)) / 2.0


def moving_window_size(height: np.ndarray | float) -> np.ndarray | float:
    """Variable LMF window diameter, in metres, matching ``f_ws`` in R."""
    return np.where(
        np.asarray(height) < 10.0,
        2.0,
        np.where(np.asarray(height) < 25.0, np.asarray(height) * 0.1 + 0.3,
                 np.asarray(height) * 0.15 + 1.0),
    )


def read_buffered_chm(
    chm_file: Path, vrt_path: Path
) -> tuple[np.ndarray, rasterio.Affine, rasterio.coords.BoundingBox]:
    """Read the tile extent plus the same 25 m VRT buffer used by Terra."""
    with rasterio.open(chm_file) as core_source:
        core_bounds = core_source.bounds

    buffered_bounds = (
        core_bounds.left - BUFFER_METRES,
        core_bounds.bottom - BUFFER_METRES,
        core_bounds.right + BUFFER_METRES,
        core_bounds.top + BUFFER_METRES,
    )

    with rasterio.open(vrt_path) as vrt_source:
        window = from_bounds(*buffered_bounds, transform=vrt_source.transform)
        window = window.round_offsets().round_lengths()
        chm = vrt_source.read(1, window=window, masked=True)
        transform = vrt_source.window_transform(window)

    return chm.filled(np.nan).astype(np.float32), transform, core_bounds


def smooth_chm(chm: np.ndarray) -> np.ndarray:
    """Apply Terra-compatible 3 x 3 median smoothing while ignoring NoData."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return generic_filter(
            chm,
            np.nanmedian,
            size=3,
            output=np.float64,
            mode="constant",
            cval=np.nan,
        )


def circular_footprint(radius: float, x_resolution: float, y_resolution: float) -> np.ndarray:
    row_radius = int(math.ceil(radius / y_resolution))
    col_radius = int(math.ceil(radius / x_resolution))
    rows, cols = np.ogrid[
        -row_radius : row_radius + 1,
        -col_radius : col_radius + 1,
    ]
    squared_distance = (cols * x_resolution) ** 2 + (rows * y_resolution) ** 2
    return squared_distance <= radius**2 + 1e-12


def locate_trees_lmf(
    chm: np.ndarray, transform: rasterio.Affine
) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Reproduce lidR's variable circular local-maximum filter for a raster."""
    valid = np.isfinite(chm)
    eligible = valid & (chm >= MIN_HEIGHT)
    if not np.any(eligible):
        return empty_treetops(), np.zeros(chm.shape, dtype=np.int32)

    x_resolution = abs(float(transform.a))
    y_resolution = abs(float(transform.e))

    # A cheap first pass removes pixels that cannot be maxima.  The smallest
    # possible window from f_ws is used, so this does not discard valid peaks.
    all_windows = moving_window_size(chm[valid])
    minimum_radius = float(np.min(all_windows)) / 2.0
    first_footprint = circular_footprint(
        minimum_radius, x_resolution, y_resolution
    )
    finite_chm = np.where(valid, chm, -np.inf)
    first_maximum = maximum_filter(
        finite_chm,
        footprint=first_footprint,
        mode="constant",
        cval=-np.inf,
    )
    candidates = np.argwhere(eligible & (chm == first_maximum))

    # lidR visits raster cells in row-major order.  Equal-height maxima already
    # accepted inside a candidate's window take precedence.
    accepted = np.zeros(chm.shape, dtype=bool)
    detected: list[tuple[int, int, float]] = []

    for row, col in candidates:
        height = float(chm[row, col])
        radius = float(moving_window_size(height)) / 2.0
        row_radius = int(math.ceil(radius / y_resolution))
        col_radius = int(math.ceil(radius / x_resolution))
        row_start = max(0, row - row_radius)
        row_stop = min(chm.shape[0], row + row_radius + 1)
        col_start = max(0, col - col_radius)
        col_stop = min(chm.shape[1], col + col_radius + 1)

        rows, cols = np.ogrid[
            row_start - row : row_stop - row,
            col_start - col : col_stop - col,
        ]
        inside = (
            (cols * x_resolution) ** 2 + (rows * y_resolution) ** 2
            <= radius**2 + 1e-12
        )
        neighbourhood = chm[row_start:row_stop, col_start:col_stop]
        neighbourhood_values = neighbourhood[inside]

        if np.any(neighbourhood_values > height):
            continue
        accepted_neighbours = accepted[
            row_start:row_stop, col_start:col_stop
        ][inside]
        if np.any(accepted_neighbours & (neighbourhood_values == height)):
            continue

        accepted[row, col] = True
        detected.append((int(row), int(col), height))

    if not detected:
        return empty_treetops(), np.zeros(chm.shape, dtype=np.int32)

    tree_ids = np.arange(1, len(detected) + 1, dtype=np.int32)
    heights = np.asarray([item[2] for item in detected], dtype=np.float64)
    geometries = []
    marker_raster = np.zeros(chm.shape, dtype=np.int32)

    for tree_id, (row, col, height) in zip(tree_ids, detected, strict=True):
        x_coord, y_coord = xy(transform, row, col, offset="center")
        geometries.append(Point(float(x_coord), float(y_coord), height))
        marker_raster[row, col] = tree_id

    treetops = gpd.GeoDataFrame(
        {"treeID": tree_ids, "Z": heights},
        geometry=geometries,
        crs=PROJECT_CRS,
    )
    return treetops, marker_raster


def empty_treetops() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"treeID": np.asarray([], dtype=np.int32), "Z": np.asarray([], dtype=float)},
        geometry=gpd.GeoSeries([], crs=PROJECT_CRS),
        crs=PROJECT_CRS,
    )


def cimg_watershed(marker_raster: np.ndarray, priority: np.ndarray) -> np.ndarray:
    """Port CImg's high-connectivity priority watershed used by ``imager``.

    ``imager::watershed`` passes its third argument (named ``fill_lines`` in
    R) directly to CImg as ``is_high_connectivity``. ForestTools uses the
    default ``TRUE``, which means eight-neighbour connectivity in 2D.
    """
    output = marker_raster.astype(np.int32, copy=True)
    queued_labels = np.zeros(output.shape, dtype=np.uint32)
    seed_locations: list[tuple[int, int]] = []
    priority_queue: list[list[float | int]] = []
    row_count, col_count = output.shape
    neighbours = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    )

    def queue_insert(value: float, row: int, col: int, seed_number: int) -> None:
        if queued_labels[row, col] != 0:
            return
        queued_labels[row, col] = seed_number
        priority_queue.append([float(value), row, col])
        position = len(priority_queue) - 1
        while position:
            parent = (position + 1) // 2 - 1
            if not value > priority_queue[parent][0]:
                break
            priority_queue[position], priority_queue[parent] = (
                priority_queue[parent],
                priority_queue[position],
            )
            position = parent

    def queue_remove_maximum() -> tuple[int, int]:
        root = priority_queue[0]
        last = priority_queue.pop()
        if priority_queue:
            priority_queue[0] = last
            # CImg intentionally narrows this comparison value to float.
            value = float(np.float32(last[0]))
            position = 0
            queue_size = len(priority_queue)
            while True:
                left = 2 * position + 1
                right = left + 1
                if right < queue_size and value < priority_queue[right][0]:
                    swap = (
                        left
                        if priority_queue[left][0] > priority_queue[right][0]
                        else right
                    )
                elif left < queue_size and value < priority_queue[left][0]:
                    swap = left
                else:
                    break
                priority_queue[position], priority_queue[swap] = (
                    priority_queue[swap],
                    priority_queue[position],
                )
                position = swap
        return int(root[1]), int(root[2])

    # An R matrix becomes a CImg whose x axis is the matrix row axis. CImg
    # therefore discovers seeds column-by-column from the raster perspective.
    seed_rows_and_cols = np.argwhere(output.T != 0)
    for col, row in seed_rows_and_cols:
        row = int(row)
        col = int(col)
        seed_locations.append((row, col))
        seed_number = len(seed_locations)
        for row_delta, col_delta in neighbours:
            neighbour_row = row + row_delta
            neighbour_col = col + col_delta
            if (
                0 <= neighbour_row < row_count
                and 0 <= neighbour_col < col_count
                and output[neighbour_row, neighbour_col] == 0
            ):
                queue_insert(
                    priority[neighbour_row, neighbour_col],
                    neighbour_row,
                    neighbour_col,
                    seed_number,
                )
        queued_labels[row, col] = seed_number

    while priority_queue:
        row, col = queue_remove_maximum()
        inherited_seed_number = int(queued_labels[row, col])
        nearest_distance = math.inf
        nearest_seed_index = 0
        nearest_label = 0

        for row_delta, col_delta in neighbours:
            neighbour_row = row + row_delta
            neighbour_col = col + col_delta
            if not (
                0 <= neighbour_row < row_count
                and 0 <= neighbour_col < col_count
            ):
                continue

            if output[neighbour_row, neighbour_col] != 0:
                seed_index = int(queued_labels[neighbour_row, neighbour_col]) - 1
                seed_row, seed_col = seed_locations[seed_index]
                distance = float(
                    (row - seed_row) ** 2 + (col - seed_col) ** 2
                )
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_seed_index = seed_index
                    nearest_label = int(output[seed_row, seed_col])
            else:
                queue_insert(
                    priority[neighbour_row, neighbour_col],
                    neighbour_row,
                    neighbour_col,
                    inherited_seed_number,
                )

        output[row, col] = nearest_label
        queued_labels[row, col] = nearest_seed_index + 1

    return output


def segment_crowns(
    chm: np.ndarray,
    transform: rasterio.Affine,
    marker_raster: np.ndarray,
    core_tree_ids: set[int],
) -> gpd.GeoDataFrame:
    """Perform marker-controlled watershed and polygonize retained crowns."""
    canopy_mask = np.isfinite(chm) & (chm >= MIN_HEIGHT)
    priority = np.where(np.isfinite(chm), chm, 0.0)
    priority[~canopy_mask] = 0.0
    labels = cimg_watershed(marker_raster, priority)
    labels[~canopy_mask] = 0

    if core_tree_ids:
        retained = np.fromiter(sorted(core_tree_ids), dtype=np.int32)
        labels = np.where(np.isin(labels, retained), labels, 0).astype(np.int32)

    pieces: dict[int, list] = {}
    for geometry_mapping, value in shapes(
        labels,
        mask=labels > 0,
        transform=transform,
        connectivity=4,
    ):
        tree_id = int(value)
        pieces.setdefault(tree_id, []).append(shape(geometry_mapping))

    tree_ids: list[int] = []
    polygons = []
    for tree_id in sorted(pieces):
        tree_ids.append(tree_id)
        polygons.append(unary_union(pieces[tree_id]))

    crowns = gpd.GeoDataFrame(
        {"treeID": np.asarray(tree_ids, dtype=np.int32)},
        geometry=polygons,
        crs=PROJECT_CRS,
    )
    crowns["area_m2"] = crowns.geometry.area.astype(float)
    return crowns


def inside_core_mask(
    treetops: gpd.GeoDataFrame, bounds: rasterio.coords.BoundingBox
) -> np.ndarray:
    x_coord = treetops.geometry.x.to_numpy()
    y_coord = treetops.geometry.y.to_numpy()
    return (
        (x_coord >= bounds.left)
        & (x_coord < bounds.right)
        & (y_coord >= bounds.bottom)
        & (y_coord < bounds.top)
    )


def write_outputs_atomically(
    treetops: gpd.GeoDataFrame,
    crowns: gpd.GeoDataFrame,
    treetop_path: Path,
    crown_path: Path,
) -> None:
    """Write both GeoPackages before replacing any existing final output."""
    token = uuid.uuid4().hex
    temporary_treetops = treetop_path.with_name(f".{treetop_path.stem}.{token}.gpkg")
    temporary_crowns = crown_path.with_name(f".{crown_path.stem}.{token}.gpkg")
    try:
        treetops.to_file(
            temporary_treetops,
            layer=treetop_path.stem,
            driver="GPKG",
            engine="pyogrio",
            index=False,
        )
        crowns.to_file(
            temporary_crowns,
            layer=crown_path.stem,
            driver="GPKG",
            engine="pyogrio",
            index=False,
        )
        os.replace(temporary_treetops, treetop_path)
        os.replace(temporary_crowns, crown_path)
    finally:
        temporary_treetops.unlink(missing_ok=True)
        temporary_crowns.unlink(missing_ok=True)


def existing_feature_count(path: Path) -> int:
    try:
        return int(pyogrio.read_info(path)["features"])
    except Exception:
        return 0


def process_tile(
    chm_file: Path,
    vrt_path: Path,
    segmentation_dir: Path,
    overwrite: bool,
) -> TileResult:
    tile_id = chm_file.stem.removeprefix("chm_")
    crown_path = segmentation_dir / f"crowns_{tile_id}.gpkg"
    treetop_path = segmentation_dir / f"ttops_{tile_id}.gpkg"

    crown_exists = crown_path.exists()
    treetop_exists = treetop_path.exists()
    if not overwrite and crown_exists and treetop_exists:
        return TileResult(
            tile_id,
            "skipped",
            existing_feature_count(treetop_path),
            "paired outputs already exist",
        )
    if not overwrite and crown_exists != treetop_exists:
        present = crown_path.name if crown_exists else treetop_path.name
        missing = treetop_path.name if crown_exists else crown_path.name
        return TileResult(
            tile_id,
            "error",
            detail=f"asymmetric outputs: {present} exists but {missing} is missing",
        )

    try:
        chm, transform, core_bounds = read_buffered_chm(chm_file, vrt_path)
        finite = chm[np.isfinite(chm)]
        if finite.size == 0 or float(np.max(finite)) < MIN_HEIGHT:
            return TileResult(tile_id, "empty", detail="no CHM values at least 2 m")

        smoothed = smooth_chm(chm)
        all_treetops, markers = locate_trees_lmf(smoothed, transform)
        if all_treetops.empty:
            return TileResult(tile_id, "empty", detail="no local maxima detected")

        core_mask = inside_core_mask(all_treetops, core_bounds)
        final_treetops = all_treetops.loc[core_mask].copy()
        if final_treetops.empty:
            return TileResult(tile_id, "empty", detail="no tree tops inside core tile")

        final_treetops["dbh"] = np.round(
            dbh_naslund(final_treetops["Z"].to_numpy()), 2
        )
        core_tree_ids = set(final_treetops["treeID"].astype(int))
        final_crowns = segment_crowns(smoothed, transform, markers, core_tree_ids)

        if len(final_crowns) != len(final_treetops):
            raise RuntimeError(
                f"created {len(final_crowns)} crowns for {len(final_treetops)} tree tops"
            )

        write_outputs_atomically(
            final_treetops,
            final_crowns,
            treetop_path,
            crown_path,
        )
        return TileResult(tile_id, "processed", len(final_treetops))
    except Exception as error:
        return TileResult(tile_id, "error", detail=f"{type(error).__name__}: {error}")


def main() -> int:
    args = parse_args()
    start = time.monotonic()
    input_dir = args.input_dir.resolve()
    segmentation_dir = args.output_dir.resolve() / "Segmentation3"
    vrt_path = input_dir / "00_CHM_Full.vrt"
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    chm_files = sorted(input_dir.glob("chm_*.tif"))
    if args.tile:
        requested = set(args.tile)
        chm_files = [
            path for path in chm_files
            if path.stem.removeprefix("chm_") in requested
        ]
        found = {path.stem.removeprefix("chm_") for path in chm_files}
        missing = requested - found
        if missing:
            print(f"Błąd: nie znaleziono kafli: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2

    if not chm_files:
        print(f"Błąd: Brak plików CHM. Sprawdź folder: {input_dir}", file=sys.stderr)
        return 2
    if not vrt_path.is_file():
        print(f"Błąd: Brak pliku VRT: {vrt_path}", file=sys.stderr)
        return 2

    print(">>> C1. Start Segmentacji (Python)...")
    results = [
        process_tile(path, vrt_path, segmentation_dir, args.overwrite)
        for path in tqdm(chm_files, unit="tile", desc="Segmentacja")
    ]

    processed = [result for result in results if result.status == "processed"]
    skipped = [result for result in results if result.status == "skipped"]
    empty = [result for result in results if result.status == "empty"]
    errors = [result for result in results if result.status == "error"]

    print(f"Nowo wykryto drzew: {sum(result.trees_count for result in processed)}")
    if skipped:
        print(
            "Pominięto istniejące wyniki: "
            f"{len(skipped)} kafli, {sum(result.trees_count for result in skipped)} drzew"
        )
    if empty:
        print(f"Puste/pominięte kafle: {len(empty)}")
        for result in empty:
            print(f"  - {result.tile_id}: {result.detail}")
    if errors:
        print(f"Błędy: {len(errors)}", file=sys.stderr)
        for result in errors:
            print(f"  - {result.tile_id}: {result.detail}", file=sys.stderr)

    elapsed_minutes = (time.monotonic() - start) / 60.0
    print(f"Koniec Etapu C. Czas: {elapsed_minutes:.2f} min")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
