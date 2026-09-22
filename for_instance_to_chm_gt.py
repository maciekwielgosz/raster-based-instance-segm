#!/usr/bin/env python3
"""Convert FOR-instance LAS annotations to CHM and crown GT products.

The output layout matches the existing TreeScan-derived input directory:

    chm_<collection>_<source>.tif
    gt_<collection>_<source>.gpkg
    topmost_gt_<collection>_<source>.tif
    topmost_gt_<collection>_<source>.gpkg

Collection names are included because source names such as ``plot_1_annotated``
and ``train`` are not globally unique.  The official ``dev``/``test`` split is
preserved in ``for_instance_file_manifest.csv`` and in output metadata.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# Configure GDAL/PROJ before importing geospatial packages when this script is
# run directly with the project's conda interpreter.
ENVIRONMENT_PREFIX = Path(sys.prefix)
PROJ_DATA = ENVIRONMENT_PREFIX / "share" / "proj"
GDAL_DATA = ENVIRONMENT_PREFIX / "share" / "gdal"
GDAL_PLUGINS = ENVIRONMENT_PREFIX / "lib" / "gdalplugins"
if PROJ_DATA.is_dir():
    os.environ["PROJ_DATA"] = str(PROJ_DATA)
if GDAL_DATA.is_dir():
    os.environ["GDAL_DATA"] = str(GDAL_DATA)
if GDAL_PLUGINS.is_dir():
    os.environ["GDAL_DRIVER_PATH"] = str(GDAL_PLUGINS)

try:
    import geopandas as gpd
    import laspy
    import numpy as np
    import pyogrio
    import rasterio
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    from scipy.ndimage import (
        binary_closing,
        binary_fill_holes,
        distance_transform_edt,
        generate_binary_structure,
        label,
        median_filter,
        minimum_filter,
    )
    from shapely.geometry import MultiPolygon, Polygon, shape
    from shapely.ops import unary_union
except ImportError as error:
    raise SystemExit(
        "Missing dependency in the treescan environment. Required: laspy, "
        "numpy, scipy, rasterio, geopandas, pyogrio and shapely. "
        f"Original error: {error}"
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent
PROJECT_DIR = RUN_DIR.parent

DEFAULT_INPUT_DIR = PROJECT_DIR / "FOR-instance"
DEFAULT_OUTPUT_DIR = RUN_DIR / "data_input_from_laz_for_instance"
SPLIT_METADATA_FILENAME = "data_split_metadata.csv"
OUTPUT_MANIFEST_FILENAME = "for_instance_file_manifest.csv"

PIXEL_SIZE_METRES = 0.5
MIN_CANOPY_HEIGHT_METRES = 2.0
CHUNK_SIZE_POINTS = 2_000_000
CHM_NODATA = -9999.0
LABEL_NODATA = -1
TREE_CLASSES = (4, 5, 6)
OUTSIDE_CLASS = 3
GROUND_CLASS = 2
MASK_CONNECTIVITY = 8
POLYGON_CONNECTIVITY = 4
MORPHOLOGICAL_CLOSING_ITERATIONS = 1
MIN_CROWN_AREA_M2 = 0.25

GT_LAYER_NAME = "crowns_gt"
TOPMOST_LAYER_NAME = "crowns_gt_topmost"

# These two LAS collections omit CRS VLRs. Their coordinates are already in
# metre-based UTM/MGA grids, so only the missing CRS metadata is supplied.
# TUWIEN: WGS 84 / UTM 33N. RMIT: GDA94 / MGA zone 55 (Tasmania).
FALLBACK_COLLECTION_CRS = {
    "RMIT": "EPSG:28355",
    "TUWIEN": "EPSG:32633",
    # IDEAS-ALS collections distributed in local metre coordinates.  Regional
    # projected CRSs are attached only to satisfy the raster/vector metre-CRS
    # contract; no coordinate transformation is performed.
    "CEDAR_CYPRESS": "EPSG:32652",
    "FGI_EMIT": "EPSG:3067",
    "SEPILOK": "EPSG:32650",
    "WYTHAM": "EPSG:27700",
}

# Defaults preserve the original FOR-instance conversion exactly.  The
# IDEAS-ALS wrapper selects the alternative modes explicitly on the CLI.
TREE_POINT_MODE = "semantic"
CHM_POINT_MODE = "annotated-tree"
DTM_MODE = "class2"
DATASET_NAME = "FOR-instance"


@dataclass(frozen=True)
class SourcePlot:
    source_path: Path
    relative_path: str
    collection: str
    split: str
    dataset_id: str


@dataclass(frozen=True)
class Grid:
    left: float
    bottom: float
    right: float
    top: float
    width: int
    height: int
    transform: rasterio.Affine

    @property
    def cell_count(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class TreeMetadata:
    tree_id: int
    dbh_cm: float


@dataclass(frozen=True)
class ConversionResult:
    source: SourcePlot
    status: str
    point_count: int
    tree_count: int
    visible_tree_count: int
    crs: str
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CHM and crown ground truth from FOR-instance LAS files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="ID",
        help="Process one dataset ID, relative LAS path, filename or stem; repeatable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--split",
        action="append",
        choices=("dev", "test"),
        default=[],
        help="Process only this manifest split; repeatable.",
    )
    parser.add_argument(
        "--tree-point-mode",
        choices=("semantic", "instance"),
        default=TREE_POINT_MODE,
        help="semantic uses classes 4-6; instance uses every treeID>0 except ground/outside.",
    )
    parser.add_argument(
        "--chm-point-mode",
        choices=("annotated-tree", "all-nonground"),
        default=CHM_POINT_MODE,
        help="Points used to build the CHM surface.",
    )
    parser.add_argument(
        "--dtm-mode",
        choices=("class2", "auto", "zero", "cell-min"),
        default=DTM_MODE,
        help="Terrain normalization method; auto prefers class 2, then normalized Z or a lower envelope.",
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--manifest-name", default=OUTPUT_MANIFEST_FILENAME)
    return parser.parse_args()


def safe_dataset_id(collection: str, stem: str) -> str:
    value = f"{collection}_{stem}"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def read_sources(input_dir: Path) -> list[SourcePlot]:
    metadata_path = input_dir / SPLIT_METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing split metadata: {metadata_path}")

    sources: list[SourcePlot] = []
    missing_from_archive: list[str] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "folder", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Split metadata is missing columns: " + ", ".join(sorted(missing))
            )
        for row in reader:
            relative_path = row["path"].strip().replace("\\", "/")
            collection = row["folder"].strip()
            split = row["split"].strip().lower()
            source_path = input_dir / relative_path
            if split not in {"dev", "test"}:
                raise ValueError(f"Unexpected split {split!r} for {relative_path}")
            if not source_path.is_file():
                missing_from_archive.append(relative_path)
                continue
            sources.append(
                SourcePlot(
                    source_path=source_path,
                    relative_path=relative_path,
                    collection=collection,
                    split=split,
                    dataset_id=safe_dataset_id(collection, source_path.stem),
                )
            )

    ids = [source.dataset_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("Generated dataset IDs are not unique")
    if missing_from_archive:
        collections = sorted({path.split("/", 1)[0] for path in missing_from_archive})
        print(
            "Warning: split metadata lists "
            f"{len(missing_from_archive)} LAS files absent from the archive "
            f"(collections: {', '.join(collections)}); processing available files only.",
            file=sys.stderr,
        )
    return sources


def select_sources(sources: list[SourcePlot], requested: list[str]) -> list[SourcePlot]:
    if not requested:
        return sources
    selected: list[SourcePlot] = []
    missing: list[str] = []
    for value in requested:
        normalized = value.replace("\\", "/")
        matches = [
            source
            for source in sources
            if normalized
            in {
                source.dataset_id,
                source.relative_path,
                source.source_path.name,
                source.source_path.stem,
            }
        ]
        if len(matches) == 1:
            if matches[0] not in selected:
                selected.append(matches[0])
        elif len(matches) > 1:
            raise ValueError(
                f"Ambiguous --file {value!r}; use the collection-prefixed dataset ID"
            )
        else:
            missing.append(value)
    if missing:
        raise FileNotFoundError("Requested sources not found: " + ", ".join(missing))
    return selected


def output_paths(output_dir: Path, dataset_id: str) -> dict[str, Path]:
    return {
        "chm": output_dir / f"chm_{dataset_id}.tif",
        "gt": output_dir / f"gt_{dataset_id}.gpkg",
        "top_raster": output_dir / f"topmost_gt_{dataset_id}.tif",
        "top_vector": output_dir / f"topmost_gt_{dataset_id}.gpkg",
    }


def source_crs(source: SourcePlot, header: laspy.LasHeader):
    crs = header.parse_crs()
    used_fallback = False
    if crs is None:
        fallback = FALLBACK_COLLECTION_CRS.get(source.collection)
        if fallback is None:
            raise ValueError(
                f"{source.relative_path} has no CRS and no collection fallback"
            )
        crs = rasterio.crs.CRS.from_string(fallback)
        used_fallback = True
    else:
        crs = rasterio.crs.CRS.from_user_input(crs)
    if crs.is_geographic:
        raise ValueError(f"Expected a projected metre CRS, found {crs}")
    return crs, used_fallback


def make_grid(header: laspy.LasHeader) -> Grid:
    min_x, min_y = (float(header.mins[0]), float(header.mins[1]))
    max_x, max_y = (float(header.maxs[0]), float(header.maxs[1]))
    left = math.floor(min_x / PIXEL_SIZE_METRES) * PIXEL_SIZE_METRES
    bottom = math.floor(min_y / PIXEL_SIZE_METRES) * PIXEL_SIZE_METRES
    right = math.ceil(max_x / PIXEL_SIZE_METRES) * PIXEL_SIZE_METRES
    top = math.ceil(max_y / PIXEL_SIZE_METRES) * PIXEL_SIZE_METRES
    width = int(round((right - left) / PIXEL_SIZE_METRES))
    height = int(round((top - bottom) / PIXEL_SIZE_METRES))
    if width <= 0 or height <= 0:
        raise ValueError("LAS bounds produce an empty raster")
    return Grid(
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        width=width,
        height=height,
        transform=from_origin(left, top, PIXEL_SIZE_METRES, PIXEL_SIZE_METRES),
    )


def point_cells(x: np.ndarray, y: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    columns = np.floor((x - grid.left) / PIXEL_SIZE_METRES).astype(np.int64)
    rows = np.floor((grid.top - y) / PIXEL_SIZE_METRES).astype(np.int64)
    columns = np.clip(columns, 0, grid.width - 1)
    rows = np.clip(rows, 0, grid.height - 1)
    return rows, columns


def integer_tree_ids(points) -> np.ndarray:
    raw = np.asarray(points["treeID"])
    if np.issubdtype(raw.dtype, np.floating):
        finite = np.isfinite(raw)
        rounded = np.zeros(raw.shape, dtype=np.int64)
        rounded[finite] = np.rint(raw[finite]).astype(np.int64)
        if np.any(np.abs(raw[finite] - rounded[finite]) > 1e-6):
            raise ValueError("treeID contains non-integer floating-point values")
        return rounded
    return raw.astype(np.int64, copy=False)


def annotated_tree_points(tree_ids: np.ndarray, classification: np.ndarray) -> np.ndarray:
    if TREE_POINT_MODE == "semantic":
        return (tree_ids > 0) & np.isin(classification, TREE_CLASSES)
    if TREE_POINT_MODE == "instance":
        return (tree_ids > 0) & ~np.isin(
            classification, (GROUND_CLASS, OUTSIDE_CLASS)
        )
    raise ValueError(f"Unexpected tree point mode: {TREE_POINT_MODE}")


def first_pass(
    source: SourcePlot,
    grid: Grid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int]]:
    coverage_count = np.zeros(grid.cell_count, dtype=np.uint32)
    ground_sum = np.zeros(grid.cell_count, dtype=np.float64)
    ground_count = np.zeros(grid.cell_count, dtype=np.uint32)
    present_tree_ids: set[int] = set()

    with laspy.open(source.source_path) as reader:
        dimensions = set(reader.header.point_format.dimension_names)
        if "treeID" not in dimensions:
            raise ValueError(f"{source.relative_path} has no treeID dimension")
        for points in reader.chunk_iterator(CHUNK_SIZE_POINTS):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            classification = np.asarray(points.classification, dtype=np.uint8)
            tree_ids = integer_tree_ids(points)
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if not np.any(finite):
                continue
            rows, columns = point_cells(x[finite], y[finite], grid)
            cells = rows * grid.width + columns
            classes = classification[finite]
            z_values = z[finite]
            ids = tree_ids[finite]

            inside_annotation = classes != OUTSIDE_CLASS
            np.add.at(coverage_count, cells[inside_annotation], 1)

            ground = classes == GROUND_CLASS
            np.add.at(ground_sum, cells[ground], z_values[ground])
            np.add.at(ground_count, cells[ground], 1)

            annotated_tree = annotated_tree_points(ids, classes)
            present_tree_ids.update(int(value) for value in np.unique(ids[annotated_tree]))

    return coverage_count, ground_sum, ground_count, present_tree_ids


def class2_dtm(
    ground_sum: np.ndarray, ground_count: np.ndarray, grid: Grid
) -> np.ndarray:
    known = ground_count > 0
    if not np.any(known):
        raise ValueError("No class-2 terrain points available for height normalization")
    dtm = np.full(grid.cell_count, np.nan, dtype=np.float64)
    dtm[known] = ground_sum[known] / ground_count[known]
    dtm = dtm.reshape((grid.height, grid.width))
    known_2d = known.reshape((grid.height, grid.width))
    nearest_indices = distance_transform_edt(
        ~known_2d,
        return_distances=False,
        return_indices=True,
    )
    return dtm[tuple(nearest_indices)]


def lower_envelope_dtm(source: SourcePlot, grid: Grid) -> np.ndarray:
    """Approximate terrain from the spatial lower envelope when ground is unlabelled."""
    surface_min = np.full(grid.cell_count, np.inf, dtype=np.float64)
    with laspy.open(source.source_path) as reader:
        for points in reader.chunk_iterator(CHUNK_SIZE_POINTS):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            classification = np.asarray(points.classification, dtype=np.uint8)
            selected = (
                np.isfinite(x)
                & np.isfinite(y)
                & np.isfinite(z)
                & (classification != OUTSIDE_CLASS)
            )
            if not np.any(selected):
                continue
            rows, columns = point_cells(x[selected], y[selected], grid)
            cells = rows * grid.width + columns
            np.minimum.at(surface_min, cells, z[selected])
    known = np.isfinite(surface_min)
    if not np.any(known):
        raise ValueError("No finite points available for lower-envelope DTM")
    surface = surface_min.reshape((grid.height, grid.width))
    known_2d = known.reshape((grid.height, grid.width))
    nearest_indices = distance_transform_edt(
        ~known_2d, return_distances=False, return_indices=True
    )
    surface = surface[tuple(nearest_indices)]
    # A 2.5 m erosion followed by a 4.5 m median removes isolated low returns
    # while retaining broad terrain slope on the small benchmark plots.
    return median_filter(minimum_filter(surface, size=5), size=9)


def build_dtm(
    source: SourcePlot,
    ground_sum: np.ndarray,
    ground_count: np.ndarray,
    grid: Grid,
) -> tuple[np.ndarray, str]:
    if DTM_MODE == "class2":
        return class2_dtm(ground_sum, ground_count, grid), "class2-nearest"
    if DTM_MODE == "zero":
        return np.zeros((grid.height, grid.width), dtype=np.float64), "zero-normalized"
    if DTM_MODE == "cell-min":
        return lower_envelope_dtm(source, grid), "cell-min-lower-envelope"
    if DTM_MODE != "auto":
        raise ValueError(f"Unexpected DTM mode: {DTM_MODE}")
    if np.any(ground_count > 0):
        return class2_dtm(ground_sum, ground_count, grid), "class2-nearest"
    with laspy.open(source.source_path) as reader:
        minimum_z = float(reader.header.mins[2])
    if minimum_z <= 2.0:
        return np.zeros((grid.height, grid.width), dtype=np.float64), "zero-normalized"
    return lower_envelope_dtm(source, grid), "cell-min-lower-envelope"


def plot_id_from_source(source: SourcePlot) -> str | None:
    match = re.fullmatch(r"plot_(.+)_annotated", source.source_path.stem)
    return match.group(1) if match else None


def read_tree_metadata(
    source: SourcePlot,
    input_dir: Path,
    present_tree_ids: set[int],
) -> dict[int, TreeMetadata]:
    metadata_path = input_dir / source.collection / f"tree_data_{source.collection}.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing tree metadata: {metadata_path}")
    plot_id = plot_id_from_source(source)
    records: dict[int, TreeMetadata] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if not {"treeID", "DBH"}.issubset(fields):
            raise ValueError(f"Unexpected columns in {metadata_path}")
        for row in reader:
            if "plotID" in fields and plot_id is not None:
                if row["plotID"].strip() != plot_id:
                    continue
            tree_id = int(float(row["treeID"]))
            if tree_id not in present_tree_ids:
                continue
            raw_dbh = row["DBH"].strip()
            dbh_cm = float(raw_dbh) if raw_dbh else math.nan
            if math.isfinite(dbh_cm) and dbh_cm <= 0:
                raise ValueError(f"Invalid DBH for tree {tree_id} in {metadata_path}")
            if tree_id in records:
                raise ValueError(f"Duplicate treeID {tree_id} in {metadata_path}")
            records[tree_id] = TreeMetadata(tree_id=tree_id, dbh_cm=dbh_cm)
    missing = sorted(present_tree_ids - set(records))
    if missing:
        for tree_id in missing:
            records[tree_id] = TreeMetadata(tree_id=tree_id, dbh_cm=math.nan)
        print(
            f"  note: {len(missing)} annotated trees have no field DBH; "
            "retaining them with DBH_available=0",
            flush=True,
        )
    return records


def second_pass(
    source: SourcePlot,
    grid: Grid,
    dtm: np.ndarray,
    tree_metadata: dict[int, TreeMetadata],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sorted_tree_ids = np.asarray(sorted(tree_metadata), dtype=np.int64)
    raw_masks = np.zeros((len(sorted_tree_ids), grid.cell_count), dtype=bool)
    point_counts = np.zeros(len(sorted_tree_ids), dtype=np.int64)
    annotated_best_height = np.full(grid.cell_count, -np.inf, dtype=np.float64)
    chm_best_height = np.full(grid.cell_count, -np.inf, dtype=np.float64)
    winner_tree_id = np.zeros(grid.cell_count, dtype=np.int64)
    flat_dtm = dtm.ravel()

    with laspy.open(source.source_path) as reader:
        for points in reader.chunk_iterator(CHUNK_SIZE_POINTS):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            classification = np.asarray(points.classification, dtype=np.uint8)
            tree_ids = integer_tree_ids(points)
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

            if CHM_POINT_MODE == "all-nonground":
                chm_selected = finite & ~np.isin(
                    classification, (GROUND_CLASS, OUTSIDE_CLASS)
                )
                if np.any(chm_selected):
                    chm_rows, chm_columns = point_cells(
                        x[chm_selected], y[chm_selected], grid
                    )
                    chm_cells = chm_rows * grid.width + chm_columns
                    chm_heights = z[chm_selected] - flat_dtm[chm_cells]
                    valid_chm = np.isfinite(chm_heights) & (chm_heights >= 0)
                    np.maximum.at(
                        chm_best_height,
                        chm_cells[valid_chm],
                        chm_heights[valid_chm],
                    )
            elif CHM_POINT_MODE != "annotated-tree":
                raise ValueError(f"Unexpected CHM point mode: {CHM_POINT_MODE}")

            selected = finite & annotated_tree_points(tree_ids, classification)
            if not np.any(selected):
                continue
            x = x[selected]
            y = y[selected]
            z = z[selected]
            ids = tree_ids[selected]
            rows, columns = point_cells(x, y, grid)
            cells = rows * grid.width + columns
            heights = z - flat_dtm[cells]
            valid_height = np.isfinite(heights) & (heights >= 0)
            if not np.any(valid_height):
                continue
            cells = cells[valid_height]
            ids = ids[valid_height]
            heights = heights[valid_height]

            compact_ids = np.searchsorted(sorted_tree_ids, ids)
            valid_id = compact_ids < len(sorted_tree_ids)
            valid_id &= sorted_tree_ids[np.minimum(compact_ids, len(sorted_tree_ids) - 1)] == ids
            if not np.all(valid_id):
                unknown = np.unique(ids[~valid_id])
                raise ValueError(f"Tree IDs disappeared from metadata: {unknown[:20]}")
            np.add.at(point_counts, compact_ids, 1)
            raw_masks.ravel()[compact_ids * grid.cell_count + cells] = True

            # One candidate per raster cell in this chunk: highest point wins;
            # exact ties are resolved by the lower canonical tree ID.
            order = np.lexsort((ids, -heights, cells))
            ordered_cells = cells[order]
            first = np.empty(ordered_cells.size, dtype=bool)
            first[0] = True
            first[1:] = ordered_cells[1:] != ordered_cells[:-1]
            chosen = order[first]
            candidate_cells = cells[chosen]
            candidate_heights = heights[chosen]
            candidate_ids = ids[chosen]
            current_heights = annotated_best_height[candidate_cells]
            current_ids = winner_tree_id[candidate_cells]
            better = candidate_heights > current_heights + 1e-9
            tied = np.abs(candidate_heights - current_heights) <= 1e-9
            better |= tied & ((current_ids == 0) | (candidate_ids < current_ids))
            update_cells = candidate_cells[better]
            annotated_best_height[update_cells] = candidate_heights[better]
            winner_tree_id[update_cells] = candidate_ids[better]

    if CHM_POINT_MODE == "annotated-tree":
        chm_best_height = annotated_best_height.copy()
    return (
        sorted_tree_ids,
        raw_masks,
        point_counts,
        np.vstack((annotated_best_height, winner_tree_id)),
        chm_best_height,
    )


def largest_component(mask: np.ndarray) -> np.ndarray:
    structure = generate_binary_structure(2, 2)
    labeled, component_count = label(mask, structure=structure)
    if component_count <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def clean_crown_mask(mask: np.ndarray) -> np.ndarray:
    structure = generate_binary_structure(2, 2)
    result = largest_component(mask)
    if MORPHOLOGICAL_CLOSING_ITERATIONS > 0:
        result = binary_closing(
            result,
            structure=structure,
            iterations=MORPHOLOGICAL_CLOSING_ITERATIONS,
            border_value=1,
        )
    result = binary_fill_holes(result)
    return np.asarray(largest_component(result), dtype=bool)


def mask_geometry(mask: np.ndarray, transform) -> MultiPolygon:
    pieces = [
        shape(mapping)
        for mapping, value in shapes(
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
    if isinstance(merged, Polygon):
        polygons = [merged]
    elif isinstance(merged, MultiPolygon):
        polygons = list(merged.geoms)
    else:
        polygons = [item for item in getattr(merged, "geoms", []) if isinstance(item, Polygon)]
    return MultiPolygon(polygons)


def build_overlapping_gt(
    source: SourcePlot,
    crs,
    grid: Grid,
    tree_ids: np.ndarray,
    raw_masks: np.ndarray,
    point_counts: np.ndarray,
    tree_metadata: dict[int, TreeMetadata],
) -> gpd.GeoDataFrame:
    rows: list[dict] = []
    geometries: list[MultiPolygon] = []
    pixel_area = PIXEL_SIZE_METRES**2
    for compact_id, tree_id_value in enumerate(tree_ids):
        tree_id = int(tree_id_value)
        raw_mask = raw_masks[compact_id].reshape((grid.height, grid.width))
        clean_mask = clean_crown_mask(raw_mask)
        if np.count_nonzero(clean_mask) * pixel_area < MIN_CROWN_AREA_M2:
            continue
        geometry = mask_geometry(clean_mask, grid.transform)
        if geometry.is_empty or not geometry.is_valid:
            continue
        rows.append(
            {
                "treeID": tree_id,
                "DBH_cm": tree_metadata[tree_id].dbh_cm,
                "DBH_available": int(math.isfinite(tree_metadata[tree_id].dbh_cm)),
                "collection": source.collection,
                "split": source.split,
                "evaluation_eligible": 1,
                "point_count_used": int(point_counts[compact_id]),
                "raw_cells": int(np.count_nonzero(raw_mask)),
                "clean_cells": int(np.count_nonzero(clean_mask)),
                "gt_area_m2": float(geometry.area),
                "source_las": source.relative_path,
                "gt_method": "all_annotated_tree_cells",
            }
        )
        geometries.append(geometry)
    if not rows:
        raise ValueError("No overlapping crown polygons were created")
    return gpd.GeoDataFrame(rows, geometry=geometries, crs=crs).sort_values(
        "treeID"
    ).reset_index(drop=True)


def build_topmost_gt(
    source: SourcePlot,
    crs,
    grid: Grid,
    labels: np.ndarray,
    heights: np.ndarray,
    tree_metadata: dict[int, TreeMetadata],
) -> gpd.GeoDataFrame:
    pieces: dict[int, list[Polygon]] = {}
    for mapping, value in shapes(
        labels,
        mask=labels > 0,
        transform=grid.transform,
        connectivity=POLYGON_CONNECTIVITY,
    ):
        geometry = shape(mapping)
        if isinstance(geometry, Polygon):
            pieces.setdefault(int(value), []).append(geometry)
    rows: list[dict] = []
    geometries: list[MultiPolygon] = []
    for tree_id in sorted(pieces):
        merged = unary_union(pieces[tree_id])
        if isinstance(merged, Polygon):
            polygons = [merged]
        elif isinstance(merged, MultiPolygon):
            polygons = list(merged.geoms)
        else:
            polygons = [item for item in getattr(merged, "geoms", []) if isinstance(item, Polygon)]
        geometry = MultiPolygon(polygons)
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"Invalid topmost geometry for tree {tree_id}")
        tree_cells = labels == tree_id
        rows.append(
            {
                "treeID": tree_id,
                "DBH_cm": tree_metadata[tree_id].dbh_cm,
                "DBH_available": int(math.isfinite(tree_metadata[tree_id].dbh_cm)),
                "collection": source.collection,
                "split": source.split,
                "evaluation_eligible": 1,
                "visible_cells": int(np.count_nonzero(tree_cells)),
                "topmost_max_height_m": float(np.max(heights[tree_cells])),
                "gt_area_m2": float(geometry.area),
                "source_las": source.relative_path,
                "gt_method": "topmost_chm_cell",
            }
        )
        geometries.append(geometry)
    if not rows:
        raise ValueError("No topmost crown polygons were created")
    return gpd.GeoDataFrame(rows, geometry=geometries, crs=crs).sort_values(
        "treeID"
    ).reset_index(drop=True)


def raster_profile(grid: Grid, crs, dtype: str, nodata) -> dict:
    return {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 3 if dtype.startswith("float") else 2,
    }


def write_outputs(
    paths: dict[str, Path],
    source: SourcePlot,
    crs,
    grid: Grid,
    chm: np.ndarray,
    labels: np.ndarray,
    overlapping_gt: gpd.GeoDataFrame,
    topmost_gt: gpd.GeoDataFrame,
    dtm_method: str,
) -> None:
    token = uuid.uuid4().hex
    temporary = {
        key: path.with_name(f".{path.stem}.{token}{path.suffix}")
        for key, path in paths.items()
    }
    try:
        with rasterio.open(
            temporary["chm"], "w", **raster_profile(grid, crs, "float32", CHM_NODATA)
        ) as destination:
            destination.write(chm.astype(np.float32), 1)
            destination.set_band_description(1, f"canopy height; DTM={dtm_method}")
            destination.update_tags(
                DATASET=DATASET_NAME,
                COLLECTION=source.collection,
                SPLIT=source.split,
                SOURCE_LAS=source.relative_path,
                PIXEL_SIZE_METRES=str(PIXEL_SIZE_METRES),
                TREE_POINT_MODE=TREE_POINT_MODE,
                CHM_POINT_MODE=CHM_POINT_MODE,
                DTM_METHOD=dtm_method,
                EXCLUDED_OUTSIDE_CLASS=str(OUTSIDE_CLASS),
            )
        with rasterio.open(
            temporary["top_raster"],
            "w",
            **raster_profile(grid, crs, "int32", LABEL_NODATA),
        ) as destination:
            destination.write(labels.astype(np.int32), 1)
            destination.set_band_description(1, "topmost crown instance ID")
            destination.update_tags(
                DATASET=DATASET_NAME,
                COLLECTION=source.collection,
                SPLIT=source.split,
                SOURCE_LAS=source.relative_path,
                MIN_CANOPY_HEIGHT_METRES=str(MIN_CANOPY_HEIGHT_METRES),
                TREE_ID_SOURCE="treeID",
            )
        overlapping_gt.to_file(
            temporary["gt"],
            layer=GT_LAYER_NAME,
            driver="GPKG",
            engine="pyogrio",
            index=False,
        )
        topmost_gt.to_file(
            temporary["top_vector"],
            layer=TOPMOST_LAYER_NAME,
            driver="GPKG",
            engine="pyogrio",
            index=False,
        )

        with rasterio.open(temporary["chm"]) as check:
            if check.shape != chm.shape or check.crs != crs:
                raise RuntimeError("CHM verification failed")
            if not np.array_equal(check.read(1), chm.astype(np.float32)):
                raise RuntimeError("CHM values changed during writing")
        with rasterio.open(temporary["top_raster"]) as check:
            if check.shape != labels.shape or check.crs != crs:
                raise RuntimeError("Topmost raster verification failed")
            if not np.array_equal(check.read(1), labels.astype(np.int32)):
                raise RuntimeError("Topmost labels changed during writing")
        if int(pyogrio.read_info(temporary["gt"], layer=GT_LAYER_NAME)["features"]) != len(overlapping_gt):
            raise RuntimeError("Overlapping GT feature-count verification failed")
        if int(pyogrio.read_info(temporary["top_vector"], layer=TOPMOST_LAYER_NAME)["features"]) != len(topmost_gt):
            raise RuntimeError("Topmost GT feature-count verification failed")

        for key, final_path in paths.items():
            os.replace(temporary[key], final_path)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def convert_source(
    source: SourcePlot,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> ConversionResult:
    started = time.monotonic()
    paths = output_paths(output_dir, source.dataset_id)
    existing = [path for path in paths.values() if path.exists()]
    if existing and len(existing) != len(paths) and not overwrite:
        raise FileExistsError(
            f"Partial outputs exist for {source.dataset_id}; use --overwrite after inspection"
        )
    if len(existing) == len(paths) and not overwrite:
        with laspy.open(source.source_path) as reader:
            crs, _ = source_crs(source, reader.header)
            point_count = int(reader.header.point_count)
        gt_info = pyogrio.read_info(paths["gt"], layer=GT_LAYER_NAME)
        top_info = pyogrio.read_info(paths["top_vector"], layer=TOPMOST_LAYER_NAME)
        return ConversionResult(
            source=source,
            status="skipped",
            point_count=point_count,
            tree_count=int(gt_info["features"]),
            visible_tree_count=int(top_info["features"]),
            crs=crs.to_string(),
            elapsed_seconds=time.monotonic() - started,
        )

    with laspy.open(source.source_path) as reader:
        header = reader.header
        point_count = int(header.point_count)
        crs, used_fallback = source_crs(source, header)
        grid = make_grid(header)

    coverage_count, ground_sum, ground_count, present_ids = first_pass(source, grid)
    tree_metadata = read_tree_metadata(source, input_dir, present_ids)
    dtm, dtm_method = build_dtm(source, ground_sum, ground_count, grid)
    tree_ids, raw_masks, point_counts, winner_data, chm_best_height = second_pass(
        source, grid, dtm, tree_metadata
    )
    annotated_best_height = winner_data[0]
    winner_tree_id = winner_data[1].astype(np.int64)
    coverage = coverage_count > 0

    chm_flat = np.full(grid.cell_count, CHM_NODATA, dtype=np.float32)
    chm_flat[coverage] = 0.0
    canopy = np.isfinite(chm_best_height) & coverage
    chm_flat[canopy] = chm_best_height[canopy].astype(np.float32)
    chm = chm_flat.reshape((grid.height, grid.width))

    labels_flat = np.full(grid.cell_count, LABEL_NODATA, dtype=np.int32)
    labels_flat[coverage] = 0
    visible = (
        np.isfinite(annotated_best_height)
        & coverage
        & (annotated_best_height >= MIN_CANOPY_HEIGHT_METRES)
    )
    labels_flat[visible] = winner_tree_id[visible].astype(np.int32)
    labels = labels_flat.reshape((grid.height, grid.width))
    heights = annotated_best_height.reshape((grid.height, grid.width))

    overlapping_gt = build_overlapping_gt(
        source, crs, grid, tree_ids, raw_masks, point_counts, tree_metadata
    )
    topmost_gt = build_topmost_gt(
        source, crs, grid, labels, heights, tree_metadata
    )
    write_outputs(
        paths,
        source,
        crs,
        grid,
        chm,
        labels,
        overlapping_gt,
        topmost_gt,
        dtm_method,
    )
    fallback_note = " (fallback CRS)" if used_fallback else ""
    print(
        f"created {source.dataset_id}: {len(overlapping_gt)} GT trees, "
        f"{len(topmost_gt)} topmost trees, DTM={dtm_method}, {crs}{fallback_note}",
        flush=True,
    )
    return ConversionResult(
        source=source,
        status="created",
        point_count=point_count,
        tree_count=len(overlapping_gt),
        visible_tree_count=len(topmost_gt),
        crs=crs.to_string(),
        elapsed_seconds=time.monotonic() - started,
    )


def write_manifest(
    output_dir: Path, all_sources: list[SourcePlot], manifest_name: str
) -> None:
    path = output_dir / manifest_name
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fields = [
        "dataset_id",
        "collection",
        "split",
        "source_las",
        "chm_file",
        "gt_file",
        "topmost_gt_raster",
        "topmost_gt_vector",
    ]
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for source in all_sources:
                paths = output_paths(output_dir, source.dataset_id)
                writer.writerow(
                    {
                        "dataset_id": source.dataset_id,
                        "collection": source.collection,
                        "split": source.split,
                        "source_las": source.relative_path,
                        "chm_file": paths["chm"].name,
                        "gt_file": paths["gt"].name,
                        "topmost_gt_raster": paths["top_raster"].name,
                        "topmost_gt_vector": paths["top_vector"].name,
                    }
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    global TREE_POINT_MODE, CHM_POINT_MODE, DTM_MODE, DATASET_NAME
    args = parse_args()
    TREE_POINT_MODE = args.tree_point_mode
    CHM_POINT_MODE = args.chm_point_mode
    DTM_MODE = args.dtm_mode
    DATASET_NAME = args.dataset_name
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    try:
        all_sources = read_sources(input_dir)
        if args.split:
            requested_splits = set(args.split)
            all_sources = [
                source for source in all_sources if source.split in requested_splits
            ]
        selected = select_sources(all_sources, args.file)
    except (FileNotFoundError, ValueError) as error:
        print(f"Input validation failed: {error}", file=sys.stderr)
        return 2

    print(f"{DATASET_NAME} input: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Selected LAS files: {len(selected)} of {len(all_sources)}")
    print(f"Pixel size: {PIXEL_SIZE_METRES} m")
    print(
        f"Modes: tree={TREE_POINT_MODE}, CHM={CHM_POINT_MODE}, DTM={DTM_MODE}"
    )
    if args.dry_run:
        for source in selected:
            print(
                f"{source.relative_path} [{source.split}] -> "
                f"chm_{source.dataset_id}.tif + GT products"
            )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ConversionResult] = []
    try:
        for index, source in enumerate(selected, start=1):
            print(
                f"[{index}/{len(selected)}] {source.relative_path} [{source.split}]",
                flush=True,
            )
            results.append(
                convert_source(source, input_dir, output_dir, args.overwrite)
            )
        write_manifest(output_dir, selected, args.manifest_name)
    except Exception as error:
        print(f"Conversion failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    created = sum(result.status == "created" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    elapsed = sum(result.elapsed_seconds for result in results)
    print(
        f"Done: created={created}, skipped={skipped}, "
        f"plots={len(results)}, processing_time={elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
