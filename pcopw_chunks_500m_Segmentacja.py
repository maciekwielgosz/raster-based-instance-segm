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

For standalone rasters that are not parts of a continuous VRT mosaic, use
``--independent-tiles``. Each raster is then padded with NoData for the
configured buffer instead of reading neighboring pixels from a VRT.
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
from scipy.ndimage import convolve, gaussian_filter, generic_filter, maximum_filter
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent


# =============================================================================
# USER-TUNABLE CONFIGURATION
# =============================================================================
# The defaults below reproduce the original R workflow. Keep them unchanged
# when comparing Python results with pcopw_chunks_500m_Segmentacja.R.

# Input and output layout
DEFAULT_INPUT_DIR = RUN_DIR / "data_input"
DEFAULT_OUTPUT_DIR = RUN_DIR / "data_output"
SEGMENTATION_SUBDIRECTORY = "Segmentation3"
CHM_TILE_GLOB = "chm_*.tif"
CHM_TILE_PREFIX = "chm_"
CHM_BAND_INDEX = 1  # Raster bands are numbered starting at 1.
VRT_FILENAME = "00_CHM_Full.vrt"
TREETOP_OUTPUT_PREFIX = "ttops_"
CROWN_OUTPUT_PREFIX = "crowns_"
OUTPUT_EXTENSION = ".gpkg"
OUTPUT_DRIVER = "GPKG"
OUTPUT_ENGINE = "pyogrio"
OVERWRITE_EXISTING_OUTPUTS = False

# Spatial and canopy-height settings. Output features inherit the projected,
# metre-based CRS of each input CHM. This supports multi-region datasets such
# as FOR-instance without changing any segmentation hyperparameters.
BUFFER_METRES = 25.0  # Context around each core tile for reducing edge effects.
MIN_HEIGHT = 2.0  # Minimum CHM height used for both detection and crowns.
MEDIAN_FILTER_SIZE = 3  # Pixel window; use a positive odd integer.
CHM_PIT_FILL_SIZE = 1  # Odd pixel window; 1 disables local pit filling.
CHM_PIT_DEPTH = 0.0  # Minimum depression depth replaced by the local median.
CHM_GAUSSIAN_SIGMA = 0.0  # Pixel standard deviation; 0 disables smoothing.

# Variable local-maximum-filter window diameter f_ws(height), in metres:
#   height < LMF_LOW_HEIGHT_LIMIT:  LMF_LOW_WINDOW_SIZE
#   height < LMF_HIGH_HEIGHT_LIMIT: height * LMF_MID_SLOPE + LMF_MID_INTERCEPT
#   otherwise:                      height * LMF_HIGH_SLOPE + LMF_HIGH_INTERCEPT
LMF_LOW_HEIGHT_LIMIT = 10.0
LMF_HIGH_HEIGHT_LIMIT = 25.0
LMF_LOW_WINDOW_SIZE = 2.0
LMF_MID_SLOPE = 0.1
LMF_MID_INTERCEPT = 0.3
LMF_HIGH_SLOPE = 0.15
LMF_HIGH_INTERCEPT = 1.0

# Optional monotonic four-point LMF curve. When all values are supplied on the
# command line, it replaces the legacy three-piece function above. The first
# control height is MIN_HEIGHT and the remaining heights are configurable.
LMF_CONTROL_HEIGHTS: tuple[float, float, float, float] | None = None
LMF_CONTROL_WINDOWS: tuple[float, float, float, float] | None = None

# Marker-controlled watershed and raster-to-polygon settings
WATERSHED_CONNECTIVITY = 8  # Allowed values: 4 or 8; ForestTools uses 8.
POLYGON_CONNECTIVITY = 4  # Allowed values: 4 or 8; ForestTools output uses 4.

# Näslund DBH estimate and output precision. A and B are empirical coefficients.
DBH_PARAMETER_A = 10.0
DBH_PARAMETER_B = 0.6
DBH_REFERENCE_HEIGHT = 1.3  # Breast-height reference in metres.
DBH_DECIMAL_PLACES = 2
# =============================================================================
# END USER-TUNABLE CONFIGURATION
# =============================================================================

# Numerical implementation detail; this is not a model hyperparameter.
FOOTPRINT_DISTANCE_TOLERANCE = 1e-12


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
        default=DEFAULT_INPUT_DIR,
        help=(
            f"Directory containing {CHM_TILE_GLOB}; also requires "
            f"{VRT_FILENAME} unless --independent-tiles is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Base output directory; files are written below "
            f"{SEGMENTATION_SUBDIRECTORY}."
        ),
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
        default=OVERWRITE_EXISTING_OUTPUTS,
        help="Replace existing paired GeoPackage outputs.",
    )
    parser.add_argument(
        "--independent-tiles",
        action="store_true",
        help=(
            "Process every TIFF independently, padding its buffer with NoData "
            "instead of reading 00_CHM_Full.vrt."
        ),
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=MIN_HEIGHT,
        help=f"Minimum CHM height for detection and crowns; default {MIN_HEIGHT:g} m.",
    )
    parser.add_argument(
        "--median-filter-size",
        type=int,
        default=MEDIAN_FILTER_SIZE,
        help=f"Positive odd median-filter window size; default {MEDIAN_FILTER_SIZE}.",
    )
    parser.add_argument(
        "--chm-pit-fill-size",
        type=int,
        default=CHM_PIT_FILL_SIZE,
        help="Odd focal window in pixels for conditional CHM pit filling; 1 disables it.",
    )
    parser.add_argument(
        "--chm-pit-depth",
        type=float,
        default=CHM_PIT_DEPTH,
        help="Minimum depth below the focal median that is treated as a CHM pit.",
    )
    parser.add_argument(
        "--chm-gaussian-sigma",
        type=float,
        default=CHM_GAUSSIAN_SIGMA,
        help="Gaussian smoothing sigma in pixels; 0 disables it.",
    )
    parser.add_argument(
        "--lmf-low-height-limit",
        type=float,
        default=LMF_LOW_HEIGHT_LIMIT,
        help=f"First LMF height breakpoint; default {LMF_LOW_HEIGHT_LIMIT:g} m.",
    )
    parser.add_argument(
        "--lmf-high-height-limit",
        type=float,
        default=LMF_HIGH_HEIGHT_LIMIT,
        help=f"Second LMF height breakpoint; default {LMF_HIGH_HEIGHT_LIMIT:g} m.",
    )
    parser.add_argument(
        "--lmf-low-window-size",
        type=float,
        default=LMF_LOW_WINDOW_SIZE,
        help=f"LMF window diameter below the first breakpoint; default {LMF_LOW_WINDOW_SIZE:g} m.",
    )
    parser.add_argument(
        "--lmf-mid-slope",
        type=float,
        default=LMF_MID_SLOPE,
        help=f"Middle LMF line slope; default {LMF_MID_SLOPE:g}.",
    )
    parser.add_argument(
        "--lmf-mid-intercept",
        type=float,
        default=LMF_MID_INTERCEPT,
        help=f"Middle LMF line intercept; default {LMF_MID_INTERCEPT:g} m.",
    )
    parser.add_argument(
        "--lmf-high-slope",
        type=float,
        default=LMF_HIGH_SLOPE,
        help=f"Upper LMF line slope; default {LMF_HIGH_SLOPE:g}.",
    )
    parser.add_argument(
        "--lmf-high-intercept",
        type=float,
        default=LMF_HIGH_INTERCEPT,
        help=f"Upper LMF line intercept; default {LMF_HIGH_INTERCEPT:g} m.",
    )
    for index in range(1, 4):
        parser.add_argument(
            f"--lmf-control-height-{index}",
            type=float,
            default=None,
            help=f"Height of flexible LMF control point {index}; requires all control values.",
        )
    for index in range(4):
        parser.add_argument(
            f"--lmf-control-window-{index}",
            type=float,
            default=None,
            help=f"Window diameter at flexible LMF control point {index}; requires all control values.",
        )
    parser.add_argument(
        "--watershed-connectivity",
        type=int,
        choices=(4, 8),
        default=WATERSHED_CONNECTIVITY,
        help=f"Watershed neighbourhood connectivity; default {WATERSHED_CONNECTIVITY}.",
    )
    parser.add_argument(
        "--polygon-connectivity",
        type=int,
        choices=(4, 8),
        default=POLYGON_CONNECTIVITY,
        help=f"Raster polygonization connectivity; default {POLYGON_CONNECTIVITY}.",
    )
    return parser.parse_args()


def apply_cli_hyperparameters(args: argparse.Namespace) -> None:
    """Validate and apply optional CLI overrides without changing defaults."""
    global MIN_HEIGHT
    global MEDIAN_FILTER_SIZE
    global CHM_PIT_FILL_SIZE
    global CHM_PIT_DEPTH
    global CHM_GAUSSIAN_SIGMA
    global LMF_LOW_HEIGHT_LIMIT
    global LMF_HIGH_HEIGHT_LIMIT
    global LMF_LOW_WINDOW_SIZE
    global LMF_MID_SLOPE
    global LMF_MID_INTERCEPT
    global LMF_HIGH_SLOPE
    global LMF_HIGH_INTERCEPT
    global LMF_CONTROL_HEIGHTS
    global LMF_CONTROL_WINDOWS
    global WATERSHED_CONNECTIVITY
    global POLYGON_CONNECTIVITY

    numeric_values = {
        "--min-height": args.min_height,
        "--lmf-low-height-limit": args.lmf_low_height_limit,
        "--lmf-high-height-limit": args.lmf_high_height_limit,
        "--lmf-low-window-size": args.lmf_low_window_size,
        "--lmf-mid-slope": args.lmf_mid_slope,
        "--lmf-mid-intercept": args.lmf_mid_intercept,
        "--lmf-high-slope": args.lmf_high_slope,
        "--lmf-high-intercept": args.lmf_high_intercept,
        "--chm-pit-depth": args.chm_pit_depth,
        "--chm-gaussian-sigma": args.chm_gaussian_sigma,
    }
    non_finite = [name for name, value in numeric_values.items() if not math.isfinite(value)]
    if non_finite:
        raise ValueError("Non-finite hyperparameters: " + ", ".join(non_finite))
    if args.min_height < 0:
        raise ValueError("--min-height must be non-negative")
    if args.median_filter_size < 1 or args.median_filter_size % 2 == 0:
        raise ValueError("--median-filter-size must be a positive odd integer")
    if args.chm_pit_fill_size < 1 or args.chm_pit_fill_size % 2 == 0:
        raise ValueError("--chm-pit-fill-size must be a positive odd integer")
    if args.chm_pit_depth < 0:
        raise ValueError("--chm-pit-depth must be non-negative")
    if args.chm_gaussian_sigma < 0:
        raise ValueError("--chm-gaussian-sigma must be non-negative")
    if args.lmf_low_height_limit < 0:
        raise ValueError("--lmf-low-height-limit must be non-negative")
    if args.lmf_high_height_limit <= args.lmf_low_height_limit:
        raise ValueError(
            "--lmf-high-height-limit must exceed --lmf-low-height-limit"
        )
    if args.lmf_low_window_size <= 0:
        raise ValueError("--lmf-low-window-size must be positive")
    middle_windows = (
        args.lmf_low_height_limit * args.lmf_mid_slope
        + args.lmf_mid_intercept,
        args.lmf_high_height_limit * args.lmf_mid_slope
        + args.lmf_mid_intercept,
    )
    upper_window = (
        args.lmf_high_height_limit * args.lmf_high_slope
        + args.lmf_high_intercept
    )
    if min(*middle_windows, upper_window) <= 0:
        raise ValueError("The configured LMF window function must stay positive")

    control_heights = tuple(
        getattr(args, f"lmf_control_height_{index}") for index in range(1, 4)
    )
    control_windows = tuple(
        getattr(args, f"lmf_control_window_{index}") for index in range(4)
    )
    supplied = [value is not None for value in (*control_heights, *control_windows)]
    if any(supplied) and not all(supplied):
        raise ValueError("all flexible LMF control heights and windows are required")
    if all(supplied):
        heights = (args.min_height, *(float(value) for value in control_heights))
        windows = tuple(float(value) for value in control_windows)
        if any(not math.isfinite(value) for value in (*heights, *windows)):
            raise ValueError("flexible LMF control values must be finite")
        if any(right <= left for left, right in zip(heights, heights[1:])):
            raise ValueError("flexible LMF control heights must be strictly increasing")
        if any(value <= 0 for value in windows):
            raise ValueError("flexible LMF control windows must be positive")
        if any(right < left for left, right in zip(windows, windows[1:])):
            raise ValueError("flexible LMF control windows must be non-decreasing")
        LMF_CONTROL_HEIGHTS = heights
        LMF_CONTROL_WINDOWS = windows
    else:
        LMF_CONTROL_HEIGHTS = None
        LMF_CONTROL_WINDOWS = None

    MIN_HEIGHT = args.min_height
    MEDIAN_FILTER_SIZE = args.median_filter_size
    CHM_PIT_FILL_SIZE = args.chm_pit_fill_size
    CHM_PIT_DEPTH = args.chm_pit_depth
    CHM_GAUSSIAN_SIGMA = args.chm_gaussian_sigma
    LMF_LOW_HEIGHT_LIMIT = args.lmf_low_height_limit
    LMF_HIGH_HEIGHT_LIMIT = args.lmf_high_height_limit
    LMF_LOW_WINDOW_SIZE = args.lmf_low_window_size
    LMF_MID_SLOPE = args.lmf_mid_slope
    LMF_MID_INTERCEPT = args.lmf_mid_intercept
    LMF_HIGH_SLOPE = args.lmf_high_slope
    LMF_HIGH_INTERCEPT = args.lmf_high_intercept
    WATERSHED_CONNECTIVITY = args.watershed_connectivity
    POLYGON_CONNECTIVITY = args.polygon_connectivity


def dbh_naslund(
    height: np.ndarray,
    a: float = DBH_PARAMETER_A,
    b: float = DBH_PARAMETER_B,
) -> np.ndarray:
    """Estimate diameter at breast height using the formula from the R script."""
    hm = np.maximum(height - DBH_REFERENCE_HEIGHT, 0.0)
    coefficient_a = -hm * b
    coefficient_c = -hm * a
    discriminant = coefficient_a**2 - 4.0 * coefficient_c
    return (-coefficient_a + np.sqrt(discriminant)) / 2.0


def moving_window_size(height: np.ndarray | float) -> np.ndarray | float:
    """Variable LMF window diameter, in metres, matching ``f_ws`` in R."""
    if LMF_CONTROL_HEIGHTS is not None and LMF_CONTROL_WINDOWS is not None:
        values = np.asarray(height)
        interpolated = np.interp(values, LMF_CONTROL_HEIGHTS, LMF_CONTROL_WINDOWS)
        return float(interpolated) if np.ndim(values) == 0 else interpolated
    return np.where(
        np.asarray(height) < LMF_LOW_HEIGHT_LIMIT,
        LMF_LOW_WINDOW_SIZE,
        np.where(
            np.asarray(height) < LMF_HIGH_HEIGHT_LIMIT,
            np.asarray(height) * LMF_MID_SLOPE + LMF_MID_INTERCEPT,
            np.asarray(height) * LMF_HIGH_SLOPE + LMF_HIGH_INTERCEPT,
        ),
    )


def read_buffered_chm(
    chm_file: Path, vrt_path: Path | None
) -> tuple[
    np.ndarray,
    rasterio.Affine,
    rasterio.coords.BoundingBox,
    rasterio.crs.CRS,
]:
    """Read the configured buffer from a VRT or pad one independent tile."""
    with rasterio.open(chm_file) as core_source:
        core_bounds = core_source.bounds
        core_crs = core_source.crs

    if core_crs is None:
        raise ValueError(f"CHM has no CRS: {chm_file.name}")
    if not core_crs.is_projected:
        raise ValueError(f"CHM CRS must be projected: {chm_file.name} ({core_crs})")
    unit_name, unit_factor = core_crs.linear_units_factor
    if not math.isclose(float(unit_factor), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"CHM CRS must use metre units: {chm_file.name} "
            f"({core_crs}, {unit_name})"
        )

    buffered_bounds = (
        core_bounds.left - BUFFER_METRES,
        core_bounds.bottom - BUFFER_METRES,
        core_bounds.right + BUFFER_METRES,
        core_bounds.top + BUFFER_METRES,
    )

    raster_path = vrt_path if vrt_path is not None else chm_file
    with rasterio.open(raster_path) as raster_source:
        if raster_source.crs != core_crs:
            raise ValueError(
                f"CRS mismatch between {chm_file.name} ({core_crs}) and "
                f"{raster_path.name} ({raster_source.crs})"
            )
        window = from_bounds(*buffered_bounds, transform=raster_source.transform)
        window = window.round_offsets().round_lengths()
        chm = raster_source.read(
            CHM_BAND_INDEX,
            window=window,
            masked=True,
            boundless=vrt_path is None,
        )
        transform = raster_source.window_transform(window)

    return chm.filled(np.nan).astype(np.float32), transform, core_bounds, core_crs


def improve_chm(chm: np.ndarray) -> np.ndarray:
    """Conditionally fill canopy pits and smooth CHM values without GT data."""
    improved = chm.astype(np.float64, copy=True)
    if CHM_PIT_FILL_SIZE > 1:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            local_median = generic_filter(
                improved,
                np.nanmedian,
                size=CHM_PIT_FILL_SIZE,
                output=np.float64,
                mode="constant",
                cval=np.nan,
            )
        finite_count = convolve(
            np.isfinite(improved).astype(np.int16),
            np.ones((CHM_PIT_FILL_SIZE, CHM_PIT_FILL_SIZE), dtype=np.int16),
            mode="constant",
            cval=0,
        )
        required = math.ceil(0.60 * CHM_PIT_FILL_SIZE**2)
        depression = local_median - improved
        pits = (
            np.isfinite(local_median)
            & (local_median >= MIN_HEIGHT)
            & (finite_count >= required)
            & (~np.isfinite(improved) | (depression >= CHM_PIT_DEPTH))
        )
        improved[pits] = local_median[pits]

    if CHM_GAUSSIAN_SIGMA > 0:
        valid = np.isfinite(improved)
        weighted_values = gaussian_filter(
            np.where(valid, improved, 0.0),
            sigma=CHM_GAUSSIAN_SIGMA,
            mode="nearest",
        )
        weights = gaussian_filter(
            valid.astype(np.float64),
            sigma=CHM_GAUSSIAN_SIGMA,
            mode="nearest",
        )
        smoothed = np.divide(
            weighted_values,
            weights,
            out=np.full_like(weighted_values, np.nan),
            where=weights > 1e-12,
        )
        improved[valid] = smoothed[valid]

    return improved


def smooth_chm(chm: np.ndarray) -> np.ndarray:
    """Improve the CHM, then apply Terra-compatible median smoothing."""
    improved = improve_chm(chm)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return generic_filter(
            improved,
            np.nanmedian,
            size=MEDIAN_FILTER_SIZE,
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
    return squared_distance <= radius**2 + FOOTPRINT_DISTANCE_TOLERANCE


def locate_trees_lmf(
    chm: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Reproduce lidR's variable circular local-maximum filter for a raster."""
    valid = np.isfinite(chm)
    eligible = valid & (chm >= MIN_HEIGHT)
    if not np.any(eligible):
        return empty_treetops(crs), np.zeros(chm.shape, dtype=np.int32)

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
            <= radius**2 + FOOTPRINT_DISTANCE_TOLERANCE
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
        return empty_treetops(crs), np.zeros(chm.shape, dtype=np.int32)

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
        crs=crs,
    )
    return treetops, marker_raster


def empty_treetops(crs: rasterio.crs.CRS) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"treeID": np.asarray([], dtype=np.int32), "Z": np.asarray([], dtype=float)},
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )


def cimg_watershed(marker_raster: np.ndarray, priority: np.ndarray) -> np.ndarray:
    """Port CImg's priority watershed used by ``imager``.

    ``imager::watershed`` passes its third argument (named ``fill_lines`` in
    R) directly to CImg as ``is_high_connectivity``. ForestTools uses the
    default ``TRUE``, which means eight-neighbour connectivity in 2D.
    """
    output = marker_raster.astype(np.int32, copy=True)
    queued_labels = np.zeros(output.shape, dtype=np.uint32)
    seed_locations: list[tuple[int, int]] = []
    priority_queue: list[list[float | int]] = []
    row_count, col_count = output.shape
    four_neighbours = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    )
    diagonal_neighbours = (
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    )
    if WATERSHED_CONNECTIVITY == 8:
        neighbours = four_neighbours + diagonal_neighbours
    elif WATERSHED_CONNECTIVITY == 4:
        neighbours = four_neighbours
    else:
        raise ValueError("WATERSHED_CONNECTIVITY must be 4 or 8")

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
    crs: rasterio.crs.CRS,
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
        connectivity=POLYGON_CONNECTIVITY,
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
        crs=crs,
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
    temporary_treetops = treetop_path.with_name(
        f".{treetop_path.stem}.{token}{OUTPUT_EXTENSION}"
    )
    temporary_crowns = crown_path.with_name(
        f".{crown_path.stem}.{token}{OUTPUT_EXTENSION}"
    )
    try:
        treetops.to_file(
            temporary_treetops,
            layer=treetop_path.stem,
            driver=OUTPUT_DRIVER,
            engine=OUTPUT_ENGINE,
            index=False,
        )
        crowns.to_file(
            temporary_crowns,
            layer=crown_path.stem,
            driver=OUTPUT_DRIVER,
            engine=OUTPUT_ENGINE,
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
    vrt_path: Path | None,
    segmentation_dir: Path,
    overwrite: bool,
) -> TileResult:
    tile_id = chm_file.stem.removeprefix(CHM_TILE_PREFIX)
    crown_path = segmentation_dir / (
        f"{CROWN_OUTPUT_PREFIX}{tile_id}{OUTPUT_EXTENSION}"
    )
    treetop_path = segmentation_dir / (
        f"{TREETOP_OUTPUT_PREFIX}{tile_id}{OUTPUT_EXTENSION}"
    )

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
        chm, transform, core_bounds, chm_crs = read_buffered_chm(chm_file, vrt_path)
        finite = chm[np.isfinite(chm)]
        if finite.size == 0 or float(np.max(finite)) < MIN_HEIGHT:
            return TileResult(
                tile_id,
                "empty",
                detail=f"no CHM values at least {MIN_HEIGHT:g} m",
            )

        smoothed = smooth_chm(chm)
        all_treetops, markers = locate_trees_lmf(smoothed, transform, chm_crs)
        if all_treetops.empty:
            return TileResult(tile_id, "empty", detail="no local maxima detected")

        core_mask = inside_core_mask(all_treetops, core_bounds)
        final_treetops = all_treetops.loc[core_mask].copy()
        if final_treetops.empty:
            return TileResult(tile_id, "empty", detail="no tree tops inside core tile")

        final_treetops["dbh"] = np.round(
            dbh_naslund(final_treetops["Z"].to_numpy()), DBH_DECIMAL_PLACES
        )
        core_tree_ids = set(final_treetops["treeID"].astype(int))
        final_crowns = segment_crowns(
            smoothed,
            transform,
            markers,
            core_tree_ids,
            chm_crs,
        )

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
    try:
        apply_cli_hyperparameters(args)
    except ValueError as error:
        print(f"Błąd parametrów: {error}", file=sys.stderr)
        return 2
    input_dir = args.input_dir.resolve()
    segmentation_dir = args.output_dir.resolve() / SEGMENTATION_SUBDIRECTORY
    vrt_path = None if args.independent_tiles else input_dir / VRT_FILENAME
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    chm_files = sorted(input_dir.glob(CHM_TILE_GLOB))
    if args.tile:
        requested = set(args.tile)
        chm_files = [
            path for path in chm_files
            if path.stem.removeprefix(CHM_TILE_PREFIX) in requested
        ]
        found = {
            path.stem.removeprefix(CHM_TILE_PREFIX) for path in chm_files
        }
        missing = requested - found
        if missing:
            print(f"Błąd: nie znaleziono kafli: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2

    if not chm_files:
        print(f"Błąd: Brak plików CHM. Sprawdź folder: {input_dir}", file=sys.stderr)
        return 2
    if vrt_path is not None and not vrt_path.is_file():
        print(f"Błąd: Brak pliku VRT: {vrt_path}", file=sys.stderr)
        return 2

    print(">>> C1. Start Segmentacji (Python)...")
    print(
        "Hyperparameters: "
        f"min_height={MIN_HEIGHT:g}, median_filter={MEDIAN_FILTER_SIZE}, "
        f"CHM pit=({CHM_PIT_FILL_SIZE}, {CHM_PIT_DEPTH:g}), "
        f"CHM gaussian={CHM_GAUSSIAN_SIGMA:g}, "
        f"LMF breaks=({LMF_LOW_HEIGHT_LIMIT:g}, {LMF_HIGH_HEIGHT_LIMIT:g}), "
        f"LMF low={LMF_LOW_WINDOW_SIZE:g}, "
        f"LMF mid={LMF_MID_SLOPE:g}*h+{LMF_MID_INTERCEPT:g}, "
        f"LMF high={LMF_HIGH_SLOPE:g}*h+{LMF_HIGH_INTERCEPT:g}, "
        f"LMF controls={list(zip(LMF_CONTROL_HEIGHTS, LMF_CONTROL_WINDOWS)) if LMF_CONTROL_HEIGHTS else 'legacy'}, "
        f"watershed={WATERSHED_CONNECTIVITY}, polygon={POLYGON_CONNECTIVITY}"
    )
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
